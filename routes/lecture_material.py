from fastapi import APIRouter, HTTPException, File, Form, Response, UploadFile
from config.db import conn
from models.index import lecture_materials
from schemas.index import LectureMaterialSchema
from typing import List
from fastapi.responses import JSONResponse
import shutil
import os
from datetime import datetime

import librosa
import io
import torch
from whisper import load_model, log_mel_spectrogram
from django.conf import settings
import os
import pandas as pd
import shutil
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from multimodal_rag import transcribe_audio_files, process_all_pdfs, generate_captions_for_images, create_documents_from_captions, process_videos_in_directory, text_preprocess, update_metadata, save_doc

lecture_material = APIRouter()

# Retrieve all lecture materials
@lecture_material.get("/")
async def read_data():
    result = conn.execute(lecture_materials.select()).fetchall()
    response = [] 
    for row in result:
        response.append({
            "id": row.id,
            "file_name": row.file_name,
            "file_type": row.file_type,
            "uploaded_at": row.uploaded_at.isoformat()  
        })
    return response

# Retrieve a lecture material by ID
@lecture_material.get("/{id}")
async def read_data(id: int):
    result = conn.execute(lecture_materials.select().where(lecture_materials.c.id == id)).fetchone()
    if result:
        return {
            "id": result.id,
            "file_name": result.file_name,
            "file_type": result.file_type,
            "uploaded_at": result.uploaded_at.isoformat()
        }
    else:
        raise HTTPException(status_code=404, detail="Lecture Material not found")

# Create a new lecture material
UPLOAD_DIRECTORY = "uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@lecture_material.post("/")
async def write_data(
    course: str = Form(...),
    subject: str = Form(...),
    files: List[UploadFile] = File(...)
):
    responses = []

    audio_files = []
    converted_audio_files = []
    pdf_files = []
    video_files = []
    text_files = []

    file_info = []
    for file in files:
        file_info.append({
            "filename": file.filename,
            "course": course,
            "subject": subject
        })

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename, file_extension = os.path.splitext(file.filename)
        new_filename = f"{filename}_{timestamp}{file_extension}"
        file_location = os.path.join(UPLOAD_DIRECTORY, new_filename)
    
        # Open the file location and write the uploaded file content
        with open(file_location, "wb") as f:
            content = file.file.read()  # Read the file content
            f.write(content) 

        insert_stmt = lecture_materials.insert().values(
            file_name=file.filename,
            file_type=file.content_type
        )
        result = conn.execute(insert_stmt)
        conn.connection.commit()

        file_id = result.lastrowid

        file.filename = str(file_id) + "-" + os.path.splitext(file.filename)

        if file.filename.endswith((".m4a", ".mp3", ".webm", ".mp4", ".mpga", ".wav", ".mpeg")):
            audio_files.append((file, file.name))
        elif file.filename.endswith(".pdf"):
            pdf_files.append(file)
            text_files.append(file)
        elif file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_files.append(file)
        elif file.filename.endswith((".docx", ".txt", ".html", ".md", ".c", ".epub", ".pptx", ".csv", ".xlsx", ".ipynb", ".py", ".xml")):
            text_files.append(file)
        else:
            pass

    ############## Audio ###############

    if audio_files:
        for file, file_name in audio_files:
            file_content = file.read()
            audio, sr = librosa.load(io.BytesIO(file_content), sr=None)
            converted_audio_files.append((audio, file_name))

        transcribed_audio_files = transcribe_audio_files(converted_audio_files, course, subject)
    
    ############## Images ###############

    pdf_dir = os.path.join(settings.MEDIA_ROOT, 'pdfs')
    extracted_images_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_images')

    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(extracted_images_dir, exist_ok=True) 

    for file in pdf_files:
        file_path = os.path.join(pdf_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.read())

    process_all_pdfs(pdf_dir, extracted_images_dir)

    captions = generate_captions_for_images(extracted_images_dir)
    pdf_image_captions = create_documents_from_captions(captions, course, subject)


    ############## Video ###############

    video_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
    os.makedirs(video_dir, exist_ok=True)  # Create the videos directory if it doesn't exist

    for file in video_files:
        file_path = os.path.join(video_dir, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.read())

    frame_dir = os.path.join(settings.MEDIA_ROOT, 'video_frames')
    os.makedirs(frame_dir, exist_ok=True) 

    frame_rate = 1
    proccessed_videos = process_videos_in_directory(video_dir, frame_dir, frame_rate, course, subject)
    # print(proccessed_videos)


    ############## Text ###############

    text_files_dir = os.path.join(settings.MEDIA_ROOT, 'textFiles')
    os.makedirs(text_files_dir, exist_ok=True)  # Create the textFiles directory if it doesn't exist

    for file in text_files:
        file_path = os.path.join(text_files_dir, file.name)
        with open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                    f.write(chunk)

    preproccessed_text = text_preprocess(text_files_dir)
    # print(preproccessed_text)

    documents = update_metadata(preproccessed_text, course, subject)

    ############## Update Metadata ###############

    if audio_files:
        for doc in transcribed_audio_files:
            documents.append(doc)

    if pdf_files:
        for doc in pdf_image_captions:
            documents.append(doc)

    if video_files:
        for doc in proccessed_videos:
            documents.append(doc)

    print(len(documents))
    # print(documents)

    ############## Cleanup ###############

    # Remove the directories and their contents
    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)
    if os.path.exists(extracted_images_dir):
        shutil.rmtree(extracted_images_dir)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    if os.path.exists(frame_dir):
        shutil.rmtree(frame_dir)
    if os.path.exists(text_files_dir):
        shutil.rmtree(text_files_dir)

    return JSONResponse(content={"message": "Files uploaded successfully!", "files": file_info})

# Update an existing lecture material by ID
@lecture_material.put("/{id}")
async def update_data(id: int, material: LectureMaterialSchema):
    result = conn.execute(lecture_materials.select().where(lecture_materials.c.id == id)).fetchone()
    if result:
        conn.execute(lecture_materials.update().where(lecture_materials.c.id == id).values(
            file=material.file,  # Update the Binary data
            file_name=material.file_name,
            file_type=material.file_type
        ))
        conn.connection.commit()
        return {"message": "Lecture material updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Lecture Material not found")

pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

# Set your Pinecone index name
INDEX_NAME = os.getenv("PINECONE_INDEX")

# Delete a lecture material by ID
@lecture_material.delete("/{id}")
async def delete_data(id: int):
    result = conn.execute(lecture_materials.select().where(lecture_materials.c.id == id)).fetchone()
    if result:
        conn.execute(lecture_materials.delete().where(lecture_materials.c.id == id))
        conn.connection.commit()

        # Initialize Pinecone index
        index = pc.Index(INDEX_NAME)

        # Delete vector from Pinecone
        try:
            index.delete(filter={"id": id}) 
            return {"message": "Lecture material deleted successfully"}
        except Exception as e:
            return Response({"error": str(e)})   
    else:
        raise HTTPException(status_code=404, detail="Lecture Material not found")
