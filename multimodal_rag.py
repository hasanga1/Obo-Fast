import os
import whisper
from langchain.schema import Document
import fitz
import google.generativeai as genai
import PIL.Image
import pandas as pd
import cv2
from PIL import Image
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load environment variables from .env file
load_dotenv()


######################### AUDIO ##########################

def transcribe_audio_files(files_with_names, course, subject):
    documents = []

    model = whisper.load_model("base")

    for file, file_name in files_with_names:
        # Convert the NumPy array back to a suitable format for Whisper if needed
        # Here, we assume that the model can process raw audio data directly
        result = model.transcribe(file)
        transcription = result["text"]
        file_id, file_name = file_name.split('-', 1)

        document = Document(
            page_content=transcription,
            metadata={"course": course, "subject": subject, "format": "audio", "source": file_name, "id": int(file_id)}  # Adjust source if needed
        )

        documents.append(document)

    return documents

######################### IMAGES ##########################

def extract_images_from_pdf(pdf_path, output_dir):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the PDF's base name without the extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Iterate through each page in the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{pdf_name}.pdf_{page_number+1}_img_{img_index+1}.{image_ext}"

            # Save the extracted image to the output directory
            with open(os.path.join(output_dir, image_filename), "wb") as img_file:
                img_file.write(image_bytes)

    pdf_document.close()


def process_all_pdfs(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"Processing {pdf_path}")
            extract_images_from_pdf(pdf_path, output_dir)


# Step 1: Generate Captions for All Images in Output Directory
def generate_captions_for_images(output_dir):
    genai.configure(api_key=os.getenv('GOOGLE_API_'))
    model = genai.GenerativeModel('gemini-1.5-flash')

    captions = []
    i = 0
    # Iterate through each image file in the output directory
    for image_filename in sorted(os.listdir(output_dir)):
        image_path = os.path.join(output_dir, image_filename)
        i += 1

        # Ensure it's a file (not a subdirectory) and an image
        if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img = PIL.Image.open(image_path)
            response = model.generate_content(img)
            caption = response.text.strip()  # Clean up the caption text

            # Store the caption along with image reference
            captions.append((image_filename, caption))
        if i >= 15:
            break
    return captions

# Step 2: Create a Document with All Captions
def create_documents_from_captions(captions, course, subject):
    documents = []

    for image_filename, caption in captions:
        # Extracting the source file name and page number from the filename
        # Assuming the filename format is "source_file_page_1000_img_1.png"
        parts = image_filename.split('_')
        source_file = '_'.join(parts[:-3])  # This extracts the source file name
        page_number = parts[-3]  # This extracts '1000' from 'page_1000_img_1.png'
        img_number = parts[-1].split('.')[0]  # This extracts '1' from 'page_1000_img_1.png'
        file_id, file_name = source_file.split('-', 1)

        document = Document(
            page_content=caption,
            metadata={"course": course, "subject": subject, "format": "image", "source": file_name, "page": int(page_number), "img": int(img_number), "id": int(file_id)}
        )
        documents.append(document)

    return documents

######################### VIDEOS ##########################

def extract_frames_from_video(video_path, output_dir, desired_fps=None):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the original FPS of the video
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the exact number of frames to skip to match the desired FPS
    frame_interval = original_fps / desired_fps if desired_fps else 1

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the video filename without extension
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    frame_count = 0
    extracted_count = 0
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Calculate timestamp for the current frame
        timestamp = frame_count / original_fps  # in seconds

        # Extract frame based on the interval
        if frame_count >= extracted_count * frame_interval:
            # Convert frame to RGB (PIL expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            img = Image.fromarray(frame_rgb)

            # Save the frame as an image file with the desired format
            frame_filename = f"{video_filename}_time_{timestamp:.2f}_frame_{extracted_count:04d}.png"
            img.save(os.path.join(output_dir, frame_filename))
            extracted_count += 1

        frame_count += 1

    video_capture.release()
    print(f"Extracted {extracted_count} frames from {video_path} at {desired_fps or original_fps} FPS")


def transcribe_frames(input_dir):
    model_name = "gemini-1.5-flash"
    genai.configure(api_key=os.getenv('GOOGLE_API_'))
    model = genai.GenerativeModel(model_name)

    transcriptions = {}
    i =0

    # Iterate over all image files in the directory
    for filename in sorted(os.listdir(input_dir)):
        i += 1
        if filename.endswith(".png"):
            file_path = os.path.join(input_dir, filename)

            # Open the image
            img = PIL.Image.open(file_path)

            # Generate content
            response = model.generate_content(img)
            transcription = response.text

            # Store the transcription
            transcriptions[filename] = transcription

        if i >= 15:
            break

    return transcriptions


def create_documents_from_frames(captions, course, subject):
    documents = []

    for image_filename, caption in captions.items():
        # Extracting the source file name, timestamp, and frame number from the filename
        # Assuming the filename format is "video_filename_time_XX.XX_frame_XXXX.png"
        parts = image_filename.split('_')
        video_filename = parts[0]  # Extract the source video file name
        timestamp = parts[-3]  # Extract timestamp
        frame_number = parts[-1].split('.')[0]  # Extract frame number
        file_id, file_name = os.path.basename(image_filename).split('-', 1)

        document = Document(
            page_content=caption,
            metadata={
                "course": course, 
                "subject": subject, 
                "format": "video",
                "source": file_name,
                "timestamp": timestamp,
                "frame": int(frame_number),
                "id": int(file_id)
            }
        )
        documents.append(document)

    return documents


def process_videos_in_directory(input_dir, output_dir_base, frame_rate, course, subject):
    documents=[]
    # Iterate over all video files in the input directory
    for video_filename in os.listdir(input_dir):
        if video_filename.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Add other video formats if needed
            video_path = os.path.join(input_dir, video_filename)

            # Create a specific output directory for each video
            video_output_dir = os.path.join(output_dir_base, os.path.splitext(video_filename)[0])

            # Extract frames from the video
            extract_frames_from_video(video_path, video_output_dir, frame_rate)

            # Transcribe the frames
            transcriptions = transcribe_frames(video_output_dir)

            # Create documents with the required format
            document = create_documents_from_frames(transcriptions, course, subject)
            for doc in document:
                documents.append(doc)

    return documents

######################### DOCS ##########################


from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, UnstructuredEPubLoader, UnstructuredExcelLoader, NotebookLoader, PythonLoader, SQLDatabaseLoader, UnstructuredXMLLoader
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define loaders for different file types

def text_preprocess(input_dir):
    pdf_loader = PyPDFDirectoryLoader(input_dir)
    word_loader = DirectoryLoader(input_dir, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    text_loader = DirectoryLoader(input_dir, glob="*.txt", loader_cls=TextLoader)
    html_loader = DirectoryLoader(input_dir, glob="*.html", loader_cls=UnstructuredHTMLLoader)
    markdown_loader = DirectoryLoader(input_dir, glob="*.c", loader_cls=UnstructuredMarkdownLoader)
    epub_loader = DirectoryLoader(input_dir, glob="*.epub", loader_cls=UnstructuredEPubLoader)
    powerpoint_loader = DirectoryLoader(input_dir, glob="*.pptx", loader_cls=UnstructuredPowerPointLoader)
    csv_loader = DirectoryLoader(input_dir, glob="*.csv", loader_cls=CSVLoader)
    excel_loader = DirectoryLoader(input_dir, glob="*.xlsx", loader_cls=UnstructuredExcelLoader)
    notebook_loader = DirectoryLoader(input_dir, glob="*.ipynb", loader_cls=NotebookLoader)
    python_loader = DirectoryLoader(input_dir, glob="*.py", loader_cls=PythonLoader)
    xml_loader = DirectoryLoader(input_dir, glob="*.xml", loader_cls=UnstructuredXMLLoader)

    # Load documents from all loaders
    pdf_documents = pdf_loader.load()
    word_documents = word_loader.load()
    text_documents = text_loader.load()
    html_documents = html_loader.load()
    markdown_documents = markdown_loader.load()
    epub_documents = epub_loader.load()
    powerpoint_documents = powerpoint_loader.load()
    csv_documents = csv_loader.load()
    excel_documents = excel_loader.load()
    notebook_documents = notebook_loader.load()
    python_documents = python_loader.load()
    xml_documents = xml_loader.load()

    # Combine all documents
    all_documents = (
        pdf_documents + word_documents + text_documents + html_documents +
        markdown_documents + epub_documents + powerpoint_documents +
        csv_documents + excel_documents + notebook_documents +
        python_documents + xml_documents
    )

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(all_documents)

    return documents



def update_metadata(documents, course, subject):
    updated_documents = []

    for doc in documents:
        # Access existing metadata
        updated_metadata = doc.metadata.copy()

        # Extract the filename from the file path
        file_path = doc.metadata.get("source", "")
        file_id, file_name = os.path.basename(file_path).split('-', 1)

        # Modify metadata as needed
        updated_metadata['course'] = course 
        updated_metadata['subject'] = subject 
        updated_metadata['format'] = 'text'
        updated_metadata['source'] = file_name
        updated_metadata['page'] = doc.metadata.get("page", "")
        updated_metadata['id'] = int(file_id)


        # Create a new document with updated metadata
        updated_doc = Document(page_content=doc.page_content, metadata=updated_metadata)
        updated_documents.append(updated_doc)

    return updated_documents


def save_doc(documents):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    index_name = os.getenv("PINECONE_INDEX")

    pinecone = PineconeVectorStore.from_documents(
        documents, embeddings, index_name=index_name
    )

    # Initialize Pinecone with your API key
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY")
    )

    # Connect to the index
    index = pc.describe_index(name=index_name)