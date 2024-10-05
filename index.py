from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.index import lecture_material

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lecture_material)