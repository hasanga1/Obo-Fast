from pydantic import BaseModel

class LectureMaterialSchema(BaseModel):
    # file: bytes 
    file_name: str 
    file_type: str
