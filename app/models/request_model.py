from pydantic import BaseModel

class DetectionRequest(BaseModel):
    video_path: str

class BodyAttributes(BaseModel):
    image_path:str
