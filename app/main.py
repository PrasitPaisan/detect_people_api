# app/main.py
from fastapi import APIRouter
from .models.request_model import DetectionRequest
from .services.detection_service import detect_stream

router = APIRouter()

@router.post("/detect/stream")
async def detect_stream_api(request: DetectionRequest):
    return await detect_stream(request)
