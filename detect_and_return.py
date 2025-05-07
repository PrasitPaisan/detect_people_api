from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import base64
import cv2
import torch
import numpy as np
from datetime import datetime
import uuid
from pydantic import BaseModel
from ultralytics import YOLO
import os

app = FastAPI()

MODEL_PATH = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\yolo12n.pt"
roi_points = [
    [268.23, 387.93],
    [1419.42, 260.64],
    [1853.65, 476.83],
    [1859.71, 1103.19],
    [342.96, 1119.35]
]
roi_intrusion = np.array(roi_points, dtype=np.int32)
model = YOLO(MODEL_PATH, verbose=False)

class DetectionRequest(BaseModel):
    video_path: str

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def save_detection(crop, frame, track_id, timestamp):
    saved_dir = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api"
    os.makedirs(os.path.join(saved_dir, "saved_detections"), exist_ok=True)
    os.makedirs(os.path.join(saved_dir, "overview_detections"), exist_ok=True)

    obj_path = os.path.join(saved_dir, "saved_detections", f"crop_{track_id}_{timestamp}.jpg")
    overview_path = os.path.join(saved_dir, "overview_detections", f"frame_{track_id}_{timestamp}.jpg")
    print("save detection to",obj_path)
    print("save overview to",obj_path)
    cv2.imwrite(obj_path, crop)
    cv2.imwrite(overview_path, frame)


@app.post("/detect/stream")
async def detect_stream(request: DetectionRequest):
    async def event_generator():
        tracked_ids = set()
        for results in model.track(
            source=request.video_path,
            stream=True,
            save=False,
            show=False,
            classes=0,
            conf=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=True
        ):
            frame = results.orig_img
            showframe = frame.copy()

            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().item())
                    track_id = int(box.id[0].cpu().item()) if box.id is not None else None
                    class_name = model.names[cls_id]

                    if class_name.lower() != "person":
                        continue

                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if cv2.pointPolygonTest(roi_intrusion, center, False) >= 0 and track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        crop = frame[y1:y2, x1:x2]
                        overview_image = encode_image_to_base64(showframe)
                        b64_image = encode_image_to_base64(crop)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        save_detection(crop, showframe, track_id=track_id, timestamp=timestamp)
                        data = {
                            "helmet_detection_log_uuid": str(uuid.uuid4()),
                            "ai_camera_uuid": str(uuid.uuid4()),
                            "snapshot_image": "b64_image",
                            "overview_image": "overview_image",
                            "body_attribute": {}
                        }

                        # ส่งข้อมูลแบบ stream line by line
                        yield f"data: {data}\n\n"

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
