from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch
import numpy as np
import cv2
import os
from datetime import datetime
from ultralytics import YOLO

app = FastAPI()

MODEL_PATH = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\yolo12n.pt."  
SAVE_DIR = "saved_detections"
os.makedirs(SAVE_DIR, exist_ok=True)

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

class Item(BaseModel):
    name: str
    price: float | None = None
    discount: float | None = None

def save_detection(crop, track_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"human_{track_id}_{timestamp}.jpg")
    cv2.imwrite(path, crop)
    return path

def detect_and_track(video_path: str):
    tracked_ids = set()
    for results in model.track(
        source=video_path,
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

                center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                inside_intrusion = cv2.pointPolygonTest(roi_intrusion, center_point, False) >= 0

                if inside_intrusion and track_id not in tracked_ids:
                    tracked_ids.add(track_id)
                    crop = frame[y1:y2, x1:x2]
                    obj_path = save_detection(crop, track_id)
                    print(f"[SAVED] {class_name} (ID:{track_id}) at {obj_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

@app.post("/detect/{user_video_path}")
async def detect_people(request: DetectionRequest, background_tasks: BackgroundTasks,user_video_path: str):
    background_tasks.add_task(detect_and_track, request.video_path)
    response = {"message":f"Detection staded for {request.video_path}"}
    return {"message": f"Detection started for {request.video_path}"}

@app.get('/test/{nameUsername}')
async def root( nameUsername: str):

    return {f" message": "hello world this is test detection api "+nameUsername}

@app.post('/test/post/')
async def get_post(item: Item):
    return {'name': item.name, 'price': item.price, 'discount': item.discount}


