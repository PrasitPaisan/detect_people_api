from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import cv2
import numpy as np
from datetime import datetime
import uuid
from pydantic import BaseModel
from ultralytics import YOLO
import os
import math

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


app = FastAPI()

MODEL_PATH = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\yolo12s.pt"
roi_points = [
    [268.23, 387.93],
    [1419.42, 260.64],
    [1853.65, 376.83],
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
    print("save overview to",overview_path)
    cv2.imwrite(obj_path, crop)
    cv2.imwrite(overview_path, frame)

    return  obj_path, overview_path


@app.post("/detect/stream")
async def detect_stream(request: DetectionRequest):
    async def event_generator():
        tracked_ids = set()
        for results in model.track(
            source=request.video_path,
            stream=True,
            save=False,
            show=True,
            classes=[0,1,3],
            conf=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=True
        ):
            frame = results.orig_img
            showframe = frame.copy()

            if results.boxes is not None:
                persons = []
                vehicles = []

                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().item())
                    track_id = int(box.id[0].cpu().item()) if box.id is not None else None
                    conf = float(box.conf[0].cpu().item())
                    class_name = model.names[cls_id].lower()

                    center = (int((x1+x2)/2),int((y1+y2)/2))
                    inside = cv2.pointPolygonTest(roi_intrusion, center, False) >= 0

                    if not inside:
                        continue

                    cv2.rectangle(showframe, (x1,y1),(x2,y2), (0,255,0), 2)
                    cv2.putText(showframe,f"{class_name} : {conf:.2f}",(x1, y1 - 10 ),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    if cls_id == 0:
                        persons.append({"box":(x1,y1,x2,y2), "id": track_id})
                    if cls_id in {1,3}:
                        vehicles.append({"box":(x1,y1,x2,y2), "id": track_id})

                used_vehicle_ids = set()

                for person in persons:
                    px1, py1, px2, py2 = person["box"]
                    pid = person["id"]
                    p_center = (int((px1+px2)/2), int((py1+py2)/2))

                    closest_vehicle = None
                    min_dis = float("inf")

                    for vahicle in vehicles:
                        vid = vahicle["id"]

                        if vid in used_vehicle_ids:
                            continue

                        vx1, vy1, vx2, vy2 = vahicle["box"]
                        v_center = (int((vx1+vx2)/2), int((vy1+vy2)/2))
                        dist = math.sqrt((p_center[0] - v_center[0]) ** 2 + (p_center[1] - v_center[1]) ** 2)
                       

                        if dist < min_dis and dist < 400 :
                            min_dis = dist
                            closest_vehicle = vahicle

                    if closest_vehicle:
                        vx1, vy1, vx2, vy2 = closest_vehicle["box"]
                        vid = closest_vehicle["id"]

                        if pid not in tracked_ids and vid not in tracked_ids:
                            x1 = min(px1, vx1)
                            y1 = min(py1, vy1)
                            x2 = max(px2, vx2)
                            y2 = max(py2, vy2)

                            cv2.polylines(showframe, [roi_intrusion], isClosed=True, color=(0, 0, 255), thickness=2)
                            cv2.rectangle(showframe, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(showframe, f"Rider", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                            crop = frame[y1:y2, x1:x2]
                            timestamp = datetime.now().strftime("%H%M%S")
                            save_detection(crop, showframe,pid,timestamp)
                            tracked_ids.update({pid, vid})
                            used_vehicle_ids.add(vid)
                            print(f"Saved pair: person ID {pid}, vehicle ID {vid} distance : {dist}")

                            data = {
                                "helmet_detection_log_uuid": str(uuid.uuid4()),
                                "ai_camera_uuid": str(uuid.uuid4()),
                                "snapshot_image": "b64_image",
                                "overview_image": "overview_image",
                                "body_attribute": {}
                            }

                            yield f"data: {data}\n\n"

            await asyncio.sleep(0.1)


    return StreamingResponse(event_generator(), media_type="text/event-stream")