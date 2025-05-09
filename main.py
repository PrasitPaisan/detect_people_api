from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import asyncio
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import os

from app.utils.save_image_util import save_detection, save_image_to_googleStorage
from app.utils.generate_data_util import generate_data
from app.utils.image_processing_util import extract_detected_objects, draw_detection, find_closest_vehicle
from app.models.request_model import DetectionRequest

app = FastAPI()
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
crop_path = os.getenv("CROP_GOOGLE_DRIVE_PATH")
overview_path = os.getenv("OVERVIEW_GOOGLE_DRIVE_PATH")
path_to_private_key = os.getenv("PRIVATE_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_private_key

roi_points = [
    [268.23, 387.93],
    [1419.42, 260.64],
    [1853.65, 376.83],
    [1859.71, 1103.19],
    [342.96, 1119.35]
]
roi_intrusion = np.array(roi_points, dtype=np.int32)
model = YOLO(MODEL_PATH, verbose=False)

@app.post("/detect/stream")
async def detect_stream(request: DetectionRequest):

    async def event_generator():
        tracked_ids = set()
        for results in model.track(
            source=request.video_path,
            stream=True,
            save=False,
            show=False,
            classes=[0, 1, 3],
            conf=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=True
        ):
            frame = results.orig_img
            showframe = frame.copy()

            if results.boxes is not None:

                persons, vehicles = extract_detected_objects(results, roi_intrusion)
                used_vehicle_ids = set()

                for person in persons:
                    closest_vehicle, pid, dist = find_closest_vehicle(person, vehicles, used_vehicle_ids, 200)
                    if closest_vehicle:
                        vid = closest_vehicle["id"]
                        if pid not in tracked_ids and vid not in tracked_ids:
                            #update id used
                            tracked_ids.update({pid, vid})
                            used_vehicle_ids.add(vid)
                            #crop and draw
                            timestamp = datetime.now().strftime("%H%M%S")
                            snapshot = draw_detection(frame, showframe, roi_intrusion, person["box"], closest_vehicle["box"])
                            #save image
                            save_detection(snapshot, showframe, pid, timestamp)
                            snap_path = save_image_to_googleStorage(snapshot, "opp-intrusion-image", "staging/helmet/snapshot", f"snapshot_{pid}_{timestamp}")
                            ovew_path = save_image_to_googleStorage(showframe, "opp-intrusion-image", "staging/helmet/overview_image", f"frame_{pid}_{timestamp}")

                            print(f"Saved pair: person ID {pid}, vehicle ID {vid} distance : {dist}")
                            data = generate_data(snap_path, ovew_path)

                            yield f"data: {data}\n\n"

            await asyncio.sleep(0.1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}