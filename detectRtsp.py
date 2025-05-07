import torch
print("GPU CUDA is", torch.cuda.is_available())
import numpy as np

from ultralytics import YOLO
import cv2
import os
from datetime import datetime

MODEL_PATH = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\yolo12s.pt"
VIDEO_PATH = r"rtsp://mioc_cms:Mi0C2023Cms@10.54.2.2:554/Streaming/Channels/5301"
SAVE_DIR = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\saved_detections"
os.makedirs(SAVE_DIR, exist_ok=True)

roi_points = [
    [268.2335034687419, 387.935523020233],
    [1419.4285865910601, 260.64417952921906],
    [1853.651293382812, 300.83741371236977],
    [1859.7102148729293, 1100.1916435887877],
    [342.96020184685733, 1119.3556237146306]
]

roi_intrusion = np.array(roi_points, dtype=np.int32)


def load_model(path):
    return YOLO(path, verbose=False)

def save_detection(crop, track_id, showframe):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_path = os.path.join(SAVE_DIR, f"human_{track_id}_{timestamp}.jpg")
    frame_path = os.path.join(SAVE_DIR, f"frame_{track_id}_{timestamp}.jpg")
    cv2.imwrite(object_path, crop)
    cv2.imwrite(frame_path, showframe)

    return object_path, frame_path

def main():
    model = load_model(MODEL_PATH)
    tracked_ids = set()

    for results in model.track(
        source=VIDEO_PATH,
        stream=True,
        save=False,
        show=True,
        classes=[0, 1, 3],  # person, bicycle, motorcycle
        conf=0.5,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
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

                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                inside = cv2.pointPolygonTest(roi_intrusion, center, False) >= 0
                if not inside:
                    continue
                cv2.rectangle(showframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(showframe, f"{class_name} : {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if class_name == "person":
                    persons.append({"box": (x1, y1, x2, y2), "id": track_id})
                elif class_name in {"bicycle", "motorcycle"}:
                    vehicles.append({"box": (x1, y1, x2, y2), "id": track_id})

            used_vehicle_ids = set()

            for person in persons:
                px1, py1, px2, py2 = person["box"]
                pid = person["id"]
                p_center = ((px1 + px2) // 2, (py1 + py2) // 2)

                closest_vehicle = None
                min_dist = float("inf")

                for vehicle in vehicles:
                    vid = vehicle["id"]
                    if vid in used_vehicle_ids:
                        continue  # อย่าใช้ซ้ำ

                    vx1, vy1, vx2, vy2 = vehicle["box"]
                    v_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                    dist = ((p_center[0] - v_center[0]) ** 2 + (p_center[1] - v_center[1]) ** 2) ** 0.5
                    print(f"track_id : {track_id}  dis")

                    if dist < min_dist and dist < 400:  # ตั้ง threshold ระยะที่เหมาะสม
                        min_dist = dist
                        closest_vehicle = vehicle

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
                        cv2.putText(showframe, f"person_on_vehicle", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        crop = frame[y1:y2, x1:x2]
                        obj_path, frame_path = save_detection(crop, pid, showframe)
                        tracked_ids.update({pid, vid})
                        used_vehicle_ids.add(vid)
                        print(f"Saved pair: person ID {pid}, vehicle ID {vid} dist : {dist}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        cv2.imwrite('people_on_vahicle.jpg',showframe)

    cv2.destroyAllWindows()
    print("Tracking finished.")



if __name__ == "__main__":
    main()
