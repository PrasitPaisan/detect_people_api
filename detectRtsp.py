import torch
print("GPU CUDA is", torch.cuda.is_available())
import numpy as np

from ultralytics import YOLO
import cv2
import os
from datetime import datetime

MODEL_PATH = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\yolo12n.pt"
VIDEO_PATH = r"rtsp://mioc_cms:Mi0C2023Cms@10.54.2.2:554/Streaming/Channels/5301"
SAVE_DIR = r"C:\Users\metthier\Desktop\Prasit_Paisan\helmet_detection\detect_people_api\Helmet_people"
os.makedirs(SAVE_DIR, exist_ok=True)

roi_points = [
    [268.2335034687419, 387.935523020233],
    [1419.4285865910601, 260.64417952921906],
    [1853.651293382812, 476.83741371236977],
    [1859.7102148729293, 1103.1916435887877],
    [342.96020184685733, 1119.3556237146306]
]

roi_intrusion = np.array(roi_points, dtype=np.int32)


def load_model(path):
    return YOLO(path, verbose=False)

def save_detection(crop, track_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_path = os.path.join(SAVE_DIR, f"human_{track_id}_{timestamp}.jpg")
    cv2.imwrite(object_path, crop)
    return object_path

def main():
    model = load_model(MODEL_PATH)
    tracked_ids = set()

   
    for results in model.track(
        source=VIDEO_PATH,
        stream=True,
        save=False,
        show=True,
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
                center_point = (int((x1+x2)/2),int((y1+y2)/2))
                inside_intrusion = cv2.pointPolygonTest(roi_intrusion,center_point,False) >= 0

                

                cv2.rectangle(showframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(showframe, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.polylines(showframe, [roi_intrusion], isClosed=True, color=(0, 0, 255), thickness=2)

                if inside_intrusion:
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                        crop = frame[y1:y2, x1:x2]
                        obj_path = save_detection(crop, track_id)
                        print(f"[SAVED] {class_name} (ID:{track_id}) at {obj_path}")

        cv2.imwrite('test.jpg',showframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break

    cv2.destroyAllWindows()
    print("Tracking finished.")

if __name__ == "__main__":
    main()
