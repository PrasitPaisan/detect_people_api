import os 
import io
import cv2
from google.cloud import storage

def save_detection(crop, frame, track_id, timestamp):
    
    saved_dir = r"C:\Users\metthier\Downloads\for_test\Helmet_code_api"

    try:
        os.makedirs(os.path.join(saved_dir, "saved_detections"), exist_ok=True)
        os.makedirs(os.path.join(saved_dir, "overview_detections"), exist_ok=True)
    except Exception as e :
        print(f"Failed to create directory : {e}")

    obj_path = os.path.join(saved_dir, "saved_detections", f"crop_{track_id}_{timestamp}.jpg")
    overview_path = os.path.join(saved_dir, "overview_detections", f"frame_{track_id}_{timestamp}.jpg")

    try:
        if not cv2.imwrite(obj_path, crop):
            print(f"Failed to save crop image")
    except Exception as e:
        print(f"Failed to save crop iamge error : {e}")

    try:
        if not cv2.imwrite(overview_path, frame):
            print(f"Failed to save overview image")
    except Exception as e:
        print(f"Fail to save overview image error : {e}")

    return obj_path, overview_path