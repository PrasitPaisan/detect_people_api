import os 
import io
import cv2
from google.cloud import storage

def save_image_to_googleStorage(image, bucket_name, path, name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    image_path = f"{path}/{name}".strip("/")
    blob = bucket.blob(image_path)

    if isinstance(image, str):
        blob.upload_from_filename(image)
    else:
        # Convert numpy image to JPEG in memory
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image to JPEG")
        image_bytes = encoded_image.tobytes()
        image_io = io.BytesIO(image_bytes)
        image_io.seek(0)

        # blob.upload_from_file(image_io, content_type="image/jpeg")

    return f"https://storage.googleapis.com/{bucket_name}/{image_path}"

def save_detection(crop, frame, track_id, timestamp):
    saved_dir = r"C:\Users\metthier\Downloads\for_test\Helmet_code_api"
    os.makedirs(os.path.join(saved_dir, "saved_detections"), exist_ok=True)
    os.makedirs(os.path.join(saved_dir, "overview_detections"), exist_ok=True)

    obj_path = os.path.join(saved_dir, "saved_detections",
                            f"crop_{track_id}_{timestamp}.jpg")
    overview_path = os.path.join(
        saved_dir, "overview_detections", f"frame_{track_id}_{timestamp}.jpg")
    cv2.imwrite(obj_path, crop)
    cv2.imwrite(overview_path, frame)

    return obj_path, overview_path