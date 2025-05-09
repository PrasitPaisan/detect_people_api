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
