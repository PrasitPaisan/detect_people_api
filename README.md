# Detect People API (FastAPI)

A FastAPI-based system for detecting people riding motorcycles and bicycles. The system captures detection images and metadata, then uploads them to Google Cloud Storage using YOLO and ByteTrack.

---

## Features

- Real-time video stream detection via RTSP
- Detects people riding motorcycles and bicycles using YOLO
- Attribute analysis (e.g., helmet detection)
- Saves snapshot, overview, and overview-with-ROI images
- Uploads images to Google Cloud Storage
- Returns image paths and detection data to the client

---

## Tech Stack

- Backend: FastAPI
- Model: YOLOv12
- Database: PostgreSQL
- Storage: Google Cloud Storage (GCS)
- Tracking: ByteTrack
- Libraries: OpenCV, Requests, Numpy, Python-dotenv

---

## Project Structure

```
app/
├── controllers/        # FastAPI route handlers
├── dbs/                # Database access layer
├── models/             # Pydantic models for request/response
├── utils/              # Helper functions (e.g., generate_data, save_image_to_googleStorage, ...)
main.py                 # Application entry point
```

---

## Environment Variables

Create a `.env` file in the project root with the following content:

```env
MODEL_PATH=model/best.pt
PRIVATE_KEY=path/to/your-google-service-key.json
BUCKET_NAME=your-gcs-bucket-name
```

---

## Getting Started

1. Change to the project directory:
    ```sh
    cd human_body_detection
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the server:
    ```sh
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

---

## Example API Usage

**POST** `/api/detect/stream`

**Request Body**
```json
{
  "video_path": "rtsp://<user>:<pass>@<ip>:<port>/path"
}
```

**Response**
```json
{
  "snapshot": "...",
  "overview": "...",
  "overview_with_roi": "...",
  "timestamp": "..."
}
```
