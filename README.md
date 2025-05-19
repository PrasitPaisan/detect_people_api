# 🛡️ Detect People API (FastAPI)

ระบบตรวจจับบุคคลขับขี่มอเตอร์ไซค์ และจักรยาน พร้อมส่งภาพและข้อมูลการตรวจจับไปเก็บใน Google Cloud Storage ด้วย YOLO และ FastAPI

---

## 📦 Features

- 🎥 รองรับการ Stream วิดีโอแบบ Real-time ผ่าน RTSP
- 👤 ตรวจจับบุคคลที่ขับขี่มอเตอร์ไซค์ และจักรยานด้วย YOLO
- 🧠 ตรวจสอบ attributes เช่น ใส่หมวกนิรภัย
- 🖼️ บันทึก Snapshot, Overview Image และ Overview Image with ROI
- ☁️ เชื่อมต่อและบันทึกภาพบน Google Cloud Storage
- 🗃️ ส่ง path ของรูปภาพกลับไปยัง client เพื่อใช้งานต่อ

---

## 🚀 Tech Stack

- **Backend**: FastAPI  
- **Model**: YOLOv12 
- **Database**: PostgreSQL  
- **Storage**: Google Cloud Storage (GCS)  
- **Tracking**: ByteTrack  
- **Libraries**: OpenCV, Requests, Numpy, Python-dotenv

---

## 📂 Project Structure

app/
├── controllers/ # FastAPI route handlers
├── dbs/ # Database access layer
├── models/ # Pydantic models for request/response
├── utils/ # Helper functions (e.g. generate_data, save_image_to_googleStorage,. . .)
main.py # Application entry point

---

## ⚙️ Environment Variables

สร้างไฟล์ `.env` และตั้งค่าดังนี้:

```env
MODEL_PATH=model/best.pt
PRIVATE_KEY=path/to/your-google-service-key.json
BUCKET_NAME=your-gcs-bucket-name


```
### 1. เข้าถึง path ของ project
cd human_body_detectio

### 2. ติดตั้ง dependencies
pip install -r requirements.txt

### 3. รันเซิร์ฟเวอร์
uvicorn main:app --reload --host 0.0.0.0 --port 8000

---

## Example API
 📡 POST `/api/detect/stream`

**input**
{
  "video_path": "rtsp://<user>:<pass>@<ip>:<port>/path"
}

**response**
data: {"snapshot": "...", "overview": "...", "overview_with_roi": "...", "timestamp": "..." }

