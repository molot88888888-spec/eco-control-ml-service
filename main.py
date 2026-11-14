import os
import shutil
import uuid
import json
import hashlib
import zipfile  # <-- ДОБАВЛЕНО ДЛЯ ZIP
from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse  # <-- ДОБАВЛЕН FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import SessionLocal, Detection
from detector import analyze_video

app = FastAPI(title="EcoControl")
templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    detections = db.query(Detection).order_by(Detection.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"detections": detections}
    )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join("uploads", stored_name)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_hash = get_file_hash(path)
    result = analyze_video(path)

    det = Detection(
        filename=stored_name,
        file_hash=file_hash,
        car_detected=result.get("car_count", 0),
        plate_text=result.get("plate_text"),
        confidence=result.get("confidence", 0.0),
        duration_sec=result.get("duration_sec", 0.0),
        audio_detected=1 if result.get("audio_detected") else 0,
        ml_raw_json=json.dumps(result, ensure_ascii=False),
    )
    db.add(det)
    db.commit()

    return RedirectResponse(url="/", status_code=303)


@app.post("/update_status/{det_id}/{status}")
async def update_status(det_id: int, status: str, db: Session = Depends(get_db)):
    det = db.query(Detection).filter(Detection.id == det_id).first()
    if det and status in ["confirmed", "rejected"]:
        det.status = status
        db.commit()
    return RedirectResponse(url="/", status_code=303)


# --- НОВЫЙ ЭНДПОИНТ ДЛЯ СКАЧИВАНИЯ ZIP ---
@app.get("/download_evidence/{det_id}")
async def download_evidence(det_id: int, db: Session = Depends(get_db)):
    """Формирование ZIP-архива с доказательствами"""
    det = db.query(Detection).filter(Detection.id == det_id).first()
    if not det:
        return {"error": "Событие не найдено"}

    zip_filename = f"evidence_{det_id}.zip"
    zip_path = os.path.join("output", zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # 1. Добавляем исходное видео
        video_path = os.path.join("uploads", det.filename)
        if os.path.exists(video_path):
            zipf.write(video_path, arcname="source_video.mp4")

        # 2. Добавляем кадр с номером
        proof_path = os.path.join("output", f"proof_{det.filename}.jpg")
        if os.path.exists(proof_path):
            zipf.write(proof_path, arcname="proof_frame.jpg")

        # 3. Добавляем текстовый отчет
        report_text = (
            f"ОТЧЕТ ОБ ЭКОЛОГИЧЕСКОМ ПРАВОНАРУШЕНИИ\n"
            f"ID события: {det.id}\n"
            f"Время: {det.created_at}\n"
            f"Номер авто: {det.plate_text}\n"
            f"Длительность: {det.duration_sec} сек.\n"
            f"Уверенность ML: {det.confidence}\n"
            f"Звук двигателя: {'Да' if det.audio_detected else 'Нет'}\n"
            f"SHA-256 исходного видео: {det.file_hash}\n"
            f"Статус оператора: {det.status}\n"
        )
        zipf.writestr("report.txt", report_text)

    return FileResponse(zip_path, filename=zip_filename, media_type="application/zip")