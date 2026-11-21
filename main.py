import hashlib
import io
import json
import os
import secrets
import uuid
import zipfile
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from database import Detection, SessionLocal

ROOT = Path(__file__).resolve().parent
UPLOADS = Path("uploads")
OUTPUT = Path("output")
MAX_BYTES = 100 * 1024 * 1024
ANALYSIS_LOCK = Lock()
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
SOURCES = {
    "camera-1": ("video", "Камера 1"),
    "camera-2": ("video", "Камера 2"),
    "microphone-1": ("audio", "Микрофон 1"),
    "microphone-2": ("audio", "Микрофон 2"),
}

app = FastAPI(title="EcoControl")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("ECO_SESSION_SECRET") or secrets.token_hex(32),
    max_age=3600,
    https_only=os.getenv("ECO_SECURE_COOKIE") == "1",
)
templates = Jinja2Templates(directory=ROOT / "Templates")
UPLOADS.mkdir(exist_ok=True)
OUTPUT.mkdir(exist_ok=True)


def get_db():
    with SessionLocal() as db:
        yield db


def require_operator(request: Request):
    if request.session.get("operator") != "admin":
        raise HTTPException(401, "Войдите как оператор")


def check_csrf(request: Request, csrf: str = Form(...)):
    expected = request.session.get("csrf")
    if not expected or not secrets.compare_digest(expected, csrf):
        raise HTTPException(403, "Обновите страницу и повторите действие")


def get_file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def proof_path(det):
    return OUTPUT / f"proof_{det.filename}.jpg"


def get_detection(db, det_id):
    det = db.get(Detection, det_id)
    if det is None:
        raise HTTPException(404, "Событие не найдено")
    return det


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if request.session.get("operator"):
        return RedirectResponse("/", status_code=303)
    request.session.setdefault("csrf", secrets.token_urlsafe(32))
    return templates.TemplateResponse(request=request, name="login.html")


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          _: None = Depends(check_csrf)):
    user_ok = secrets.compare_digest(username.encode("utf-8"), b"admin")
    password_ok = secrets.compare_digest(
        password.encode("utf-8"), os.getenv("ECO_ADMIN_PASSWORD", "admin").encode("utf-8")
    )
    if not (user_ok and password_ok):
        return templates.TemplateResponse(
            request=request, name="login.html",
            context={"error": "Неверный логин или пароль"}, status_code=401,
        )
    request.session.clear()
    request.session.update(operator="admin", csrf=secrets.token_urlsafe(32))
    return RedirectResponse("/", status_code=303)


@app.post("/logout", dependencies=[Depends(require_operator), Depends(check_csrf)])
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def index(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("operator"):
        return RedirectResponse("/login", status_code=303)
    detections = db.query(Detection).order_by(Detection.created_at.desc()).all()
    entries = [(det, json.loads(det.ml_raw_json or "{}"), proof_path(det).is_file())
               for det in detections]
    return templates.TemplateResponse(
        request=request, name="index.html", context={"entries": entries, "sources": SOURCES}
    )


@app.post("/analyze", dependencies=[Depends(require_operator), Depends(check_csrf)])
def analyze(file: UploadFile = File(...), camera: str = Form(...),
            microphone: str = Form(...), db: Session = Depends(get_db)):
    if camera not in {"camera-1", "camera-2"} or microphone not in {"microphone-1", "microphone-2"}:
        raise HTTPException(422, "Выберите демонстрационные источники")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(415, "Поддерживаются MP4, MOV, MKV, AVI и WebM")

    path = UPLOADS / f"{uuid.uuid4().hex}{extension}"
    frame = OUTPUT / f"proof_{path.name}.jpg"
    try:
        digest = hashlib.sha256()
        size = 0
        with path.open("xb") as stream:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, "Размер видео превышает 100 МиБ")
                digest.update(chunk)
                stream.write(chunk)
        if not size:
            raise HTTPException(422, "Файл пуст")

        from detector import analyze_video

        try:
            with ANALYSIS_LOCK:
                result = analyze_video(str(path))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        result["observation"] = {
            "mode": "demo",
            "camera": camera,
            "microphone": microphone,
            "audio_origin": "embedded",
        }
        det = Detection(
            filename=path.name,
            file_hash=digest.hexdigest(),
            car_detected=result["car_count"],
            plate_text=result["plate_text"],
            confidence=result["confidence"],
            duration_sec=result["duration_sec"],
            audio_detected=int(result["audio_detected"]),
            ml_raw_json=json.dumps(result, ensure_ascii=False),
        )
        db.add(det)
        db.commit()
    except Exception:
        db.rollback()
        path.unlink(missing_ok=True)
        frame.unlink(missing_ok=True)
        raise
    return RedirectResponse("/", status_code=303)


@app.post("/update_status/{det_id}/{status}",
          dependencies=[Depends(require_operator), Depends(check_csrf)])
def update_status(det_id: int, status: str, db: Session = Depends(get_db)):
    if status not in {"confirmed", "rejected"}:
        raise HTTPException(422, "Недопустимый статус")
    updated = db.query(Detection).filter(
        Detection.id == det_id, Detection.status == "pending"
    ).update({Detection.status: status})
    if not updated:
        get_detection(db, det_id)
        raise HTTPException(409, "Событие уже обработано")
    db.commit()
    return RedirectResponse("/", status_code=303)


@app.get("/proof/{det_id}", dependencies=[Depends(require_operator)])
def proof(det_id: int, db: Session = Depends(get_db)):
    path = proof_path(get_detection(db, det_id))
    if not path.is_file():
        raise HTTPException(404, "Кадр отсутствует")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/download_evidence/{det_id}", dependencies=[Depends(require_operator)])
def download_evidence(det_id: int, db: Session = Depends(get_db)):
    det = get_detection(db, det_id)
    source = UPLOADS / det.filename
    if not source.is_file():
        raise HTTPException(409, "Исходное видео отсутствует")

    with io.BytesIO() as archive:
        with zipfile.ZipFile(archive, "w") as bundle:
            digest = hashlib.sha256()
            with source.open("rb") as stream, bundle.open(f"source{source.suffix}", "w") as entry:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    entry.write(chunk)
            if not secrets.compare_digest(digest.hexdigest(), det.file_hash):
                raise HTTPException(409, "Контрольная сумма исходного видео не совпадает")
            frame = proof_path(det)
            if frame.is_file():
                bundle.write(frame, "proof_frame.jpg")
            bundle.writestr("analysis.json", det.ml_raw_json or "{}")
            bundle.writestr("report.txt", "\n".join((
                "Материалы наблюдения для проверки оператором",
                f"ID: {det.id}",
                f"Время регистрации UTC: {det.created_at}",
                f"Номер автомобиля: {det.plate_text or 'Не распознан'}",
                f"Статус оператора: {det.status}",
                f"SHA-256 исходного видео: {det.file_hash}",
            )))
        return Response(
            archive.getvalue(), media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="evidence_{det_id}.zip"',
                "Cache-Control": "no-store",
            },
        )
