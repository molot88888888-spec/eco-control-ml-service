# EcoControl

[Русская версия](README.md)

EcoControl is a demonstration service for reviewing environmental monitoring recordings. An operator uploads a video and selects a simulated camera and microphone. YOLO detects vehicles, EasyOCR reads Russian license plates, and YAMNet checks the embedded audio for vehicle or engine sounds. The operator dashboard is in Russian.

## Run on Windows

From the repository directory in PowerShell, using Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1
```

Open [the operator dashboard](http://127.0.0.1:8000). The local demo login is `admin` with password `admin`. The first analysis may download model weights.

### NVIDIA GPU acceleration

After installing the requirements, install the CUDA-enabled PyTorch and torchvision wheels in the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" --index-url https://download.pytorch.org/whl/cu128
```

Restart the server. YOLO and EasyOCR use the GPU when CUDA is available and fall back to the CPU otherwise. YAMNet runs on the CPU in this native Windows setup.

## Review a recording

1. Sign in and upload an MP4, MOV, MKV, AVI, or WebM file up to 100 MiB. For an audio demo, use a short clip with an audible engine in the first 10 seconds.
2. Review the detected plate, OCR confidence, and audio result. If a plate was recognized, open the saved frame. Confirm or reject the event.
3. Download the evidence archive if needed. The service checks the source video's SHA-256 hash before returning an archive with the original video, analysis JSON, a text report, and the saved frame when available.

The selected sources record where the test file came from. The service reports the clip duration. It does not attribute the audio to a specific vehicle, measure how long an engine ran, or identify the vehicle make.

## Configuration

The default database is local SQLite. For PostgreSQL, set `ECO_DATABASE_URL` to a URL such as `postgresql+psycopg://user:password@localhost/eco`.

For use outside the local demo, set `ECO_ADMIN_PASSWORD` and `ECO_SESSION_SECRET`. Set `ECO_SECURE_COOKIE=1` when serving over HTTPS.
