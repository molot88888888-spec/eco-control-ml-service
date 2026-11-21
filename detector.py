import cv2
from ultralytics import YOLO
import numpy as np
import os
import torch
from audio_detector import analyze_audio
from plate_recognizer import recognize_plate  # <-- Импортируем наш новый модуль

yolo_model = YOLO("yolov8n.pt")
YOLO_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
VEHICLE_CLASSES = [2, 5, 7]


def analyze_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    frames_read = 0
    car_count = 0
    best_plate = None
    best_conf = 0.0
    best_frame = None

    # Проверяем 10 кадров
    num_checks = 10
    frame_indices = np.linspace(0, max(0, total_frames - 1), num_checks, dtype=int)

    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frames_read += 1
            results = yolo_model(frame, device=YOLO_DEVICE, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls not in VEHICLE_CLASSES:
                        continue

                    car_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = frame.shape[:2]

                    # Вырезаем машину
                    car_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    if car_crop.size == 0:
                        continue

                    # Отправляем в отдельный модуль на распознавание
                    plate, conf = recognize_plate(car_crop)

                    if plate and conf > best_conf:
                        best_conf = conf
                        best_plate = plate
                        best_frame = frame.copy()
    finally:
        cap.release()

    if not frames_read:
        raise ValueError("Видео не содержит читаемых кадров")

    # Сохраняем кадр-доказательство
    if best_frame is not None:
        os.makedirs("output", exist_ok=True)
        proof_path = os.path.join("output", f"proof_{os.path.basename(video_path)}.jpg")
        cv2.imwrite(proof_path, best_frame)

    # Аудиоанализ
    audio_result = analyze_audio(video_path)

    return {
        "car_count": max(car_count, 1 if best_plate else 0),
        "plate_text": best_plate,
        "confidence": round(best_conf, 3) if best_plate else 0.0,
        "duration_sec": round(duration_sec, 1),
        "audio_detected": audio_result.get("engine_detected", False),
        "audio_metrics": audio_result,
        "models": {"video": "yolov8n.pt", "audio": "YAMNet/1", "ocr": "EasyOCR/ru+en"},
        "sound_attribution": "undetermined" if audio_result.get("engine_detected") else None
    }
