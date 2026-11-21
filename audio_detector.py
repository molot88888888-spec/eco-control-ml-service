import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import urllib.request
import csv
from tempfile import TemporaryDirectory
from moviepy import VideoFileClip

print("Загрузка YAMNet (нейросеть для звука)...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

if not os.path.exists('yamnet_class_map.csv'):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
        'yamnet_class_map.csv'
    )

# Классы, которые обозначают технику и моторы
TARGET_CLASSES = [
    'Vehicle', 'Motor vehicle (road)', 'Car', 'Truck', 'Bus',
    'Engine', 'Idling', 'Heavy engine', 'Medium engine', 'Light engine'
]

target_indices = []
with open('yamnet_class_map.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        display_name = row[2]
        if any(target.lower() in display_name.lower() for target in TARGET_CLASSES):
            target_indices.append(int(row[0]))


def analyze_audio(video_path: str) -> dict:
    if not os.path.exists(video_path):
        return {"engine_detected": False, "audio_confidence": 0.0}

    try:
        # 1. Извлекаем звук
        with TemporaryDirectory() as temp_dir, VideoFileClip(video_path) as video:
            if video.audio is None:
                return {"engine_detected": False, "audio_confidence": 0.0, "error": "Нет аудиодорожки"}

            temp_wav = os.path.join(temp_dir, "audio.wav")
            video.audio.write_audiofile(temp_wav, fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
            y, _ = librosa.load(temp_wav, sr=16000, duration=10)

        if len(y) == 0:
            return {"engine_detected": False, "audio_confidence": 0.0}

        # 3. Анализ YAMNet
        scores, _, _ = yamnet_model(y)
        mean_scores = np.mean(scores.numpy(), axis=0)

        # Считаем вероятность того, что в звуке есть ЛЮБОЙ двигатель
        max_conf = 0.0
        if target_indices:
            # Берем максимальную вероятность среди всех "моторных" классов
            max_conf = float(np.max(mean_scores[target_indices]))

        # Если нейросеть уверена больше чем на 12% (0.12), считаем что мотор есть
        # Порог 0.12 подобран эмпирически, чтобы отсечь фоновый гул улицы
        is_engine = bool(max_conf > 0.12)

        print(f"DEBUG AUDIO (Только YAMNet): Уверенность={max_conf:.3f} -> Итог={is_engine}")

        return {
            "engine_detected": is_engine,
            "audio_confidence": round(max_conf, 3)
        }

    except Exception as e:
        print(f"Ошибка аудиоанализа: {e}")
        return {"engine_detected": False, "audio_confidence": 0.0, "error": "Аудио не удалось проанализировать"}