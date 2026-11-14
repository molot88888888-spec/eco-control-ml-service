import cv2
import easyocr
import numpy as np
import re

# Инициализируем OCR с русским и английским (английский обязателен для кириллицы)
ocr_reader = easyocr.Reader(["ru", "en"], gpu=True)

# Символы, которые реально могут быть на российском номере
VALID_CHARS = 'АВЕКМНОРСТУХ0123456789'


def enhance_plate_image(plate_crop):
    """Улучшает картинку для OCR: увеличивает, повышает контраст, применяет Otsu"""
    if plate_crop is None or plate_crop.size == 0:
        return None

    # 1. Увеличиваем в 3 раза
    plate_crop = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Переводим в серый
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # 3. Повышаем контраст (CLAHE) — вытягиваем буквы из тени
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. Повышаем резкость
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # 5. Бинаризация Otsu (делает буквы черными на белом фоне)
    # Исследования подтверждают, что Otsu помогает EasyOCR на низкокачественных номерах
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def clean_plate_text(raw_text: str) -> str:
    """Ищет паттерн номера РФ в тексте с учетом возможных замен"""
    text = "".join(c for c in raw_text.upper() if c.isalnum())

    # Паттерн: Буква (АВЕКМНОРСТУХ) + 3 цифры + 2 буквы + 2-3 цифры региона
    match = re.search(r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}', text)
    if match:
        return match.group(0)

    # Если паттерн не найден, пробуем найти его с заменой похожих символов
    # Например, если OCR прочитал 0 вместо О, 4 вместо А и т.д.
    replacements = {'0': 'О', '1': 'Т', '8': 'В', '6': 'G', '4': 'А', '5': 'S'}
    soft_text = text
    for digit, letter in replacements.items():
        soft_text = soft_text.replace(digit, letter)

    match_soft = re.search(r'[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}', soft_text)
    if match_soft:
        # Возвращаем оригинальные символы из этой позиции
        return text[match_soft.start():match_soft.end()]

    # Если ничего не нашли, но длина подходящая — возвращаем как есть
    if 7 <= len(text) <= 10 and any(c.isdigit() for c in text):
        return text

    return None


def recognize_plate(car_crop):
    """
    Принимает вырезанное изображение машины.
    Возвращает кортеж (номер, уверенность) или (None, 0.0).
    """
    if car_crop is None or car_crop.size == 0:
        return None, 0.0

    # 1. Улучшаем картинку
    enhanced_crop = enhance_plate_image(car_crop)
    if enhanced_crop is None:
        return None, 0.0

    # 2. Запускаем OCR с ограничением символов и низким порогом
    ocr_results = ocr_reader.readtext(
        enhanced_crop,
        detail=1,
        allowlist=VALID_CHARS,  # Только буквы и цифры, допустимые на номерах
        low_text=0.3  # Понижаем порог, чтобы ловить слабые символы
    )

    best_plate = None
    best_conf = 0.0

    # 3. Ищем лучший результат
    for (_, text, conf) in ocr_results:
        clean = clean_plate_text(text)
        if clean and conf > best_conf:
            best_conf = float(conf)
            best_plate = clean

    return best_plate, best_conf