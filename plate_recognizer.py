import cv2
import easyocr
import numpy as np
import re

# Инициализируем OCR с русским и английским (английский обязателен для кириллицы)
ocr_reader = easyocr.Reader(["ru", "en"], gpu=True)

# Символы, которые реально могут быть на российском номере
LATIN_TO_CYRILLIC = str.maketrans('ABEKMHOPCTYX', 'АВЕКМНОРСТУХ')
PLATE_LETTERS = 'АВЕКМНОРСТУХ'
VALID_CHARS = PLATE_LETTERS + 'ABEKMHOPCTYX0123456789'
PLATE_BODY = rf'[{PLATE_LETTERS}][0-9]{{3}}[{PLATE_LETTERS}]{{2}}'


def normalize_ocr_text(raw_text: str) -> str:
    return "".join(c for c in raw_text.upper() if c.isalnum()).translate(LATIN_TO_CYRILLIC)


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


def clean_plate_text(raw_text: str) -> str | None:
    """Ищет номер стандартного формата в тексте OCR"""
    text = normalize_ocr_text(raw_text)

    # Паттерн: Буква (АВЕКМНОРСТУХ) + 3 цифры + 2 буквы + 2-3 цифры региона
    match = re.search(rf'{PLATE_BODY}[0-9]{{2,3}}', text)
    if match:
        return match.group(0)

    # EasyOCR может прочитать начальную букву О как цифру 0.
    if re.fullmatch(r'0[0-9]{3}[АВЕКМНОРСТУХ]{2}[0-9]{2,3}', text):
        return 'О' + text[1:]

    return None


def is_adjacent_region(body_box, region_box):
    body, region = map(np.asarray, (body_box, region_box))
    left = (body[0] + body[3]) / 2
    right = (body[1] + body[2]) / 2
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(body[3] - body[0])
    region_height = np.linalg.norm(region[3] - region[0])
    if width == 0 or height == 0 or not 0.5 * height <= region_height <= 1.6 * height:
        return False

    along = (right - left) / width
    across = np.array([-along[1], along[0]])
    region_left = (region[0] + region[3]) / 2
    return (
        -0.6 * height <= np.dot(region_left - right, along) <= 0.6 * height
        and abs(np.dot(region.mean(axis=0) - right, across)) <= 0.55 * height
    )


def plate_candidates(ocr_results):
    blocks = [(box, normalize_ocr_text(text), float(conf)) for box, text, conf in ocr_results]
    for box, text, conf in blocks:
        plate = clean_plate_text(text)
        if plate:
            yield plate, conf
            continue

        if re.fullmatch(rf'[{PLATE_LETTERS}0][0-9]{{3}}[{PLATE_LETTERS}]{{2}}', text):
            for region_box, region, region_conf in blocks:
                if re.fullmatch(r'[0-9]{2,3}', region) and is_adjacent_region(box, region_box):
                    yield clean_plate_text(text + region), min(conf, region_conf)

        # Исправляем типичные подмены только в двух позициях букв после цифр.
        if re.fullmatch(rf'[{PLATE_LETTERS}][0-9]{{3}}[{PLATE_LETTERS}18]{{2}}[0-9]{{2,3}}', text):
            letters = text[4:6].translate(str.maketrans('18', 'ТВ'))
            yield text[:4] + letters + text[6:], conf * 0.8


def recognize_plate(car_crop):
    """
    Принимает вырезанное изображение машины.
    Возвращает кортеж (номер, уверенность) или (None, 0.0).
    """
    if car_crop is None or car_crop.size == 0:
        return None, 0.0

    binary = enhance_plate_image(car_crop)
    results = ocr_reader.readtext(binary, detail=1, allowlist=VALID_CHARS, low_text=0.3)
    best = max(plate_candidates(results), key=lambda candidate: candidate[1], default=(None, 0.0))
    if best[0]:
        return best

    gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    results = ocr_reader.readtext(gray, detail=1, allowlist=VALID_CHARS, low_text=0.3)
    return max(plate_candidates(results), key=lambda candidate: candidate[1], default=(None, 0.0))
