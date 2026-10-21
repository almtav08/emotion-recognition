"""
Preprocesamiento y Alineación Facial de FERAC con MediaPipe
Alinea ojos en horizontal (θ = 0°), añade margen dinámico del 12% y redimensiona a 224x224.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FERAC_DIR = BASE_DIR / "data" / "raw" / "ferac"
PROCESSED_FERAC_DIR = BASE_DIR / "data" / "processed" / "ferac_aligned"


def align_and_crop_face(image_bgr, detection, output_size=(224, 224), margin=0.12):
    """
    Utiliza los keypoints de MediaPipe Face Detection para rotar la cara
    y alinear la línea interocular horizontalmente antes de recortar.
    """
    h, w, _ = image_bgr.shape
    bbox = detection.location_data.relative_bounding_box
    xmin = int(bbox.xmin * w)
    ymin = int(bbox.ymin * h)
    box_w = int(bbox.width * w)
    box_h = int(bbox.height * h)

    # Keypoints: 0: Ojo Derecho, 1: Ojo Izquierdo, 2: Nariz, 3: Boca, etc.
    keypoints = detection.location_data.relative_keypoints
    right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))

    # Calcular ángulo de rotación para horizontalizar los ojos
    d_y = right_eye[1] - left_eye[1]
    d_x = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(d_y, d_x)) - 180.0

    # Centro entre los ojos
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Matriz de rotación
    m_rot = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    rotated = cv2.warpAffine(image_bgr, m_rot, (w, h), flags=cv2.INTER_CUBIC)

    # Aplicar margen dinámico
    m_w = int(box_w * margin)
    m_h = int(box_h * margin)
    x1 = max(0, xmin - m_w)
    y1 = max(0, ymin - m_h)
    x2 = min(w, xmin + box_w + m_w)
    y2 = min(h, ymin + box_h + m_h)

    face_crop = rotated[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, output_size, interpolation=cv2.INTER_AREA)


def process_dataset():
    if not RAW_FERAC_DIR.exists():
        print(f"[!] No se encontró el directorio de datos crudos en {RAW_FERAC_DIR}")
        print("    Descárgalo primero con: uv run python scripts/download_datasets.py --ferac")
        return

    PROCESSED_FERAC_DIR.mkdir(parents=True, exist_ok=True)
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    classes = ["Anger", "Fear", "Joy", "Natural"]

    print(f"Iniciando preprocesamiento de FERAC con MediaPipe desde {RAW_FERAC_DIR}...")
    stats = {"processed": 0, "fallback_crop": 0, "failed": 0}

    for cls in classes:
        cls_raw_dir = RAW_FERAC_DIR / cls
        # Soporte para subcarpetas no estándar
        if not cls_raw_dir.exists():
            matches = list(RAW_FERAC_DIR.rglob(cls))
            if matches:
                cls_raw_dir = matches[0]

        if not cls_raw_dir.exists():
            continue

        cls_out_dir = PROCESSED_FERAC_DIR / cls
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in cls_raw_dir.iterdir() if p.suffix.lower() in valid_extensions]
        print(f"Procesando clase '{cls}': {len(images)} imágenes encontradas.")

        for img_path in tqdm(images, desc=f"Clase {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                stats["failed"] += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(img_rgb)

            out_img = None
            if results.detections:
                # Seleccionar la detección con mayor confianza
                detection = results.detections[0]
                out_img = align_and_crop_face(img, detection)
                stats["processed"] += 1
            else:
                # Fallback: recorte central si MediaPipe no detecta cara
                h, w, _ = img.shape
                sz = min(h, w)
                y1 = (h - sz) // 2
                x1 = (w - sz) // 2
                out_img = cv2.resize(img[y1:y1+sz, x1:x1+sz], (224, 224))
                stats["fallback_crop"] += 1

            if out_img is not None:
                out_path = cls_out_dir / img_path.name
                cv2.imwrite(str(out_path), out_img)

    print("\n--- Preprocesamiento de FERAC Finalizado ---")
    print(f"  * Caras alineadas y recortadas: {stats['processed']}")
    print(f"  * Recortes por fallback central: {stats['fallback_crop']}")
    print(f"  * Errores de lectura:           {stats['failed']}")
    print(f"  * Destino procesado:            {PROCESSED_FERAC_DIR}")


if __name__ == "__main__":
    process_dataset()
