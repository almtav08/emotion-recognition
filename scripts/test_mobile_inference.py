"""
Script de Inferencia On-Device y Benchmark de Latencia
Carga el modelo ONNX exportado y evalúa una imagen de prueba midiendo la latencia en milisegundos.
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ONNX = BASE_DIR / "weights" / "fer_mobile_enet_b0_8_best_vgaf_tea.onnx"
CLASSES = ['Natural', 'anger', 'fear', 'joy', 'sadness', 'surprise']


def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    # Redimensionar a 224x224 (MediaPipe crop format)
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_rgb / 255.0 - mean) / std

    # HWC -> CHW -> NCHW
    tensor = norm.transpose(2, 0, 1).astype(np.float32)
    tensor = np.expand_dims(tensor, axis=0)
    return tensor


def benchmark_mobile_model(image_path=None, model_path=None):
    onnx_file = Path(model_path) if model_path else DEFAULT_ONNX
    if not onnx_file.exists():
        print(f"[!] Archivo ONNX no encontrado en: {onnx_file}")
        return

    print("====================================================================")
    print(" BENCHMARK DE INFERENCIA ON-DEVICE (ONNX RUNTIME MOBILE)")
    print("====================================================================")
    print(f"Modelo cargado: {onnx_file.name} ({onnx_file.stat().st_size / (1024*1024):.2f} MB)")

    # Sesión con optimizaciones básicas de ejecución
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_file), opts, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if image_path and Path(image_path).exists():
        input_tensor = preprocess_image(image_path)
    else:
        # Usar una imagen real del conjunto de test
        test_imgs = list((BASE_DIR / "data" / "raw" / "ferac" / "test").glob("*/*.*"))
        if test_imgs:
            sample_img = test_imgs[0]
            print(f"Evaluando imagen de muestra: {sample_img.parent.name}/{sample_img.name}")
            input_tensor = preprocess_image(sample_img)
        else:
            input_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(5):
        _ = session.run([output_name], {input_name: input_tensor})

    # Medición de latencia en 50 iteraciones
    num_runs = 50
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        logits = session.run([output_name], {input_name: input_tensor})[0]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    # Probabilidades softmax
    exp_l = np.exp(logits[0] - np.max(logits[0]))
    probs = exp_l / np.sum(exp_l)
    pred_idx = int(np.argmax(probs))

    avg_lat = np.mean(times)
    p95_lat = np.percentile(times, 95)
    fps = 1000.0 / avg_lat

    print("\n--- Resultados de Rendimiento en CPU (Simulación Móvil) ---")
    print(f"  * Latencia Media (CPU): {avg_lat:.2f} ms")
    print(f"  * Latencia Percentil 95: {p95_lat:.2f} ms")
    print(f"  * Tasa de Cuadros (FPS): {fps:.1f} cuadros por segundo")
    print(f"  *(Nota: En NPU/GPU de móvil con TFLite/CoreML la latencia típicamente se reduce a ~6-12 ms)*")

    print("\n--- Predicción de Emoción ---")
    for cls_name, prob in zip(CLASSES, probs):
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:<10}: {prob*100:5.1f}% | {bar}")

    print(f"\nPredicción Ganadora: {CLASSES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")
    print("====================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Inferencia Móvil")
    parser.add_argument("--image", default=None, help="Ruta a una imagen específica")
    parser.add_argument("--model", default=None, help="Ruta al modelo ONNX")
    args = parser.parse_args()
    benchmark_mobile_model(image_path=args.image, model_path=args.model)
