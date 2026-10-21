"""
Script de Evaluación Detallada para Modelos FER en TEA
Carga un checkpoint entrenado (.pth) o un modelo ONNX y muestra:
  - Macro F1-Score y Balanced Accuracy (UAR)
  - Reporte por clase (Precision, Recall, F1)
  - Matriz de Confusión formateada
  - Análisis de falsos negativos hacia la clase 'Natural' (Hipomimia)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.transforms import get_mobile_transforms


def evaluate_checkpoint(checkpoint_path, backbone="enet_b0_8_best_vgaf", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print(" EVALUACIÓN DE RENDIMIENTO CLÍNICO (TEST SET FERAC)")
    print("====================================================================")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint evaluado: {checkpoint_path}")

    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"
    if not test_dir.exists():
        print(f"[!] No se encontró el directorio de test en: {test_dir}")
        return

    test_ds = datasets.ImageFolder(str(test_dir), transform=get_mobile_transforms("eval"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = test_ds.classes
    print(f"Clases evaluadas ({len(classes)}): {classes}")
    print(f"Total de imágenes de test: {len(test_ds)}")

    # Instanciar arquitectura y cargar pesos
    model = MobileFERModel(backbone_name=backbone, num_classes=len(classes), pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    uar = balanced_accuracy_score(all_targets, all_preds)
    acc = np.mean(all_preds == all_targets)

    print("\n--------------------------------------------------------------------")
    print(" MÉTRICAS GLOBALES")
    print("--------------------------------------------------------------------")
    print(f"  * Macro F1-Score:       {macro_f1:.4f}  (Métrica principal balanceada)")
    print(f"  * Balanced Accuracy:    {uar:.4f}  (UAR - Unweighted Average Recall)")
    print(f"  * Exactitud Global:     {acc*100:.2f}%")

    print("\n--------------------------------------------------------------------")
    print(" REPORTE DETALLADO POR CLASE")
    print("--------------------------------------------------------------------")
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    print("--------------------------------------------------------------------")
    print(" MATRIZ DE CONFUSIÓN (Filas: Real, Columnas: Predicción)")
    print("--------------------------------------------------------------------")
    header = "          " + " ".join([f"{c[:6]:>7}" for c in classes])
    print(header)
    for idx, row in enumerate(cm):
        row_str = f"{classes[idx][:8]:<8} |" + " ".join([f"{val:7d}" for val in row])
        print(row_str)

    # Diagnóstico de Hipomimia: cuántas emociones no neutrales se clasificaron como Natural
    if "Natural" in classes:
        nat_idx = classes.index("Natural")
        total_non_nat = sum([cm[i].sum() for i in range(len(classes)) if i != nat_idx])
        misclassified_as_nat = sum([cm[i][nat_idx] for i in range(len(classes)) if i != nat_idx])
        hypomimia_rate = (misclassified_as_nat / total_non_nat) * 100 if total_non_nat > 0 else 0.0

        print("\n--------------------------------------------------------------------")
        print(" DIAGNÓSTICO CLÍNICO DE HIPOMIMIA (SESGO HACIA 'NATURAL')")
        print("--------------------------------------------------------------------")
        print(f"  * Muestras emocionales reales predichas erróneamente como Natural: {misclassified_as_nat} / {total_non_nat}")
        print(f"  * Tasa de confusión hacia Natural: {hypomimia_rate:.2f}%")
        if hypomimia_rate < 20.0:
            print("  * Evaluación: [BUENA] La red distingue con éxito expresiones atípicas de una cara neutra.")
        else:
            print("  * Evaluación: [ATENCIÓN] Hay confusión moderada con rostros neutros típica de la hipomimia.")

    print("====================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluador de Checkpoints FER en TEA")
    parser.add_argument(
        "--checkpoint",
        default="weights/fer_model_enet_b0_8_best_vgaf_best.pth",
        help="Ruta al archivo .pth"
    )
    parser.add_argument(
        "--backbone",
        choices=["enet_b0_8_best_vgaf", "mobilenetv4_conv_small", "fastvit_sa12"],
        default="enet_b0_8_best_vgaf"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, backbone=args.backbone, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
