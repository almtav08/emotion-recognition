"""
Evaluador de Ensamble Ligero (B0 + B2) para Detección de Emociones en TEA (3 Macro-Clases)
Combina las probabilidades de EfficientNet-B0 y EfficientNet-B2 preentrenados en AffectNet.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.transforms import get_mobile_transforms
from scripts.evaluate_3classes import GroupedFERACDataset, TARGET_CLASSES


def evaluate_dual_ensemble(b0_checkpoint, b2_checkpoint, weight_b0=0.85, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print(" EVALUACIÓN DE ENSAMBLE DUAL (B0 + B2) EN 3 MACRO-CLASES")
    print("====================================================================")
    print(f"Dispositivo activo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Ponderación: B0 = {weight_b0:.2f} | B2 = {1.0 - weight_b0:.2f}")

    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"
    test_ds = GroupedFERACDataset(test_dir, transform=get_mobile_transforms("eval"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Cargar B0
    m_b0 = MobileFERModel(backbone_name="enet_b0_8_best_vgaf", num_classes=3, pretrained=False)
    m_b0.load_state_dict(torch.load(b0_checkpoint, map_location=device, weights_only=True))
    m_b0.to(device).eval()

    # Cargar B2
    m_b2 = MobileFERModel(backbone_name="enet_b2_8", num_classes=3, pretrained=False)
    m_b2.load_state_dict(torch.load(b2_checkpoint, map_location=device, weights_only=True))
    m_b2.to(device).eval()

    probs_b0, probs_b2, all_targets = [], [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            out_b0 = m_b0(images).softmax(dim=1).cpu()
            out_b2 = m_b2(images).softmax(dim=1).cpu()
            probs_b0.append(out_b0)
            probs_b2.append(out_b2)
            all_targets.append(targets)

    p_b0 = torch.cat(probs_b0, dim=0).numpy()
    p_b2 = torch.cat(probs_b2, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    # Predicciones individuales
    preds_b0 = np.argmax(p_b0, axis=1)
    preds_b2 = np.argmax(p_b2, axis=1)
    f1_b0 = f1_score(y_true, preds_b0, average="macro")
    f1_b2 = f1_score(y_true, preds_b2, average="macro")

    # Predicción combinada
    p_ens = weight_b0 * p_b0 + (1.0 - weight_b0) * p_b2
    preds_ens = np.argmax(p_ens, axis=1)
    f1_ens = f1_score(y_true, preds_ens, average="macro")
    uar_ens = balanced_accuracy_score(y_true, preds_ens)
    acc_ens = np.mean(preds_ens == y_true)

    print("\n--------------------------------------------------------------------")
    print(" COMPARATIVA DE MODELOS")
    print("--------------------------------------------------------------------")
    print(f"  * EfficientNet-B0 (Individual): Macro-F1 = {f1_b0:.4f}")
    print(f"  * EfficientNet-B2 (Individual): Macro-F1 = {f1_b2:.4f}")
    print(f"  * Ensamble Dual Ponderado:      Macro-F1 = {f1_ens:.4f}")
    print(f"  * Balanced Accuracy (UAR):     {uar_ens:.4f}")
    print(f"  * Exactitud Global (Accuracy): {acc_ens * 100:.2f}%")

    print("\n--------------------------------------------------------------------")
    print(" REPORTE DETALLADO DEL ENSAMBLE (3 MACRO-CLASES)")
    print("--------------------------------------------------------------------")
    print(classification_report(y_true, preds_ens, target_names=TARGET_CLASSES, digits=4))

    cm = confusion_matrix(y_true, preds_ens)
    print("--------------------------------------------------------------------")
    print(" MATRIZ DE CONFUSIÓN DEL ENSAMBLE")
    print("--------------------------------------------------------------------")
    print("            Negative   Neutral  Positive")
    for idx, row in enumerate(cm):
        print(f"{TARGET_CLASSES[idx]:<9} | {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")

    return f1_ens, acc_ens


def main():
    parser = argparse.ArgumentParser(description="Evaluador de Ensamble Dual B0 + B2")
    parser.add_argument("--b0-checkpoint", default="weights/fer_model_enet_b0_8_best_vgaf_3classes_best.pth")
    parser.add_argument("--b2-checkpoint", default="weights/fer_model_enet_b2_8_3classes_best.pth")
    parser.add_argument("--weight-b0", type=float, default=0.85)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    evaluate_dual_ensemble(args.b0_checkpoint, args.b2_checkpoint, weight_b0=args.weight_b0, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
