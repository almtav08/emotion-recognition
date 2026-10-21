"""
Script de Evaluación para Modelo FER en 3 Macro-Clases (TEA)
Clases:
  - 0: Negative (anger, fear, sadness)
  - 1: Neutral  (Natural)
  - 2: Positive (joy, surprise)

Muestra reporte detallado, matriz de confusión y análisis clínico de valencia e hipomimia.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.transforms import get_mobile_transforms

FOLDER_TO_GROUP = {
    'anger': 'Negative',
    'fear': 'Negative',
    'sadness': 'Negative',
    'Natural': 'Neutral',
    'joy': 'Positive',
    'surprise': 'Positive'
}
TARGET_CLASSES = ['Negative', 'Neutral', 'Positive']
CLASS_TO_IDX = {name: i for i, name in enumerate(TARGET_CLASSES)}


class GroupedFERACDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = TARGET_CLASSES
        self.class_to_idx = CLASS_TO_IDX

        self.samples = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        for folder_name, group_name in FOLDER_TO_GROUP.items():
            folder_path = self.root_dir / folder_name
            if not folder_path.exists():
                matches = [p for p in self.root_dir.iterdir() if p.is_dir() and p.name.lower() == folder_name.lower()]
                if matches:
                    folder_path = matches[0]

            if folder_path.exists():
                group_idx = self.class_to_idx[group_name]
                for file_path in folder_path.iterdir():
                    if file_path.suffix.lower() in valid_exts:
                        self.samples.append((str(file_path), group_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def evaluate_3classes(checkpoint_path, backbone="enet_b0_8_best_vgaf", batch_size=16, thresholds_config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print(" EVALUACIÓN DE RENDIMIENTO EN 3 MACRO-CLASES (TEST SET FERAC)")
    print("   * Negative: anger, fear, sadness")
    print("   * Neutral:  Natural")
    print("   * Positive: joy, surprise")
    print("====================================================================")
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Checkpoint:  {checkpoint_path}")

    # Cargar configuración de umbrales si se especifica
    thresh_cfg = None
    if thresholds_config and Path(thresholds_config).exists():
        import json
        with open(thresholds_config, "r", encoding="utf-8") as f:
            thresh_cfg = json.load(f)
        multipliers = thresh_cfg.get("multipliers", [1.0, 1.0, 1.0])
        print(f"Umbrales Calibrados Activos: {thresholds_config}")
        print(f"  * Multiplicadores de probabilidad: {multipliers}")
    else:
        print("Modo de Decisión: Estándar (Argmax sin calibrar)")

    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"
    if not test_dir.exists():
        print(f"[!] Directorio de test no encontrado en: {test_dir}")
        return

    test_ds = GroupedFERACDataset(test_dir, transform=get_mobile_transforms("eval"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    counts = {c: 0 for c in TARGET_CLASSES}
    for _, t in test_ds.samples:
        counts[TARGET_CLASSES[t]] += 1
    print(f"Muestras de test por categoría: {counts} (Total: {len(test_ds)})")

    # Cargar modelo con 3 clases
    model = MobileFERModel(backbone_name=backbone, num_classes=3, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            if thresh_cfg and "multipliers" in thresh_cfg:
                scaled_probs = probs * np.array(thresh_cfg["multipliers"])
                preds = np.argmax(scaled_probs, axis=1)
            elif thresh_cfg and "clinical_thresholds" in thresh_cfg:
                t_pos = thresh_cfg["clinical_thresholds"]["thresh_pos"]
                t_neu = thresh_cfg["clinical_thresholds"]["thresh_neu"]
                preds = []
                for p in probs:
                    if p[2] >= t_pos:
                        preds.append(2)
                    elif p[1] >= t_neu:
                        preds.append(1)
                    else:
                        preds.append(0)
                preds = np.array(preds)
            else:
                preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    uar = balanced_accuracy_score(all_targets, all_preds)
    acc = np.mean(all_preds == all_targets)

    print("\n--------------------------------------------------------------------")
    print(" MÉTRICAS GLOBALES (3 MACRO-CLASES)")
    print("--------------------------------------------------------------------")
    print(f"  * Macro F1-Score:    {macro_f1:.4f}  (Métrica balanceada)")
    print(f"  * Balanced Accuracy: {uar:.4f}  (UAR)")
    print(f"  * Exactitud Global:  {acc*100:.2f}%")

    print("\n--------------------------------------------------------------------")
    print(" REPORTE DETALLADO POR MACRO-CLASE")
    print("--------------------------------------------------------------------")
    print(classification_report(all_targets, all_preds, target_names=TARGET_CLASSES, digits=4))

    cm = confusion_matrix(all_targets, all_preds)
    print("--------------------------------------------------------------------")
    print(" MATRIZ DE CONFUSIÓN (Filas: Real, Columnas: Predicción)")
    print("--------------------------------------------------------------------")
    print("            Negative   Neutral  Positive")
    for idx, row in enumerate(cm):
        print(f"{TARGET_CLASSES[idx]:<9} | {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")

    # Diagnósticos Clínicos
    neg_to_nat = cm[0][1]
    pos_to_nat = cm[2][1]
    total_neg = cm[0].sum()
    total_pos = cm[2].sum()
    inversion_neg_to_pos = cm[0][2]
    inversion_pos_to_neg = cm[2][0]

    print("\n--------------------------------------------------------------------")
    print(" ANÁLISIS CLÍNICO Y DE VALENCIA (TEA)")
    print("--------------------------------------------------------------------")
    print(f"  * Negativas confundidas con Neutral (Atenuación/Hipomimia): {neg_to_nat}/{total_neg} ({neg_to_nat/total_neg*100:.1f}%)")
    print(f"  * Positivas confundidas con Neutral (Atenuación/Hipomimia): {pos_to_nat}/{total_pos} ({pos_to_nat/total_pos*100:.1f}%)")
    print(f"  * Inversiones extremas de polaridad:")
    print(f"      - Negativa clasificada como Positiva: {inversion_neg_to_pos}/{total_neg} ({inversion_neg_to_pos/total_neg*100:.1f}%)")
    print(f"      - Positiva clasificada como Negativa: {inversion_pos_to_neg}/{total_pos} ({inversion_pos_to_neg/total_pos*100:.1f}%)")
    print("====================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluador FER en 3 Macro-Clases")
    parser.add_argument(
        "--checkpoint",
        default="weights/fer_model_3classes_best.pth",
        help="Ruta al checkpoint de 3 clases"
    )
    parser.add_argument(
        "--backbone",
        choices=["enet_b0_8_best_vgaf", "enet_b2_8", "mobilenetv4_conv_small", "fastvit_sa12"],
        default="enet_b0_8_best_vgaf"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--thresholds-config",
        default=None,
        help="Ruta al archivo JSON de umbrales calibrados (ej: configs/inference_thresholds.json)"
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Carga automáticamente 'configs/inference_thresholds.json' si existe"
    )
    args = parser.parse_args()

    thresholds_file = args.thresholds_config
    if args.calibrated and thresholds_file is None:
        default_cfg = Path("configs/inference_thresholds.json")
        if default_cfg.exists():
            thresholds_file = str(default_cfg)

    evaluate_3classes(
        args.checkpoint,
        backbone=args.backbone,
        batch_size=args.batch_size,
        thresholds_config=thresholds_file
    )


if __name__ == "__main__":
    main()
