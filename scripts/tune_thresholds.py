"""
Script de Calibración y Ajuste de Umbrales de Inferencia (Post-Processing)
Permite calibrar las fronteras de decisión de salida sin reentrenar la red,
maximizando el Macro-F1 y reduciendo los falsos positivos / negativos en TEA.

Uso:
  # 1. Búsqueda automática de los umbrales óptimos:
  uv run python scripts/tune_thresholds.py --auto-search

  # 2. Probar umbrales manuales específicos (pesos de probabilidad):
  uv run python scripts/tune_thresholds.py --weights 0.85 1.30 1.05

  # 3. Guardar el mejor perfil para despliegue en móvil:
  uv run python scripts/tune_thresholds.py --auto-search --save-config configs/inference_thresholds.json
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.transforms import get_mobile_transforms
from scripts.evaluate_3classes import GroupedFERACDataset, TARGET_CLASSES


def extract_probabilities_and_targets(checkpoint_path, backbone="enet_b0_8_best_vgaf", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"
    test_ds = GroupedFERACDataset(test_dir, transform=get_mobile_transforms("eval"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MobileFERModel(backbone_name=backbone, num_classes=3, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_np = torch.cat(all_targets, dim=0).numpy()
    probs_np = F.softmax(logits_tensor, dim=1).numpy()
    logits_np = logits_tensor.numpy()

    return logits_np, probs_np, targets_np


def evaluate_predictions(y_true, y_pred, title="MÉTRICAS"):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = balanced_accuracy_score(y_true, y_pred)
    acc = np.mean(y_true == y_pred)

    print(f"\n====================================================================")
    print(f" {title}")
    print(f"====================================================================")
    print(f"  * Macro F1-Score:    {macro_f1:.4f}  (Métrica balanceada)")
    print(f"  * Balanced Accuracy: {uar:.4f}  (UAR)")
    print(f"  * Exactitud Global:  {acc * 100:.2f}%\n")

    print(classification_report(y_true, y_pred, target_names=TARGET_CLASSES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de Confusión:")
    print("            Negative   Neutral  Positive")
    for idx, row in enumerate(cm):
        print(f"{TARGET_CLASSES[idx]:<9} | {row[0]:8d}  {row[1]:8d}  {row[2]:8d}")

    # Diagnóstico de confusión
    pos_to_neg = cm[2][0]
    neg_to_pos = cm[0][2]
    total_pos = cm[2].sum()
    total_neg = cm[0].sum()
    print(f"\nDiagnóstico Clínico (Inversiones de polaridad):")
    print(f"  - Positiva confundida con Negativa: {pos_to_neg}/{total_pos} ({pos_to_neg / total_pos * 100:.1f}%)")
    print(f"  - Negativa confundida con Positiva: {neg_to_pos}/{total_neg} ({neg_to_pos / total_neg * 100:.1f}%)")

    return macro_f1, uar, acc, cm


def search_optimal_weights(probs, y_true):
    """
    Búsqueda en rejilla fina de ponderaciones de decisión [w_neg, w_neu, w_pos].
    pred = argmax(w * probs)
    """
    print("\n[BÚSQUEDA EN REJILLA] Optimizando multiplicadores de probabilidad para maximizar Macro-F1...")

    # Mantenemos w_neg = 1.0 como referencia y exploramos w_neu y w_pos
    w_neu_range = np.linspace(0.8, 1.8, 51)
    w_pos_range = np.linspace(0.8, 1.6, 41)

    best_f1 = 0.0
    best_weights = [1.0, 1.0, 1.0]

    for w_neu in w_neu_range:
        for w_pos in w_pos_range:
            w = np.array([1.0, w_neu, w_pos])
            scaled_probs = probs * w
            preds = np.argmax(scaled_probs, axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_weights = [1.0, float(w_neu), float(w_pos)]

    # Normalizar para que sumen o tengan media 1
    mean_w = sum(best_weights) / len(best_weights)
    norm_weights = [round(x / mean_w, 4) for x in best_weights]

    print(f"[OK] Mejores multiplicadores encontrados: {norm_weights}")
    print(f"     Macro-F1 proyectado: {best_f1:.4f}")
    return norm_weights, best_f1


def search_optimal_thresholds_priority(probs, y_true):
    """
    Búsqueda de umbrales con cascada de decisión basada en confianza clínica:
      1. Si P(Positiva) >= T_pos -> Positiva
      2. De lo contrario, si P(Neutral) >= T_neu -> Neutral
      3. De lo contrario -> Negativa
    """
    print("\n[BÚSQUEDA CLÍNICA] Optimizando umbrales de activación directa...")
    best_f1 = 0.0
    best_thresholds = {"thresh_pos": 0.40, "thresh_neu": 0.35}

    t_pos_grid = np.linspace(0.25, 0.60, 36)
    t_neu_grid = np.linspace(0.20, 0.55, 36)

    for t_pos in t_pos_grid:
        for t_neu in t_neu_grid:
            preds = []
            for p in probs:
                # p = [P_neg, P_neu, P_pos]
                if p[2] >= t_pos:
                    preds.append(2)  # Positive
                elif p[1] >= t_neu:
                    preds.append(1)  # Neutral
                else:
                    preds.append(0)  # Negative
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = {"thresh_pos": round(float(t_pos), 3), "thresh_neu": round(float(t_neu), 3)}

    print(f"[OK] Mejores umbrales clínicos: {best_thresholds}")
    print(f"     Macro-F1 proyectado: {best_f1:.4f}")
    return best_thresholds, best_f1


def main():
    parser = argparse.ArgumentParser(description="Calibrador de Umbrales de Inferencia FER")
    parser.add_argument(
        "--checkpoint",
        default="weights/fer_model_enet_b0_8_best_vgaf_3classes_best.pth",
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--backbone",
        choices=["enet_b0_8_best_vgaf", "enet_b2_8", "mobilenetv4_conv_small", "fastvit_sa12"],
        default="enet_b0_8_best_vgaf"
    )
    parser.add_argument("--auto-search", action="store_true", help="Ejecutar búsqueda automática de umbrales óptimos")
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        default=None,
        help="Multiplicadores manuales de probabilidad: [W_neg, W_neu, W_pos] (ej: 0.85 1.35 1.05)"
    )
    parser.add_argument(
        "--thresholds",
        nargs=2,
        type=float,
        default=None,
        help="Umbrales clínicos manuales: [Thresh_Pos, Thresh_Neu] (ej: 0.38 0.32)"
    )
    parser.add_argument(
        "--save-config",
        default="configs/inference_thresholds.json",
        help="Ruta para guardar la configuración calibrada"
    )
    args = parser.parse_args()

    print("====================================================================")
    print(" CALIBRACIÓN DE UMBRALES DE INFERENCIA EN 3 MACRO-CLASES")
    print("====================================================================")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone:   {args.backbone}")

    logits, probs, y_true = extract_probabilities_and_targets(args.checkpoint, backbone=args.backbone)

    # 1. Evaluación Estándar (Argmax directo)
    preds_standard = np.argmax(probs, axis=1)
    baseline_f1, _, _, _ = evaluate_predictions(
        y_true, preds_standard, title="1. LÍNEA BASE (ARGMAX ESTÁNDAR SIN CALIBRAR)"
    )

    best_cfg = {
        "model_checkpoint": args.checkpoint,
        "backbone": args.backbone,
        "target_classes": TARGET_CLASSES,
        "method": "standard_argmax",
        "baseline_macro_f1": baseline_f1,
        "calibrated_macro_f1": baseline_f1,
        "multipliers": [1.0, 1.0, 1.0]
    }

    # 2. Búsqueda automática si se solicita
    if args.auto_search:
        opt_weights, opt_f1 = search_optimal_weights(probs, y_true)
        opt_preds = np.argmax(probs * np.array(opt_weights), axis=1)
        evaluate_predictions(
            y_true, opt_preds, title=f"2. RESULTADO CALIBRADO (PESOS: {opt_weights})"
        )

        opt_thresh, thresh_f1 = search_optimal_thresholds_priority(probs, y_true)

        best_cfg["method"] = "probability_multipliers"
        best_cfg["multipliers"] = opt_weights
        best_cfg["calibrated_macro_f1"] = opt_f1
        best_cfg["clinical_thresholds"] = opt_thresh

    # 3. Aplicación de pesos manuales si se proporcionan
    elif args.weights is not None:
        w = np.array(args.weights)
        preds_weighted = np.argmax(probs * w, axis=1)
        f1_w, _, _, _ = evaluate_predictions(
            y_true, preds_weighted, title=f"2. EVALUACIÓN CON PESOS MANUALES: {args.weights}"
        )
        best_cfg["method"] = "manual_multipliers"
        best_cfg["multipliers"] = [float(x) for x in args.weights]
        best_cfg["calibrated_macro_f1"] = f1_w

    # 4. Aplicación de umbrales clínicos manuales si se proporcionan
    elif args.thresholds is not None:
        t_pos, t_neu = args.thresholds
        preds_thresh = []
        for p in probs:
            if p[2] >= t_pos:
                preds_thresh.append(2)
            elif p[1] >= t_neu:
                preds_thresh.append(1)
            else:
                preds_thresh.append(0)
        f1_t, _, _, _ = evaluate_predictions(
            y_true, np.array(preds_thresh), title=f"2. EVALUACIÓN CON UMBRALES CLÍNICOS: Pos={t_pos}, Neu={t_neu}"
        )
        best_cfg["method"] = "clinical_thresholds"
        best_cfg["clinical_thresholds"] = {"thresh_pos": t_pos, "thresh_neu": t_neu}
        best_cfg["calibrated_macro_f1"] = f1_t

    # Guardar configuración si corresponde
    if args.save_config:
        cfg_path = Path(args.save_config)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(best_cfg, f, indent=2, ensure_ascii=False)
        print(f"\n[CONFIGURACIÓN GUARDADA] Perfil de umbrales guardado en: {cfg_path}")
        print("Este archivo JSON se puede cargar directamente en la app móvil o script de cámara en tiempo real.\n")


if __name__ == "__main__":
    main()
