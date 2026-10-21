"""
Script de Entrenamiento en 3 Macro-Clases de Emoción (TEA)
Agrupación:
  - Positivas: 'joy', 'surprise' -> 'Positive' (2)
  - Neutral:   'Natural'         -> 'Neutral'  (1)
  - Negativas: 'anger', 'fear', 'sadness' -> 'Negative' (0)

Fine-Tuning desde pesos preentrenados de AffectNet y auto-exportación a ONNX para móvil.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.metrics import f1_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.losses import MultiClassFocalLoss
from src.transforms import get_mobile_transforms
from src.export import MobileModelExporter

# Mapeo oficial de 6 clases a 3 macro-categorías
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
                # Búsqueda case-insensitive por si acaso
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


def build_optimizer(model, lr_backbone, lr_head, weight_decay=1e-2):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr_head})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets_dev)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    avg_loss = val_loss / len(loader.dataset)
    return avg_loss, macro_f1, np.array(all_targets), np.array(all_preds)


def run_training_3classes(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print(" ENTRENAMIENTO FER EN TEA: AGRUPACIÓN EN 3 MACRO-CLASES")
    print("   * Negative: anger, fear, sadness")
    print("   * Neutral:  Natural")
    print("   * Positive: joy, surprise")
    print("====================================================================")
    print(f"Dispositivo activo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = BASE_DIR / "data" / "raw" / "ferac" / "train"
    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"

    train_ds = GroupedFERACDataset(train_dir, transform=get_mobile_transforms("train"))
    test_ds = GroupedFERACDataset(test_dir, transform=get_mobile_transforms("eval"))

    # Conteo por clase
    train_counts = {c: 0 for c in TARGET_CLASSES}
    for _, t in train_ds.samples:
        train_counts[TARGET_CLASSES[t]] += 1

    test_counts = {c: 0 for c in TARGET_CLASSES}
    for _, t in test_ds.samples:
        test_counts[TARGET_CLASSES[t]] += 1

    print(f"\nDistribución Entrenamiento: {train_counts} (Total: {len(train_ds)})")
    print(f"Distribución Test:          {test_counts} (Total: {len(test_ds)})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Instanciación del Modelo con 3 Clases de Salida
    print(f"\nInicializando arquitectura: {args.backbone} (3 clases de salida)...")
    model = MobileFERModel(
        backbone_name=args.backbone,
        num_classes=3,
        dropout=0.35,
        pretrained=True,
        affectnet_weights_path=args.affectnet_weights
    )
    model.to(device)

    # Cálculo automático de pesos de balanceo con suavizado de exponente
    total_samples = len(train_ds)
    balance_power = 0.65  # Mayor compensación para clases minoritarias (Neutral y Positiva frente a Negativa)
    raw_weights = [(total_samples / train_counts[c]) ** balance_power for c in TARGET_CLASSES]
    mean_w = sum(raw_weights) / len(raw_weights)
    class_weights = [w / mean_w for w in raw_weights]
    alpha_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"Pesos de balanceo de clase suavizados (Alpha Focal Loss): {alpha_weights.cpu().numpy().round(3).tolist()}")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    focal_criterion = MultiClassFocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=args.label_smoothing)

    # -------------------------------------------------------------
    # FASE 1A: Linear Probing (Warmup del cabezal de 3 clases)
    # -------------------------------------------------------------
    print("\n[FASE 1A] Warmup de Cabezal (Linear Probing - 5 épocas, backbone 100% congelado)...")
    model.configure_freezing(mode="linear_probe")
    opt_warmup = build_optimizer(model, lr_backbone=0.0, lr_head=1e-3)

    for ep in range(1, 6):
        loss_tr = train_epoch(model, train_loader, focal_criterion, opt_warmup, scaler, device)
        loss_val, f1_val, _, _ = evaluate(model, test_loader, device)
        print(f"  Warmup [{ep}/5] | Train Focal-Loss: {loss_tr:.4f} | Test Loss: {loss_val:.4f} | Test Macro-F1: {f1_val:.4f}")

    # -------------------------------------------------------------
    # FASE 1B: Fine-Tuning Adaptativo (último bloque + BatchNorms)
    # -------------------------------------------------------------
    print(f"\n[FASE 1B] Fine-Tuning Adaptativo ({args.epochs} épocas, último bloque + BatchNorms)...")
    model.configure_freezing(mode="stage2_last_block")
    opt_finetune = build_optimizer(model, lr_backbone=1.5e-5, lr_head=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_finetune, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    best_weights = None
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        loss_tr = train_epoch(model, train_loader, focal_criterion, opt_finetune, scaler, device)
        scheduler.step()
        loss_val, f1_val, tgts, preds = evaluate(model, test_loader, device)

        is_best = f1_val > best_f1
        if is_best:
            best_f1 = f1_val
            best_epoch = ep
            best_weights = copy.deepcopy(model.state_dict())
            tag = " <-- MEJOR MODELO"
        else:
            tag = ""

        print(f"  Época [{ep:02d}/{args.epochs:02d}] | Train Loss: {loss_tr:.4f} | Test Loss: {loss_val:.4f} | Test Macro-F1: {f1_val:.4f}{tag}")

    print("\n====================================================================")
    print(f" ENTRENAMIENTO 3-CLASES COMPLETADO | Mejor Macro-F1: {best_f1:.4f} (Época {best_epoch})")
    print("====================================================================")

    # Cargar mejores pesos
    model.load_state_dict(best_weights)
    _, final_f1, final_targets, final_preds = evaluate(model, test_loader, device)

    print("\nReporte de Clasificación Detallado en Test (3 Macro-Clases):")
    print(classification_report(final_targets, final_preds, target_names=TARGET_CLASSES, digits=4))

    cm = confusion_matrix(final_targets, final_preds)
    print("Matriz de Confusión:")
    print("           Negat  Neutr  Posit")
    for idx, row in enumerate(cm):
        print(f"{TARGET_CLASSES[idx]:<8} | {row[0]:6d} {row[1]:6d} {row[2]:6d}")

    # Guardar Checkpoints y Exportar ONNX
    weights_dir = BASE_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pth_path = weights_dir / f"fer_model_{args.backbone}_3classes_best.pth"
    torch.save(best_weights, str(pth_path))
    print(f"\n[GUARDADO] Checkpoint PyTorch 3-Clases: {pth_path}")

    # Exportación On-Device del modelo específico
    print(f"\nExportando modelo {args.backbone} a ONNX (3 clases)...")
    exporter = MobileModelExporter(model, input_shape=(1, 3, 224, 224))
    onnx_path = weights_dir / f"fer_mobile_{args.backbone}_3classes_tea.onnx"
    exporter.export_onnx(str(onnx_path))
    exporter.verify_numerical_parity(str(onnx_path))

    # Actualizar checkpoint general solo si es el mejor modelo global
    global_best_path = weights_dir / "fer_model_3classes_best.pth"
    best_meta_path = weights_dir / "best_3classes_metric.txt"
    prev_best = 0.0
    if best_meta_path.exists():
        try:
            prev_best = float(best_meta_path.read_text().strip())
        except Exception:
            prev_best = 0.0

    if best_f1 >= prev_best:
        print(f"\n[RECORD] Nuevo récord global de Macro-F1: {best_f1:.4f} (anterior: {prev_best:.4f}). Actualizando fer_model_3classes_best.pth")
        torch.save(best_weights, str(global_best_path))
        best_meta_path.write_text(f"{best_f1:.4f}\n")
        exporter.export_onnx(str(weights_dir / "fer_mobile_3classes_tea.onnx"))
    else:
        print(f"\n[INFO] Macro-F1 ({best_f1:.4f}) no supera el récord histórico ({prev_best:.4f}). Se conserva el modelo líder.")

    print("\n====================================================================")
    print(f" [ÉXITO] Modelo ({args.backbone}) de 3 clases entrenado y exportado.")
    print(f"   * Checkpoint: {pth_path}")
    print(f"   * ONNX:       {onnx_path}")
    print("====================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Entrenador FER en 3 Macro-Clases (Positive, Neutral, Negative)")
    parser.add_argument(
        "--backbone",
        choices=["enet_b0_8_best_vgaf", "enet_b2_8", "mobilenetv4_conv_small", "fastvit_sa12"],
        default="enet_b2_8",
        help="Arquitectura base (por defecto: enet_b2_8 con pesos de AffectNet)"
    )
    parser.add_argument("--affectnet-weights", default=None, help="Ruta a pesos de AffectNet (se autoselecciona si es None)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Suavizado de etiquetas para mitigar sobreconfianza")
    args = parser.parse_args()

    if args.affectnet_weights is None:
        if args.backbone == "enet_b2_8":
            args.affectnet_weights = "weights/affectnet_enet_b2_8.pth"
        elif args.backbone == "enet_b0_8_best_vgaf":
            args.affectnet_weights = "weights/affectnet_enet_b0_8_best_vgaf.pth"

    run_training_3classes(args)


if __name__ == "__main__":
    main()
