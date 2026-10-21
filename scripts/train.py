"""
Script de Entrenamiento y Exportación On-Device: Transfer Learning Directo en FERAC
Dataset: data/raw/ferac (train / test)
Modelo: MobileNetV4 (UIB) / FastViT
Exportación automática a ONNX con verificación de paridad numérica.
"""

import os
import sys
import copy
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import f1_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.losses import MultiClassFocalLoss
from src.transforms import get_mobile_transforms
from src.export import MobileModelExporter


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


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("====================================================================")
    print(" ENTRENAMIENTO ON-DEVICE FER EN TEA: TRANSFER LEARNING DIRECTO")
    print("====================================================================")
    print(f"Dispositivo activo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = BASE_DIR / "data" / "raw" / "ferac" / "train"
    test_dir = BASE_DIR / "data" / "raw" / "ferac" / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Directorios no encontrados: {train_dir} o {test_dir}")

    # Datasets
    train_ds = datasets.ImageFolder(str(train_dir), transform=get_mobile_transforms("train"))
    test_ds = datasets.ImageFolder(str(test_dir), transform=get_mobile_transforms("eval"))

    classes = train_ds.classes
    num_classes = len(classes)
    print(f"\nClases detectadas ({num_classes}): {classes}")
    print(f"Muestras de entrenamiento: {len(train_ds)}")
    print(f"Muestras de test/validación: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 1. Instanciación del Modelo
    print(f"\nInicializando arquitectura: {args.backbone} (preentrenada)...")
    affectnet_path = getattr(args, 'affectnet_weights', None)
    model = MobileFERModel(
        backbone_name=args.backbone,
        num_classes=num_classes,
        dropout=0.35,
        pretrained=True,
        affectnet_weights_path=affectnet_path
    )
    model.to(device)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    focal_criterion = MultiClassFocalLoss(gamma=2.0)

    # -------------------------------------------------------------
    # FASE 1A: Linear Probing (Warmup del nuevo cabezal de salida)
    # -------------------------------------------------------------
    print("\n[FASE 1A] Warmup de Cabezal (Linear Probing - 5 épocas, backbone 100% congelado)...")
    model.configure_freezing(mode="linear_probe")
    opt_warmup = build_optimizer(model, lr_backbone=0.0, lr_head=1e-3)

    for ep in range(1, 6):
        loss_tr = train_epoch(model, train_loader, focal_criterion, opt_warmup, scaler, device)
        loss_val, f1_val, _, _ = evaluate(model, test_loader, device)
        print(f"  Warmup [{ep}/5] | Train Focal-Loss: {loss_tr:.4f} | Test Loss: {loss_val:.4f} | Test Macro-F1: {f1_val:.4f}")

    # -------------------------------------------------------------
    # FASE 1B: Fine-Tuning de Extremos Profundos + Normalización
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
    print(f" ENTRENAMIENTO COMPLETADO | Mejor Macro-F1: {best_f1:.4f} (Época {best_epoch})")
    print("====================================================================")

    # Cargar los mejores pesos para evaluación final y exportación
    model.load_state_dict(best_weights)
    _, final_f1, final_targets, final_preds = evaluate(model, test_loader, device)

    print("\nReporte de Clasificación Detallado en Test:")
    print(classification_report(final_targets, final_preds, target_names=classes, digits=4))

    cm = confusion_matrix(final_targets, final_preds)
    print("Matriz de Confusión:")
    print(cm)

    # Guardar Checkpoint PyTorch
    weights_dir = BASE_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    pth_path = weights_dir / f"fer_model_{args.backbone}_best.pth"
    torch.save(best_weights, str(pth_path))
    print(f"\n[GUARDADO] Checkpoint PyTorch: {pth_path}")

    # -------------------------------------------------------------
    # FASE 2: Reparametrización, Exportación Móvil y Paridad Numérica
    # -------------------------------------------------------------
    print("\n[FASE 2] Exportación On-Device a ONNX con Reparametrización...")
    exporter = MobileModelExporter(model, input_shape=(1, 3, 224, 224))
    onnx_path = weights_dir / f"fer_mobile_{args.backbone}_tea.onnx"
    exporter.export_onnx(str(onnx_path))
    exporter.verify_numerical_parity(str(onnx_path))

    print("\n====================================================================")
    print(" [ÉXITO] Modelo listo para ser desplegado localmente en smartphone.")
    print(f"   * Archivo ONNX: {onnx_path}")
    print(f"   * Checkpoint:   {pth_path}")
    print("====================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Entrenador FER Móvil para TEA (Fine-Tuning desde AffectNet o ImageNet)")
    parser.add_argument(
        "--backbone",
        choices=["mobilenetv4_conv_small", "fastvit_sa12", "enet_b0_8_best_vgaf"],
        default="enet_b0_8_best_vgaf",
        help="Arquitectura base (enet_b0_8_best_vgaf contiene pesos preentrenados en AffectNet)"
    )
    parser.add_argument("--affectnet-weights", default="weights/affectnet_enet_b0_8_best_vgaf.pth", help="Ruta al checkpoint de AffectNet")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
