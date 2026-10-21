"""
Prueba Integral de Pipeline y Exportación On-Device
Valida la instanciación de MobileNetV4 y FastViT, la ejecución en GPU RTX 3080,
la reparametrización estructural, la exportación ONNX y la paridad numérica.
"""

import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import MobileFERModel
from src.losses import MultiClassFocalLoss
from src.export import MobileModelExporter


def test_mobilenetv4_pipeline():
    print("====================================================================")
    print(" 1. PRUEBA DE MOBILENETV4_CONV_SMALL (PRODUCCIÓN MÓVIL UNIVERSAL)")
    print("====================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo activo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = MobileFERModel(backbone_name="mobilenetv4_conv_small", num_classes=4, dropout=0.35, pretrained=True)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Salida del modelo (Batch=2, Clases=4): {output.shape}")
    assert output.shape == (2, 4), f"Dimensiones inesperadas: {output.shape}"

    # Probar Focal Loss
    criterion = MultiClassFocalLoss(gamma=2.0)
    dummy_targets = torch.tensor([0, 2], device=device)
    loss = criterion(output, dummy_targets)
    print(f"Cálculo de Focal Loss: {loss.item():.4f}")

    # Probar exportación a ONNX y verificación de paridad
    print("\nVerificando exportación a ONNX...")
    exporter = MobileModelExporter(model, input_shape=(1, 3, 224, 224))
    onnx_path = BASE_DIR / "weights" / "test_mobilenetv4.onnx"
    exporter.export_onnx(str(onnx_path))
    parity_ok = exporter.verify_numerical_parity(str(onnx_path))
    assert parity_ok, "Fallo en paridad numérica de ONNX"
    print("[OK] Pipeline de MobileNetV4 verificado con éxito.")


def test_fastvit_pipeline():
    print("\n====================================================================")
    print(" 2. PRUEBA DE FASTVIT_SA12 (REPARAMETRIZACIÓN ESTRUCTURAL)")
    print("====================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileFERModel(backbone_name="fastvit_sa12", num_classes=4, dropout=0.35, pretrained=True)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out_before = model(dummy_input)

    # Reparametrizar (fusión de ramas residuales RepMixer)
    print("Probando reparametrización estructural para móvil...")
    model.reparameterize_for_deployment()

    with torch.no_grad():
        out_after = model(dummy_input)

    diff = torch.max(torch.abs(out_before - out_after)).item()
    print(f"Diferencia máxima antes vs después de reparametrizar: {diff:.6e}")
    assert diff < 1e-4, "Discrepancia excesiva en reparametrización"
    print("[OK] FastViT reparametrizado y verificado con éxito.")


if __name__ == "__main__":
    test_mobilenetv4_pipeline()
    test_fastvit_pipeline()
    print("\n====================================================================")
    print(" [ÉXITO TOTAL] Todos los componentes del pipeline móvil funcionan.")
    print("====================================================================")
