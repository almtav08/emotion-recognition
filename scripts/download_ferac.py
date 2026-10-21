"""
Script de Descarga y Verificación de FERAC
Dataset: Facial Emotion Recognition in Autistic Children (rajasreechaiti/ferac-dataset)
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FERAC_DIR = BASE_DIR / "data" / "raw" / "ferac"


def check_kaggle_token():
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    env_u = os.environ.get("KAGGLE_USERNAME")
    env_k = os.environ.get("KAGGLE_KEY")
    return token_path.exists() or (env_u and env_k)


def download_via_kaggle_api():
    RAW_FERAC_DIR.mkdir(parents=True, exist_ok=True)
    if not check_kaggle_token():
        return False
    try:
        import kaggle
        print("Credenciales de Kaggle detectadas. Descargando rajasreechaiti/ferac-dataset...")
        kaggle.api.dataset_download_files(
            "rajasreechaiti/ferac-dataset",
            path=str(RAW_FERAC_DIR),
            unzip=True,
            quiet=False
        )
        print(f"[OK] FERAC descargado con éxito en: {RAW_FERAC_DIR}")
        return True
    except Exception as e:
        print(f"[AVISO] Error al usar Kaggle API: {e}")
        return False


def verify_ferac_structure():
    expected_classes = ["Anger", "Fear", "Joy", "Natural"]
    found = {}
    for cls in expected_classes:
        matches = list(RAW_FERAC_DIR.rglob(cls))
        if matches:
            num_imgs = len([f for f in matches[0].iterdir() if f.is_file()])
            found[cls] = num_imgs
        else:
            found[cls] = 0

    print("\n--- Verificación del Dataset FERAC ---")
    all_ok = True
    for cls, count in found.items():
        status = f"[OK] ({count} imágenes)" if count > 0 else "[FALTA]"
        if count == 0:
            all_ok = False
        print(f"  * Clase {cls:<10}: {status}")

    if all_ok:
        print("\n[ÉXITO] Todas las clases de FERAC están listas.")
        print("Siguiente paso: uv run python scripts/preprocess_ferac.py")
    else:
        print("\n[PENDIENTE] Faltan imágenes de FERAC.")
        print_manual_instructions()
    return all_ok


def print_manual_instructions():
    print("""
========================================================================
 CÓMO DESCARGAR FERAC PASO A PASO:
========================================================================
MÉTODO 1 (Automático con Kaggle API):
  1. Entra en https://www.kaggle.com/ y ve a tu perfil -> Settings -> API -> Create New Token.
  2. Guarda el archivo 'kaggle.json' descargado en:
       mkdir -p ~/.kaggle
       mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
  3. Ejecuta:
       uv run python scripts/download_ferac.py

MÉTODO 2 (Descarga Manual Directa desde el Navegador):
  1. Abre el enlace:
       https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset
  2. Pulsa en 'Download' (se descargará un .zip con las 4 carpetas).
  3. Descomprime las carpetas ('Anger', 'Fear', 'Joy', 'Natural') en:
       /home/alex/emotion-recognition/data/raw/ferac/
  4. Vuelve a ejecutar:
       uv run python scripts/download_ferac.py
========================================================================
""")


def main():
    print("====================================================================")
    print(" GESTOR DE DESCARGA Y VERIFICACIÓN DE FERAC (AUTISMO - 4 CLASES)")
    print("====================================================================")

    if not verify_ferac_structure():
        print("\nIntentando descarga automática con Kaggle API...")
        if download_via_kaggle_api():
            verify_ferac_structure()


if __name__ == "__main__":
    main()
