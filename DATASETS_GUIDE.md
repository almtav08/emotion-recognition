# Guía de Obtención y Preparación de Datasets (Pipeline Directo AffectNet -> FERAC)

Tras adaptar el plan para eliminar la etapa intermedia de EmoReact, la arquitectura realiza una **transferencia directa en dos fases**:

$$\text{Pesos Base Faciales (AffectNet)} \xrightarrow{\text{Transferencia Directa (Linear Probe + Fine-Tuning)}} \text{FERAC (TEA)} \xrightarrow{\text{Cuantización INT8}} \text{Móvil}$$

Esto reduce el requerimiento de datos exclusivamente a **FERAC** (~770 imágenes de niños con TEA divididas en 4 clases: `Anger`, `Fear`, `Joy`, `Natural`).

---

## 1. Estructura de Directorios

```
emotion-recognition/
├── data/
│   ├── raw/
│   │   └── ferac/              <-- Coloca aquí las 4 carpetas originales (Anger, Fear, Joy, Natural)
│   └── processed/
│       └── ferac_aligned/      <-- Salida procesada y normalizada con MediaPipe (224x224)
├── weights/                    <-- Pesos y modelos exportados (.pth, .onnx, .tflite)
├── src/                        <-- Código fuente (modelo móvil, Focal Loss, exportador)
└── scripts/                    <-- Scripts de descarga, preprocesamiento y entrenamiento
```

---

## 2. Cómo Descargar FERAC

### Método A: Descarga Automática con Kaggle API
Si tienes cuenta en Kaggle:
1. Ve a [Kaggle](https://www.kaggle.com/) -> icono de perfil -> **Settings** -> sección **API** -> haz clic en **Create New Token**.
2. Guarda el archivo `kaggle.json` descargado en `~/.kaggle/`:
   ```bash
   mkdir -p ~/.kaggle
   mv /ruta/a/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Ejecuta el script de descarga:
   ```bash
   uv run python scripts/download_ferac.py
   ```

### Método B: Descarga Manual Directa desde el Navegador (Sin API)
1. Entra directamente a la página del dataset en Kaggle:
   👉 **[https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset](https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset)**
2. Haz clic en el botón blanco **Download (ZIP)**.
3. Descomprime el archivo descargado. Verás 4 carpetas:
   * `Anger`
   * `Fear`
   * `Joy`
   * `Natural`
4. Cópialas o muévelas a:
   ```bash
   emotion-recognition/data/raw/ferac/
   ```
5. Comprueba que están en su sitio ejecutando:
   ```bash
   uv run python scripts/download_ferac.py
   ```

---

## 3. Pasos a Ejecutar tras Descargar FERAC

Una vez descargadas las imágenes en `data/raw/ferac/`:

### Paso 1: Preprocesamiento y Alineación con MediaPipe
```bash
uv run python scripts/preprocess_ferac.py
```
*Detecta los rostros con MediaPipe BlazeFace, alinea los ojos en horizontal ($\theta = 0^\circ$), añade un $12\%$ de margen dinámico perimetral y guarda las caras en $224\times 224$ píxeles en `data/processed/ferac_aligned/`.*

### Paso 2: Entrenamiento y Exportación Móvil Automática
```bash
uv run python scripts/train.py --backbone mobilenetv4_conv_small --epochs 25
```
*Ejecuta la Fase 1A (Linear Probing), la Fase 1B (Fine-Tuning con Focal Loss en GPU RTX 3080) y exporta automáticamente el mejor modelo a formato ONNX optimizado para móvil en `weights/fer_mobile_mobilenetv4_conv_small_tea.onnx`.*
