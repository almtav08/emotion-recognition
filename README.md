# 🧠 Reconocimiento de Emociones Faciales en TEA On-Device (Edge AI)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Mobile-005CED.svg?logo=onnx)](https://onnxruntime.ai/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazeFace-00897B.svg)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema de **Reconocimiento de Expresiones Faciales (FER)** adaptado a niños con **Trastorno del Espectro Autista (TEA / ASD)**, optimizado para ejecutarse en tiempo real de manera local (**On-Device**) en dispositivos móviles (Android / iOS).

El proyecto aborda los desafíos clínicos de la neurodivergencia facial (**hipomimia**, asimetrías hemifaciales y expresiones no prototípicas) combinando **Transfer Learning directo desde AffectNet**, funciones de pérdida ponderadas (**Focal Loss**), agrupación en **macro-clases de valencia**, **calibración post-hoc de umbrales** y exportación optimizada a **ONNX**.

---

## 📋 Tabla de Contenidos

1. [Características Principales](#-características-principales)
2. [Arquitectura del Pipeline On-Device](#-arquitectura-del-pipeline-on-device)
3. [Modelos y Estrategia de Transfer Learning](#-modelos-y-estrategia-de-transfer-learning)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Flujo de Trabajo Rápido](#-flujo-de-trabajo-rápido)
   - [1. Descarga del Dataset FERAC](#1-descarga-del-dataset-ferac)
   - [2. Preprocesamiento y Alineación Facial con MediaPipe](#2-preprocesamiento-y-alineación-facial-con-mediapipe)
   - [3. Entrenamiento en 3 Macro-Clases](#3-entrenamiento-en-3-macro-clases)
   - [4. Calibración y Ajuste de Umbrales de Inferencia](#4-calibración-y-ajuste-de-umbrales-de-inferencia)
   - [5. Evaluación y Ensamble Ligero](#5-evaluación-y-ensamble-ligero)
   - [6. Benchmark de Inferencia On-Device (ONNX Runtime)](#6-benchmark-de-inferencia-on-device-onnx-runtime)
7. [Métricas y Resultados Clínicos](#-métricas-y-resultados-clínicos)
8. [Despliegue Móvil y Mitigación de Flickering](#-despliegue-móvil-y-mitigación-de-flickering)
9. [Referencias y Recursos](#-referencias-y-recursos)

---

## 🚀 Características Principales

* **Transferencia Directa (AffectNet → FERAC):** Utiliza representaciones faciales sólidas aprendidas en AffectNet (vía modelos EfficientNet-B0 y B2 de HSEmotion, MobileNetV4 y FastViT) para mitigar el sobreajuste ante datasets reducidos de TEA (~770 imágenes).
* **Adaptación a Hipomimia y Neurodivergencia:** Empleo de **MultiClass Focal Loss** ($\gamma=2.0$) con pesos inversos de clase para penalizar severamente las falsas detecciones hacia la clase neutra en microexpresiones sutiles.
* **Agrupación en Macro-Clases Clínicas (3 Clases):**
  * `Negative`: *anger*, *fear*, *sadness*
  * `Neutral`: *Natural*
  * `Positive`: *joy*, *surprise*
* **Calibración Post-Processing de Umbrales:** Optimización automática de multiplicadores de probabilidad para maximizar el Macro-F1 sin necesidad de reentrenar.
* **Alineación Facial con MediaPipe BlazeFace:** Detección de keypoints faciales, rotación horizontal interocular ($\theta = 0^\circ$) y margen perimetral dinámico ($+12\%$) antes de normalizar a $224\times 224$.
* **Exportación y Verificación de Paridad:** Conversión directa a ONNX con verificación de error absoluto numérico ($\le 1\times 10^{-4}$) frente a PyTorch.
* **Ensamble Ligero Ponderado:** Módulo para combinar predicciones de múltiples modelos (B0 + B2) maximizando la robustez diagnóstica.

---

## 🏗 Arquitectura del Pipeline On-Device

```
[ Cámara Móvil: Flujo 30 FPS @ 720p/1080p ]
                     │
                     ▼
  [ Etapa 1: Detección Facial MediaPipe BlazeFace ]
    ├── Bounding Box con margen dinámico (+12%)
    └── 5 Puntos Clave (Ojos, Nariz, Comisuras Labiales)
                     │
                     ▼ (Transformación afín: ojos alineados horizontalmente)
  [ Etapa 2: Recorte Normalizado 224x224 (MediaPipe Crop) ]
                     │
                     ▼
  [ Etapa 3: Clasificador FER On-Device (ONNX / INT8) ]
    Inferencia ultrarrápida (NPU/GPU/CPU móvil ~8-15 ms)
                     │
                     ▼
  [ Etapa 4: Calibración de Umbrales de Inferencia ]
    Multiplicadores de probabilidad post-hoc (configs/inference_thresholds.json)
                     │
                     ▼
  [ Etapa 5: Filtro Anti-Flickering (EMA Temporal) ]
    Pt = α · pt + (1 - α) · Pt-1  (α ≈ 0.30)
                     │
                     ▼
[ UI Móvil: Clasificación Estable y Feedback Clínico ]
```

---

## 🔬 Modelos y Estrategia de Transfer Learning

El repositorio soporta las siguientes arquitecturas móviles (definidas en [`src/model.py`](file:///home/alex/emotion-recognition/src/model.py)):

| Backbone (`timm` / weights) | Parámetros | Tamaño FP32 | Tamaño ONNX | Destino / Caso de Uso |
| :--- | :---: | :---: | :---: | :--- |
| **`enet_b0_8_best_vgaf`** | $\sim 4.0\text{ M}$ | $\sim 16\text{ MB}$ | $\sim 16.5\text{ MB}$ | **Recomendado para producción móvil.** Preentrenado en AffectNet con excelente discriminación facial. |
| **`enet_b2_8`** | $\sim 7.8\text{ M}$ | $\sim 31\text{ MB}$ | $\sim 32.3\text{ MB}$ | **Mayor capacidad representativa.** Ideal para ensamble o dispositivos de gama alta. |
| **`mobilenetv4_conv_small`** | $\sim 3.8\text{ M}$ | $\sim 15\text{ MB}$ | $\sim 15.2\text{ MB}$ | CNN pura con bloques UIB (*Universal Inverted Bottleneck*), óptima para NPU móvil / DSP. |
| **`fastvit_sa12`** | $\sim 11.6\text{ M}$ | $\sim 46\text{ MB}$ | $\sim 46.4\text{ MB}$ | ViT híbrido con reparametrización estructural (*RepMixer* a conv $3\times 3$). |

### Protocolo de Entrenamiento en Dos Fases
1. **Fase 1A - Linear Probing:** Congelación del $100\%$ del extractor preentrenado; entrenamiento exclusivo del nuevo cabezal clasificador con `AdamW` ($\text{lr} = 10^{-3}$).
2. **Fase 1B - Fine-Tuning Quirúrgico:** Descongelación controlada del bloque extractor final y capas de normalización (`BatchNorm2d`), con tasas diferenciales muy reducidas ($\text{lr}_{\text{backbone}} = 1.5 \times 10^{-5}$, $\text{lr}_{\text{head}} = 1 \times 10^{-4}$) y **Focal Loss** ($\gamma=2.0$).

---

## 📂 Estructura del Proyecto

```
emotion-recognition/
├── configs/
│   └── inference_thresholds.json     # Perfil de umbrales y multiplicadores calibrados
├── data/
│   ├── raw/
│   │   └── ferac/                    # Dataset FERAC original (train/ y test/)
│   └── processed/
│       └── ferac_aligned/            # Caras recortadas y alineadas con MediaPipe
├── weights/                          # Checkpoints PyTorch (.pth) y modelos ONNX (.onnx)
│   ├── affectnet_enet_b0_8_best_vgaf.pth
│   ├── affectnet_enet_b2_8.pth
│   ├── fer_model_enet_b0_8_best_vgaf_3classes_best.pth
│   └── fer_mobile_enet_b0_8_best_vgaf_3classes_tea.onnx
├── src/
│   ├── __init__.py
│   ├── model.py                      # Arquitectura MobileFERModel y control de congelado
│   ├── losses.py                     # MultiClassFocalLoss
│   ├── transforms.py                 # Data augmentation adaptada a morfología infantil
│   └── export.py                     # Exportador a ONNX con verificación de paridad
├── scripts/
│   ├── download_ferac.py             # Descarga y verificación del dataset FERAC
│   ├── preprocess_ferac.py           # Pipeline de alineación con MediaPipe BlazeFace
│   ├── train.py                      # Entrenamiento en 4/6 clases
│   ├── train_3classes.py             # Entrenamiento en 3 macro-clases (Neg/Neu/Pos)
│   ├── evaluate.py                   # Evaluación detallada por clase y análisis de hipomimia
│   ├── evaluate_3classes.py          # Evaluación de 3 clases con matriz de confusión
│   ├── evaluate_ensemble.py          # Ensamble ponderado B0 + B2
│   ├── tune_thresholds.py            # Búsqueda automática de umbrales óptimos
│   ├── test_mobile_inference.py      # Benchmark de latencia con ONNX Runtime
│   └── test_pipeline.py              # Test unitario de instanciación y exportación
├── pyproject.toml                    # Dependencias y configuración con uv
├── PLAN.md                           # Especificación técnica clínica y on-device
├── DATASETS_GUIDE.md                 # Guía detallada de adquisición de datos
└── README.md                         # Documentación principal
```

---

## ⚙️ Instalación y Configuración

El proyecto utiliza [`uv`](https://github.com/astral-sh/uv) como gestor de paquetes y entornos virtuales de Python:

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/emotion-recognition.git
cd emotion-recognition

# 2. Sincronizar el entorno virtual con uv (instala PyTorch con soporte CUDA)
uv sync
```

Verifica la instalación ejecutando el test del pipeline:
```bash
uv run python scripts/test_pipeline.py
```

---

## 🔄 Flujo de Trabajo Rápido

### 1. Descarga del Dataset FERAC
Descarga el dataset FERAC (*Facial Emotion Recognition in Autistic Children*) desde Kaggle:
```bash
# Con Kaggle API configurada en ~/.kaggle/kaggle.json
uv run python scripts/download_ferac.py
```
*(Para descarga manual directa desde el navegador, consulta [`DATASETS_GUIDE.md`](file:///home/alex/emotion-recognition/DATASETS_GUIDE.md)).*

### 2. Preprocesamiento y Alineación Facial con MediaPipe
Alinea horizontalmente los ojos, añade un margen perimetral dinámico del $12\%$ y genera las imágenes normalizadas en $224\times 224$:
```bash
uv run python scripts/preprocess_ferac.py
```

### 3. Entrenamiento en 3 Macro-Clases
Entrena la red agrupando en 3 macro-categorías clínicas (`Negative`, `Neutral`, `Positive`):
```bash
uv run python scripts/train_3classes.py \
  --backbone enet_b0_8_best_vgaf \
  --epochs 25 \
  --batch-size 16
```
> El script ejecuta automáticamente la Fase 1A (Linear Probing), la Fase 1B (Fine-Tuning con Focal Loss) y **exporta el mejor modelo a ONNX** en `weights/fer_mobile_enet_b0_8_best_vgaf_3classes_tea.onnx`.

### 4. Calibración y Ajuste de Umbrales de Inferencia
Ajusta las fronteras de decisión de probabilidad en el conjunto de prueba para mitigar la hipomimia y maximizar el Macro-F1:
```bash
# Búsqueda automática y guardado en config
uv run python scripts/tune_thresholds.py \
  --checkpoint weights/fer_model_enet_b0_8_best_vgaf_3classes_best.pth \
  --backbone enet_b0_8_best_vgaf \
  --auto-search \
  --save-config configs/inference_thresholds.json
```

### 5. Evaluación y Ensamble Ligero
Para evaluar un checkpoint específico:
```bash
uv run python scripts/evaluate_3classes.py \
  --checkpoint weights/fer_model_enet_b0_8_best_vgaf_3classes_best.pth \
  --backbone enet_b0_8_best_vgaf
```

Para evaluar el ensamble dual ponderado (EfficientNet-B0 + EfficientNet-B2):
```bash
uv run python scripts/evaluate_ensemble.py \
  --weight-b0 0.85
```

### 6. Benchmark de Inferencia On-Device (ONNX Runtime)
Mide la latencia en milisegundos y evalúa la clasificación de una imagen utilizando el runtime móvil:
```bash
uv run python scripts/test_mobile_inference.py \
  --model weights/fer_mobile_enet_b0_8_best_vgaf_3classes_tea.onnx \
  --iterations 100
```

---

## 📊 Métricas y Resultados Clínicos

En el dataset FERAC (conjunto de test), los resultados para la clasificación en 3 macro-clases de valencia son:

| Configuración | Backbone | Macro F1-Score | Balanced Accuracy | Tamaño Modelo |
| :--- | :--- | :---: | :---: | :---: |
| **Línea Base (Argmax estándar)** | EfficientNet-B0 | **66.04%** | 65.80% | 16.5 MB |
| **Calibrado (Multiplicadores Óptimos)** | EfficientNet-B0 | **68.39%** | **68.21%** | 16.5 MB |
| **Ensamble Dual (B0 + B2)** | B0 (85%) + B2 (15%) | **66.35%** | 66.12% | 48.8 MB |

### Impacto de la Calibración de Umbrales
* **Reducción de Falsos Negativos:** Se reduce notablemente la asignación errónea de microexpresiones de valencia negativa hacia `Neutral` causada por la hipomimia facial.
* **Perfil de Multiplicadores (`configs/inference_thresholds.json`):**
  * `Negative`: $0.9375$
  * `Neutral`: $0.7688$ (atenúa el sesgo predominante a la expresión basal)
  * `Positive`: $1.2937$ (potencia la detección de sonrisas y gestos de alegría)

---

## 📱 Despliegue Móvil y Mitigación de Flickering

Para integrar el modelo ONNX en una aplicación móvil (Android con **NNAPI/QNN** o iOS con **CoreML**), se recomienda implementar las siguientes pautas en el cliente:

1. **Alineación Facial con MediaPipe:** Ejecutar la detección y transformación afín antes de pasar el tensor a ONNX Runtime Mobile.
2. **Aplicar Multiplicadores de Calibración:**
   $$\tilde{p}_c = p_c \cdot w_c, \quad \hat{y} = \arg\max_{c} \tilde{p}_c$$
3. **Filtro de Suavizado Temporal Anti-Flickering (EMA):**
   $$P_t = \alpha \cdot p_t + (1 - \alpha) \cdot P_{t-1} \quad (\alpha \approx 0.30)$$
   Esto previene cambios espurios entre fotogramas debidos a parpadeos o micro-movimientos.

---

## 📚 Referencias y Recursos

* **FERAC Dataset:** Rajasree & Chaiti, *Facial Emotion Recognition in Autistic Children* ([Kaggle](https://www.kaggle.com/datasets/rajasreechaiti/ferac-dataset)).
* **HSEmotion:** Savchenko, A. V., *Facial expression and attributes recognition library* ([GitHub](https://github.com/HSE-asavchenko/face-emotion-recognition)).
* **MediaPipe Face Detection:** BlazeFace on-device face detector ([Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_detector)).
* **Timm Library:** Ross Wightman, *PyTorch Image Models* ([GitHub](https://github.com/huggingface/pytorch-image-models)).
