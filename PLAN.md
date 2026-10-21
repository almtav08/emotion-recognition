# Plan de Entrenamiento y Despliegue On-Device: Transfer Learning Directo (AffectNet -> FERAC) para Reconocimiento de Emociones en TEA en Dispositivos Móviles

## 1. Definición del Problema, Estrategia de Dominio y Restricciones On-Device

El reconocimiento de expresiones faciales (FER) en niños con Trastorno del Espectro Autista (TEA) ejecutado **localmente en un teléfono móvil** enfrenta una triple convergencia de desafíos técnicos:

1. **Brecha Expresiva (Neurodivergencia e Hipomimia):** Presencia frecuente de **hipomimia** (atenuación o retraso en la activación de *Action Units* faciales), asimetría hemifacial y activación atípica o sutil de los músculos cigomático mayor y orbicular de los ojos.
2. **Brecha Morfológica y Tamaño del Dataset:** El dataset objetivo **FERAC** (~770 imágenes de niños con TEA con 4 emociones: *Ira, Miedo, Alegría, Natural*) tiene un tamaño reducido. Para evitar el sobreajuste catastrófico (*overfitting*), se requiere transferencia de conocimiento directa desde un extractor facial general robusto (**AffectNet**).
3. **Restricciones de Hardware Móvil (Edge Computing):**
   * **Presupuesto Térmico y Batería:** La inferencia en tiempo real sobre el flujo de vídeo de la cámara puede provocar *thermal throttling* si la arquitectura no está optimizada.
   * **Presupuesto de Memoria y Latencia:** La latencia por fotograma debe situarse en **$\le 15-30\text{ ms}$ ($\ge 30\text{ FPS}$)** con un consumo de RAM del modelo inferior a **$25\text{ MB}$**.
   * **Aceleración Móvil Universal (NPU / GPU Edge):** Compatibilidad con Android NNAPI / Qualcomm QNN / Apple Neural Engine vía TFLite y CoreML mediante operadores convolucionales e INT8.

### Pipeline de Transferencia Directa en Dos Fases:
Se elimina la fase intermedia de EmoReact para simplificar radicalmente la adquisición de datos y acelerar el ciclo de despliegue, concentrando la adaptación morfológica en el aumento geométrico de datos y en la sintonización fina del último bloque extractor:

$$\mathcal{D}_{\text{AffectNet}} \text{ (Extracción Facial / Emociones Generales)} \xrightarrow[\text{1B: Fine-Tuning Suave + Focal Loss}]{\text{1A: Linear Probing Head}} \mathcal{D}_{\text{FERAC}} \text{ (Expresiones en TEA)} \xrightarrow{\text{Cuantización INT8}} \text{Mobile Engine (TFLite / CoreML)}$$

---

## 2. Selección de Arquitecturas Orientadas a Hardware Móvil

| Modelo (`timm` ID) | Tipo de Arquitectura | Parámetros | Tamaño FP32 | Tamaño INT8 Cuantizado | Latencia Estimada Móvil (NPU/CPU) | Idoneidad para Despliegue Móvil |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mobilenetv4_conv_small`** | CNN pura con bloques UIB (*Universal Inverted Bottlenecks*) | $\sim 3.8\text{M}$ | $\sim 15.2\text{ MB}$ | **$\sim 3.9\text{ MB}$** | **$6 - 12\text{ ms}$** | **Máxima recomendación para producción móvil.** Diseñado específicamente para aceleradores móviles (DSP/NPU). Cuantización INT8 perfecta sin pérdidas. |
| **`fastvit_sa12`** | ViT Híbrido con reparametrización estructural (*RepMixer*) | $\sim 11.6\text{M}$ | $\sim 46.4\text{ MB}$ | $\sim 11.8\text{ MB}$ | $14 - 24\text{ ms}$ | **Excelente balance precisión/velocidad.** Durante el entrenamiento usa ramas múltiples; antes de exportar a móvil se reparametriza en convoluciones $3\times 3$ fusionadas. |
| **`mobilenetv3_small_100`** | CNN con bloques Bottleneck + Squeeze-and-Excitation | $\sim 2.5\text{M}$ | $\sim 10.1\text{ MB}$ | **$\sim 2.6\text{ MB}$** | **$4 - 8\text{ ms}$** | Alternativa ultra-ligera para dispositivos de gama baja o cuando la app móvil deba coexistir con pipelines simultáneos pesados. |

---

## 3. Armonización de Datasets y Preprocesamiento en el Dispositivo

### 3.1. Pipeline de Inferencia en Tiempo Real en Móvil
En lugar de detectores pesados de escritorio, se integra **MediaPipe BlazeFace** directamente en el dispositivo móvil:

```
[Cámara Móvil: 30 FPS @ 720p/1080p]
       │
       ▼
[Etapa A: Detección Facial MediaPipe BlazeFace]
  ├── Extrae Bounding Box con margen dinámico (+12% perimetral)
  └── Extrae 5 Puntos Clave (Ojos, Nariz, Comisuras Labiales)
       │
       ▼ (Transformación de Similitud Afín: Rotación para alinear ojos en θ = 0°)
[Etapa B: Recorte Normalizado 224x224]
       │
       ▼
[Etapa C: Clasificador FER On-Device (MobileNetV4 INT8)]
  Inferencia en NPU/GPU (~8-15 ms) -> Logits [Anger, Fear, Joy, Natural]
       │
       ▼
[Etapa D: Filtro de Estabilidad Temporal (Anti-Flickering EMA)]
  Probabilidades suavizadas -> Feedback en la UI
```

### 3.2. Espacio de Etiquetas y Mapeo Directo
* **Base (AffectNet):** Representaciones aprendidas sobre rostros y emociones faciales universales.
* **Objetivo (FERAC - 4 Clases Discretas):**
  * `0: Anger` (Ira)
  * `1: Fear` (Miedo)
  * `2: Joy` (Alegría)
  * `3: Natural` (Expresión neutra o basal)

---

## 4. Protocolo de Entrenamiento: Transferencia Directa en Dos Fases

Dado que se realiza la transición directa de AffectNet a FERAC (sin la etapa intermedia infantil de EmoReact), el protocolo de entrenamiento se blinda para transferir las representaciones faciales sin colapso en el dataset pequeño:

### Fase 1A: Linear Probing (Épocas 1 a 5)
* **Congelación Total:** Se congela el $100\%$ de los parámetros del *backbone* preentrenado.
* **Entrenamiento Exclusivo del Cabezal:** Se entrena únicamente el bloque clasificador `LayerNorm -> Dropout(p=0.35) -> Linear(d, 4)` con una tasa de aprendizaje relativamente alta ($\text{lr} = 1 \times 10^{-3}$) y optimizador `AdamW`.
* **Justificación:** Si se calcularan gradientes sobre todo el modelo con los pesos aleatorios iniciales del nuevo cabezal, la retropropagación destruiría las características faciales preentrenadas en AffectNet.

### Fase 1B: Fine-Tuning de Extremos Profundos y Normalización (Épocas 6 a 30)
* **Descongelación Quirúrgica:** Se descongela únicamente el **último bloque del extractor** (`stages[-1]` en MobileNetV4 / FastViT) y todas las capas de `BatchNorm2d`.
* **Tasas Diferenciales Ultrabajas:**
  * $\text{lr}_{\text{backbone}} = 1.5 \times 10^{-5}$ con *Cosine Annealing Scheduler*.
  * $\text{lr}_{\text{head}} = 1 \times 10^{-4}$.
* **Función de Pérdida:** **Focal Loss Multiclase** ($\gamma = 2.0$) con ponderación inversa de clases:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  Penaliza fuertemente las confusiones hacia la clase `Natural` ante expresiones tenues o atípicas de `Fear` o `Anger`.
* **Data Augmentation Compensatorio de Morfología Infantil:**
  * *RandomCrop* a $224$ desde tamaño escalado de $236$.
  * *RandomHorizontalFlip* ($p=0.5$).
  * *RandomRotation* leve ($\pm 10^\circ$).
  * *ColorJitter* moderado (brillo $\pm 0.18$, contraste $\pm 0.18$) para simular variaciones de iluminación de cámara móvil.
  * *RandomErasing* pequeño ($0.02 - 0.08$) para forzar la lectura del tercio superior (ojos y cejas) y no solo de la boca.

---

## 5. Implementación Modular en PyTorch y Exportación Móvil

El proyecto dispone del código estructurado en módulos en `src/` y scripts en `scripts/`:
* `src/model.py`: Modelo `MobileFERModel` que admite `mobilenetv4_conv_small` y `fastvit_sa12`, con dummy pass dinámico para compatibilidad de dimensiones y reparametrización estructural.
* `src/losses.py`: Implementación de `MultiClassFocalLoss`.
* `src/transforms.py`: Transformaciones de entrenamiento y evaluación.
* `src/export.py`: Exportador automatizado a ONNX, TFLite (FP16 e INT8 con calibración) y CoreML, con verificación de paridad numérica.
* `scripts/train.py`: Orquestador de entrenamiento con `StratifiedKFold` (5 pliegues) y exportación automática al mejor modelo obtenido.

---

## 6. Métricas de Evaluación y Benchmarks On-Device

1. **Métricas Clínicas:**
   * **Macro F1-Score** y **Balanced Accuracy (UAR)** como métricas principales.
   * **Tasa de Falsos Negativos hacia Neutral (`Natural`):** Menor al $12\%$ en expresiones de Miedo e Ira.
2. **Benchmarks Físicos en Smartphone:**
   * **Latencia de Inferencia:** $\le 15\text{ ms}$ (en NPU/GPU móvil).
   * **Tamaño de Binario:** $< 5\text{ MB}$ (INT8 cuantizado) / $< 15\text{ MB}$ (FP16).
   * **Uso de RAM:** $< 25\text{ MB}$ de consumo pico en ejecución.
   * **Degradación por Cuantización:** $\Delta \text{Macro-F1} \le 1.0\%$ respecto al modelo FP32.

---

## 7. Despliegue en la Aplicación Móvil (Android / iOS)

1. **Suavizado Temporal Anti-Flickering:**
   $$P_t = \alpha \cdot p_t + (1 - \alpha) \cdot P_{t-1} \quad (\alpha \approx 0.30)$$
   Evita que micro-movimientos faciales o parpadeos causen cambios erráticos en la UI.
2. **Motores de Inferencia Soportados:**
   * **Android:** TensorFlow Lite con `GpuDelegate` o `QnnDelegate` (Qualcomm Neural Engine) / ONNX Runtime Mobile con NNAPI.
   * **iOS:** CoreML ejecutándose sobre el Apple Neural Engine (ANE).