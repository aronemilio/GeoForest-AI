# Especificación Técnica: Pipeline de Detección Multitemporal (GeoForest-AI)

## 1. Visión General
**GeoForest-AI Huánuco** es un sistema experimental de alerta temprana diseñado para detectar cambios en la cobertura forestal, deforestación y quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga** (Huánuco, Perú).

## 2. Requisitos y Entradas del Pipeline

### 2.1 Datos Satelitales (Sentinel-2 L2A/L1C)
El sistema opera sobre pares temporales ($T_0$ y $T_1$) utilizando 3 bandas clave por fecha (total: 6 canales):
- **B04 (Rojo - 665 nm, 10m):** Absorción clorofílica; detección de suelo descubierto.
- **B08 (Infrarrojo Cercano NIR - 842 nm, 10m):** Vigor y estructura del dosel vegetal.
- **B11 (Infrarrojo de Onda Corta SWIR - 1610 nm, remuestreado a 10m):** Contenido de agua foliar, humedad de suelo y cicatrices de quema.

### 2.2 Índices Espectrales Calculados
- **NDVI:** $\frac{B08 - B04}{B08 + B04}$ (Vigor vegetativo).
- **NBR:** $\frac{B08 - B11}{B08 + B11}$ (Índice normalizado de quema).
- **dNDVI:** $\text{NDVI}_{T_0} - \text{NDVI}_{T_1}$ (Pérdida de biomasa verde).
- **dNBR:** $\text{NBR}_{T_0} - \text{NBR}_{T_1}$ (Severidad de quema).

## 3. Arquitectura del Modelo: U-Net (implementación `UNet`)
- **Canales de entrada:** 6 ($B04_{T_0}, B08_{T_0}, B11_{T_0}, B04_{T_1}, B08_{T_1}, B11_{T_1}$).
- **Encoder:** `inc` DoubleConv (32 canales) y 4 etapas `Down` (64, 128, 256, 512 canales) con MaxPool2d(2).
- **Bottleneck:** salida de `down4` (512 canales) antes del decoder.
- **Decoder:** 4 etapas `Up` (ConvTranspose2d + Skip Connections + DoubleConv) con 256, 128, 64 y 32 canales.
- **Capa final:** Conv2d $1 \times 1$; la salida del grafo son **logits**. La sigmoide se aplica en inferencia para obtener la probabilidad.
- **Canales de salida:** 1 (mapa de logits; tras sigmoide, $P \in [0.0, 1.0]$).

## 4. Función de Pérdida
$\mathcal{L}_{\text{total}} = 0.45\,\mathcal{L}_{\text{BCE}} + 0.55\,\mathcal{L}_{\text{Dice}}$ (`BCEDiceLoss`)
- **Binary Cross Entropy ($\mathcal{L}_{\text{BCE}}$):** Penaliza errores a nivel de píxel.
- **Dice Loss ($\mathcal{L}_{\text{Dice}}$):** Maneja el desbalance de clases extremo entre píxeles de cambio vs. bosque estable.

## 5. Exportación e Inferencia ONNX
- Checkpoint PyTorch: `models/unet_best.pth`.
- Formato ONNX: `models/unet_geoforest.onnx` (Opset **17**, dimensiones dinámicas `(batch, 6, height, width)`).
- Inferencia optimizada con `onnxruntime` mediante `ONNXForestPredictor`.

## 6. Visualización y Panel de Control (Streamlit / Folium)
- **Capas base satelitales:**
  - Esri World Imagery (Predeterminada).
  - EOX Sentinel-2 Cloudless (Sin marcas de agua).
  - OpenStreetMap & OpenTopoMap.
- **Filtros dinámicos:** Zona de estudio, tipo de evento, severidad.
- **Inferencia en vivo:** Ejecución directa del modelo ONNX sobre escenas reales, proyectando los candidatos de cambio en el visor de Huánuco; las salidas no están validadas como detecciones de deforestación.

## 7. Estado de Validación (2026-10-21)
- **Checkpoint actual:** `models/unet_best.pth`, época 8 de 12 (elegida por `combined_score` 50/50 OSCD–HCO pseudo-label), 7.75 M parámetros, FP32; ONNX opset 17.
- **Validación OSCD (Hong Kong, Mumbai, Rennes; 52 parches):** IoU 12.70 %, F1 22.54 %, precisión 39.69 %, recall 15.74 %. Son ciudades no vistas en entrenamiento, pero se usaron para seleccionar la época; **no es un test ciego**.
- **Huánuco:** IoU 65.07 %, F1 78.84 % sobre un pseudo-label ΔNDVI/ΔNBR/ATD usado también en entrenamiento; **no es validación independiente**.
- **Calibración medida (OSCD val, 3.407.872 píxeles-muestra; 52 parches con solape del 50 %):** Brier 0.0332 vs baseline 0.0348 (BSS 4.65 %), ECE-15 0.0251, MCE-15 0.4485, AUROC 0.8022, AP 0.2262. `p ≥ 0.80` → 44.49 % de precisión empírica; `p ≥ 0.90` → 50.55 %.
- **Confianza:** no calibrada. `calibrate_probabilities` solo normaliza scores; la mezcla p90/max y `confianza_pct` son heurísticas, no probabilidades de acierto.
- **Dictamen:** prototipo funcional en validación; no usar como detector de producción ni como evidencia pericial.
