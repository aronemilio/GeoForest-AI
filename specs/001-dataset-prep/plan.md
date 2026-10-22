# Plan de Desarrollo: GeoForest-AI Huánuco

> **Estado (2026-10-21):** Fases 1–4 implementadas; **Fase 5 en curso**. El ciclo expandido (12 épocas, mejor época 8) no supera el gate de validación independiente: OSCD F1 22.54 %, IoU 12.70 %; la métrica de Huánuco (F1 78.84 %, IoU 65.07 %) usa pseudo-etiquetas y no es validación independiente. La calibración OSCD muestra sobreconfianza en bins altos (`p ≥ 0.80` → 44.49 % de precisión empírica).

## Fases del Proyecto

### Fase 1: Arquitectura Base y Visualización Demostrativa
- [x] Estructurar repositorio (`app/`, `src/`, `data/`, `models/`, `specs/`).
- [x] Configurar panel de mando en español con Streamlit (`app/main.py`).
- [x] Implementar visor cartográfico interactivo con Folium centrado en Huánuco (Carpish y Valle del Huallaga).
- [x] Incorporar capas satelitales (Esri World Imagery) y delimitación de zonas.
- [x] Generar datos demostrativos iniciales y retirarlos de las vistas operativas al integrar datos reales; `src/sim/` se conserva para pruebas y fallback demostrativo.

### Fase 2: Ingesta de Datos y Preprocesamiento Espectral
- [x] Integrar dataset satelital de cambio (OSCD: Onera Satellite Change Detection) con bandas multiespectrales.
- [x] Implementar `src/dataset.py` para generación de parches multitemporales de 6 canales (B4, B8, B11 para T0 y T1).
- [x] Implementar `src/preprocess.py` con funciones radiométricas y cálculo de índices espectrales (NDVI, NBR, dNDVI, dNBR).

### Fase 3: Modelado Deep Learning y Entrenamiento Expandido (Cerrada como baseline; la calidad aún no pasa el gate)
- [x] Diseñar arquitectura U-Net convolucional (`src/unet.py`, `UNet`) con 6 canales de entrada multitemporal ($T_0$ y $T_1$).
- [x] Implementar función de pérdida combinada `BCEDiceLoss` (0.45 BCE + 0.55 Dice) en `src/train.py` para optimización directa de F1/IoU.
- [x] Implementar cargador de datos multitemporal real (`src/dataset.py`, `OSCDChangeDataset` y `load_huanuco_sentinel2_scene`).
- [x] **Ejecutar entrenamiento expandido multiescala (12 épocas, mejor época 8) y guardar checkpoint en `models/unet_best.pth`.**
  - *Métricas alcanzadas:* `HCO_F1 = 78.84%` (hasta 80.17% en época 12), `HCO_IoU = 65.07%` (hasta 66.90%); `OSCD_F1 = 22.54%`, `OSCD_IoU = 12.70%`.
  - *Calibración medida (OSCD val):* Brier 0.0332 (baseline 0.0348), ECE-15 0.0251, MCE-15 0.4485; `p ≥ 0.80` → 44.49% de precisión empírica; `p ≥ 0.90` → 50.55%. **No calibrada.**
  - *Dictamen de calidad:* Cerrada formalmente como ciclo baseline expandido; la calidad intrínseca del modelo puro sin filtro biofísico aún no pasa el gate operacional final (`>= 85% IoU` en validación independiente).
- [x] **Exportar pesos entrenados a formato ONNX opset 17 (`models/unet_geoforest.onnx`, 31.1 MB, FP32).**
- [x] Implementar pipeline de inferencia (`src/infer.py`: cálculo de certeza nuclear de alerta, umbralización adaptativa de Otsu y filtro biofísico $\Delta\text{NDVI}/\Delta\text{NBR}$).

### Fase 4: Inferencia ONNX, Integración Táctica y Panel en Vivo
- [x] Implementar motor de inferencia ONNX Runtime (`src/infer.py`, `ONNXForestPredictor`).
- [x] Desarrollar extracción y etiquetado de clusters de cambio espacial con cálculo de hectáreas.
- [x] Integrar la inferencia en `app/components/tab_inferencia.py` y `app/services/onnx_demo.py`.
- [x] Incorporar capa satelital EOX Sentinel-2 Cloudless en `app/main.py` como alternativa limpia sin marcas de agua.
- [x] Agregar sección de "Laboratorio de Inferencia U-Net en Vivo" en el panel.
- [x] Desarrollar suite de 92 pruebas unitarias automatizadas (`tests/`).

### Fase 5: Consolidación del Frontend Streamlit y Validación
- [x] Consolidar Streamlit como frontend principal único (`app/`); retirar referencias a un visor React externo.
- [x] Mantener backend ML/GIS desacoplado en `src/`.
- [ ] Recolección y descarga de pares adicionales de Sentinel-2 para temporadas secas y de lluvias en Huánuco.
- [ ] Integración con la capa de pérdida de bosque anual de Geobosques (MINAM) como referencia regional, no como verdad absoluta.
- [ ] Ajuste fino (fine-tuning) del modelo preentrenado con los datos específicos de bosque nuboso y quemas del Huallaga.
- [ ] Evaluación cuantitativa de métricas en Huánuco: F1-Score, Precision, Recall e IoU por tipo de evento.
