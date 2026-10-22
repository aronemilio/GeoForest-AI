# Lista de Tareas y Seguimiento Técnico

## Tareas Completadas

- [x] **TASK-001:** Crear estructura modular del proyecto (`app/`, `src/`, `data/`, `models/`, `specs/`, `tests/`).
- [x] **TASK-002:** Diseñar e implementar el dashboard en Streamlit con KPIs, filtros interactivos y gráficos analíticos.
- [x] **TASK-003:** Configurar mapa con Folium y capas satelitales (Esri World Imagery, OpenStreetMap, OpenTopoMap).
- [x] **TASK-004:** Implementar `src/dataset.py` con cargador `OSCDChangeDataset` para parches multitemporales de 6 canales.
- [x] **TASK-005:** Definir la arquitectura `UNet` en `src/unet.py` con 6 canales de entrada, 1 canal de salida en logits y sigmoide en inferencia.
- [x] **TASK-006:** Implementar `BCEDiceLoss` (0.45 BCE + 0.55 Dice) y bucle de entrenamiento en `src/train.py`.
- [x] **TASK-007:** Ejecutar entrenamiento expandido multiescala de la U-Net (12 épocas, AdamW, `BCEDiceLoss`). **Resultados: `HCO_F1: 78.84%`, `HCO_IoU: 65.07%` (mejor época 8; hasta 80.17%/66.90% en época 12). Validación independiente OSCD: `F1 22.54%`, `IoU 12.70%`. Calibración OSCD: Brier 0.0332, ECE-15 0.0251, MCE-15 0.4485; `p ≥ 0.80` → 44.49% de precisión empírica. No calibrada; las cifras de 98–99% de la UI son heurísticas.**
- [x] **TASK-008:** Exportar checkpoint entrenado a `models/unet_best.pth` y grafo ONNX opset 17 a `models/unet_geoforest.onnx` (31.1 MB, FP32).
- [x] **TASK-009:** Configurar carga segura `weights_only=True` en PyTorch y optimización de hilos en ONNX Runtime.
- [x] **TASK-010:** Implementar funciones de preprocesamiento espectral e índices (NDVI, NBR, dNDVI, dNBR) en `src/preprocess.py`.
- [x] **TASK-011:** Desarrollar el motor de inferencia ONNX Runtime con segmentación de componentes en `src/infer.py`.
- [x] **TASK-012:** Integrar la inferencia en `app/components/tab_inferencia.py` y `app/services/onnx_demo.py` para proyectar resultados en el visor.
- [x] **TASK-013:** Integrar capa satelital EOX Sentinel-2 Cloudless en `app/main.py`.
- [x] **TASK-014:** Agregar módulo de inferencia en tiempo real en la interfaz de Streamlit.
- [x] **TASK-015:** Implementar suite de **92 pruebas** automatizadas en `tests/` (`test_unet.py`, `test_infer.py`, `test_preprocess.py`, `test_dataset.py`, `test_sim.py`).
- [x] **TASK-016:** Integrar consulta de catálogo STAC de Sentinel-2 L2A desde `app/services/copernicus.py` (sin API keys) y carga de GeoTIFFs locales; la descarga automática de escenas no está implementada.
- [x] **TASK-017:** Descargar manualmente e integrar el par multitemporal real sobre Huánuco (T0: 2024-07-18 vs T1: 2024-09-11) en `data/raw/sentinel2/` y conectarlo a la UI.

## Tareas en Progreso / Siguientes

- [ ] **TASK-018:** Generar máscaras binarias de referencia a partir de los datos vectoriales de Geobosques (MINAM).
- [ ] **TASK-019:** Realizar fine-tuning del modelo `models/unet_best.pth` sobre las escenas locales de Huánuco.
- [ ] **TASK-020:** Evaluar métricas finales de detección (Precision, Recall, F1-Score, IoU) y calibrar umbrales óptimos para quemas y deforestación.
