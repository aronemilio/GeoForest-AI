# Lista de Tareas y Seguimiento Técnico

## Tareas Completadas

- [x] **TASK-001:** Crear estructura modular del proyecto (`app/`, `src/`, `data/`, `models/`, `specs/`, `tests/`).
- [x] **TASK-002:** Diseñar e implementar el dashboard en Streamlit con KPIs, filtros interactivos y gráficos analíticos.
- [x] **TASK-003:** Configurar mapa con Folium y capas satelitales (Esri World Imagery, OpenStreetMap, OpenTopoMap).
- [x] **TASK-004:** Implementar `src/dataset.py` con cargador para parches multitemporales de 6 canales (B04, B08, B11).
- [x] **TASK-005:** Definir la arquitectura `UNet` optimizada en `src/unet.py` (32 base features, bilinear, dropout 0.15).
- [x] **TASK-006:** Implementar pérdida combinada (`FocalTverskyLoss` + `BCELoss` + Label Smoothing) en `src/train.py`.
- [x] **TASK-007:** Ejecutar entrenamiento multiescala y resolver sobreentrenamiento mediante lotes intercalados y jitter radiométrico.
- [x] **TASK-008:** Exportar checkpoint entrenado a `models/unet_best.pth` y grafo ONNX a `models/unet_geoforest.onnx`.
- [x] **TASK-009:** Configurar carga segura `weights_only=True` en PyTorch y optimización de hilos en ONNX Runtime.
- [x] **TASK-010:** Implementar funciones de preprocesamiento espectral e índices (NDVI, NBR, dNDVI, dNBR) en `src/preprocess.py`.
- [x] **TASK-011:** Desarrollar el motor de inferencia ONNX Runtime con segmentación de polígonos en `src/infer.py`.
- [x] **TASK-012:** Integrar inferencia en `app/components/tab_inferencia.py` y proyección de resultados en el visor.
- [x] **TASK-013:** Integrar capa satelital EOX Sentinel-2 Cloudless en `app/main.py`.
- [x] **TASK-014:** Agregar módulo de inferencia en tiempo real en la interfaz de Streamlit.
- [x] **TASK-015:** Implementar suite completa de pruebas automatizadas en `tests/`.
- [x] **TASK-016:** Integrar consulta de catálogo STAC de Sentinel-2 L2A desde `app/services/copernicus.py`.
- [x] **TASK-017:** Integrar dataset multitemporal real de Huánuco (103 pares Sentinel-2 L2A).
- [x] **TASK-018:** Purgar archivos basura, logs antiguos (`train_*.log`), scripts obsoletos (`crew.py`, `inspect_data.py`) y cachés temporales.

## Tareas en Progreso / Siguientes

- [ ] **TASK-019:** Incorporar el recurso de **Google Maps Platform API** (`Google Satellite` y `Google Hybrid`) en `app/components/map_view.py` con lectura de `GOOGLE_MAPS_API_KEY` y fallback automático.
- [ ] **TASK-020:** Generar máscaras binarias de referencia a partir de los datos vectoriales de Geobosques (MINAM).
- [ ] **TASK-021:** Evaluar métricas finales de detección (Precision, Recall, F1-Score, IoU) y calibrar umbrales óptimos para quemas y deforestación en Huánuco.
