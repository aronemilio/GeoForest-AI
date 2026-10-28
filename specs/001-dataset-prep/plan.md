# Plan de Desarrollo y Arquitectura: GeoForest-AI

## 1. Fases del Proyecto

### Fase 1: Entorno, Visualización Base y Recursos Cartográficos (Completada)
- Estructura modular (`app/`, `src/`, `data/`, `models/`, `specs/`, `tests/`).
- Dashboard en Streamlit con KPIs, filtros y gráficos de distribución.
- Visor geoespacial interactivo en Folium con integración de capas:
  - **Google Maps API (Satélite / Híbrido)** como recurso satelital primario de alta resolución.
  - Esri World Imagery, EOX Cloudless, OSM y OpenTopoMap como capas de referencia y respaldo.

### Fase 2: Pipeline de Datos Multiespectrales (Completada)
- Cargador de datos `OSCDDataset` en `src/dataset.py` (6 canales: B04, B08, B11 para T0 y T1).
- Descarga estructurada de Sentinel-2 L2A vía STAC / COG en `app/services/copernicus.py`.
- Integración del dataset real de Huánuco (6 zonas agroecológicas, 103 chips `256×256`).

### Fase 3: Modelo U-Net y Entrenamiento Anti-Sobreentrenamiento (Completada)
- Arquitectura `UNet` optimizada en `src/unet.py` (4.3M parámetros, reducción del 86.1%).
- Regularización con `Dropout2d(0.15)`, `FocalTverskyLoss` + `BCEWithLogits` + Label Smoothing.
- Entrenamiento multiépoca con lotes intercalados (Huánuco + OSCD) y aceleración GPU.
- Exportación del modelo definitivo a ONNX (`models/unet_geoforest.onnx`).

### Fase 4: Integración, Geobosques y Calibración Operativa (En curso)
- Conexión de máscaras de referencia de Geobosques (MINAM).
- Integración en tiempo real del motor de inferencia ONNX (`src/infer.py`) con la app interactiva.
- Consolidación del control de capas satelitales con Google Maps API en el frontend táctico.
