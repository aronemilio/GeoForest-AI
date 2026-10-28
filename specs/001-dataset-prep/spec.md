# Especificación Técnica: Preparación del Dataset y Visor Geoespacial

## 1. Visión General
Sistema de monitoreo forestal y detección temprana de cambios en cobertura vegetal y quemas agrícolas para el Bosque Montano de Carpish y Valle del Huallaga (Huánuco, Perú).

## 2. Recursos y Fuentes de Datos
- **Sentinel-2 L2A (Copernicus / AWS S3 COG):**
  - Bandas espectrales clave: B04 (Rojo - 10m), B08 (NIR - 10m), B11 (SWIR - 20m remuestreada a 10m).
  - Parches multitemporales de 6 canales (T0 y T1).
- **Google Maps Platform API (Recurso Cartográfico Principal):**
  - **Google Maps Satellite & Hybrid Tiles API:** Integrado como capa base de alta resolución espacial para contraste óptico de detalle contra las escenas multiespectrales de Sentinel-2.
  - **Soporte de capas:**
    - `Google Satellite`: `https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}`
    - `Google Hybrid (Satélite + Toponimia/Vías)`: `https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}`
    - `Google Terrain`: `https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}`
  - **Credenciales y Configuración:** Gestión segura mediante variable de entorno `GOOGLE_MAPS_API_KEY` en `.env` y configuración en `app/.streamlit/secrets.toml`.
  - **Fallback de contingencia:** Esri World Imagery y EOX Sentinel-2 Cloudless en caso de indisponibilidad de cuota o clave.
- **Geobosques (MINAM):** Capas vectoriales de alertas tempranas de deforestación y pérdida de bosque húmedo amazónico.
- **Dataset OSCD (Onera Satellite Change Detection):** Dataset multiespectral de calibración y preentrenamiento.

## 3. Arquitectura del Modelo
- **Red:** U-Net (6 canales de entrada, 1 canal de salida con función de activación sigmoide en inferencia).
- **Parámetros optimizados:** `base_features=32`, `bilinear=True`, `Dropout2d(0.15)` en decodificador y bottleneck.
- **Función de pérdida:** Pérdida combinada `FocalTverskyLoss` + `BCELoss` con Label Smoothing.
- **Inferencia:** Exportación y ejecución acelerada mediante ONNX Runtime (`models/unet_geoforest.onnx`).

## 4. Requisitos de la Interfaz (Streamlit + Folium)
- KPIs de alertas y superficie estimada afectada (ha).
- Selector dinámico de capas satelitales incluyendo **Google Maps (Satélite / Híbrido)**, Esri World Imagery, OpenStreetMap y OpenTopoMap.
- Proyección de máscaras de cambio inferidas por ONNX sobre el lienzo cartográfico.

## 6. Protocolo de Alineamiento Técnico de Anotación
- **Sistema de Referencia Mandatorio:** EPSG:32718 (UTM Zona 18S). Reproyección automática en memoria para geometrías en WGS84 (EPSG:4326).
- **Unidad Mínima Cartográfica (UMC):** 500 m² (0.05 ha / 5 píxeles de Sentinel-2 10m). Descarte estricto de entidades sub-UMC.
- **Rasterización sin sesgo:** Regla `all_touched=False` para asignar target = 1 exclusivamente a píxeles cuyo centroide cae dentro del polígono delimitado.
- **Esquema de Atributos:** Validación obligatoria de `evento`, `confianza_validacion` (umbral >= 0.80), `fecha_t0`, `fecha_t1`, `agente_causante` y `validador`.
- **Control de Calidad Previo:** Ejecución de `src/validate_annotations.py` previo al empaquetado de lotes para entrenamiento.
