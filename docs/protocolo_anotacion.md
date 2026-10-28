# Protocolo de Anotación y Validación de Deforestación: GeoForest-AI Huánuco

## 1. Objetivo
Estandarizar los criterios de digitalización vectorial de cambios en cobertura forestal (deforestación y quemas) sobre imágenes multitemporales Sentinel-2 L2A (Huánuco, Tile 18LUQ) para generar el Ground Truth independiente de entrenamiento y validación.

## 2. Configuración en SIG (QGIS)
- **Sistema de Coordenadas de Referencia (SCR):** EPSG:32718 (WGS 84 / UTM Zona 18S).
- **Composiciones de Bandas Sentinel-2 (T0 vs. T1):**
  - Falso Color Vegetación: B08 (NIR), B04 (Red), B03 (Green).
  - SWIR / Quemas: B12 (SWIR-2), B8A (NIR), B04 (Red).
- **Capas de Soporte:**
  - Máscara anual de bosque no intervenido (PNCB Geobosques - MINAM).
  - Mosaicos mensuales de alta resolución de referencia (Planet-NICFI / Google Satellite).

## 3. Criterios de Inclusión (Digitalizar)
- **Tala rasa y desbroce:** Remoción de dosel arbóreo que expone suelo desnudo o rastrojo.
- **Quemas agrícolas:** Cicatrices de ceniza oscura o carbón que penetran áreas de bosque.
- **Vías de penetración:** Trazados lineales detectables de caminos no consolidados dentro de masa boscosa.
- **Unidad Mínima Cartográfica:** Área continua ≥ 0.05 hectáreas (≥ 5 píxeles contiguos de 10 m).

## 4. Criterios de Exclusión (No Digitalizar)
- **Estiaje / Variación fenológica:** Amarillamiento o pérdida de follaje estacional sin remoción física de árboles.
- **Sombras:** Sombras orográficas en pendientes pronunciadas o proyecciones de nubes.
- **Áreas agrícolas consolidadas:** Rotación periódica de cultivos, cosechas o barbecho en parcelas preexistentes.
- **Dinámica fluvial:** Variaciones de cauce o playas en el Valle del Huallaga.

## 5. Estructura de Atributos del Vector
La capa exportada debe incluir la siguiente tabla de atributos:
- `id` (Integer): Identificador numérico único correlativo.
- `tipo_evento` (String): `deforestacion_tala`, `quema_agricola`, `via_apertura`.
- `confianza` (String): `alta` (verificada en alta resolución), `media` (borde difuso).
- `t0_fecha` (Date: YYYY-MM-DD): Fecha de la imagen base.
- `t1_fecha` (Date: YYYY-MM-DD): Fecha de la imagen con cambio confirmado.
- `validador` (String): Iniciales o identificador del operador.

## 6. Formato de Entrega y Control de Calidad
- Geometrías válidas (sin auto-intersecciones ni solapes entre polígonos).
- Formato: GeoJSON o GeoPackage (`.gpkg`).
- Ruta de almacenamiento: `data/raw/ground_truth/`.
