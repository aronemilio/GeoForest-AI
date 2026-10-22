# Memoria Técnica del Proyecto: GeoForest-AI Huánuco

## 1. Contexto y Objetivos Operativos
- **Proyecto:** GeoForest-AI Huánuco.
- **Objetivo:** Monitoreo y detección temprana de cambios en la cobertura forestal, deforestación y quemas agrícolas en el departamento de Huánuco, Perú, priorizando dos zonas clave:
  1. **Bosque Montano de Carpish (Chinchao):** Ecosistema de bosque nuboso de alta fragilidad y endemismo.
  2. **Valle del Huallaga (Chinchao, Churubamba, Acomayo):** Corredor agroforestal de transición hacia selva alta.

## 2. Pila Tecnológica
- **Lenguaje:** Python 3.10.20
- **Frameworks de ML/DL:** PyTorch 2.5 (CUDA 12.1), ONNX 1.23, ONNX Runtime 1.23
- **Librerías Geoespaciales:** Rasterio 1.4, GeoPandas 1.1, Shapely 2.1, PyProj 3.7
- **Interfaz y Cartografía:** Streamlit 1.65, Folium 0.20, Streamlit-Folium 0.26
- **Pruebas:** Pytest 9.1 (**92 pruebas automatizadas activas con 100% de éxito**)

## 3. Estado del Entrenamiento y Modelos

### 3.1 Pesos y Grafos ONNX
- **Dataset de Base:** OSCD (14 ciudades) + adaptación de dominio con par multitemporal real Sentinel-2 L2A Huánuco (Tile `18LUQ`: 2024-07-18 vs 2024-09-11) y alertas oficiales ATD de Geobosques (MINAM).
- **Checkpoints y Grafos Exportados:**
  - `models/unet_best.pth` (31.1 MB, FP32, `base_features=32`).
  - `models/unet_geoforest.onnx` (31.1 MB, FP32, opset 17, dynamic axes).
  - Enlaces simbólicos: `unet_sentinel2_huanuco.onnx` y `unet_sentinel2_multitemporal.onnx`.
  - Hashes criptográficos registrados en `models/manifest.json`.
  - **Trazabilidad pendiente:** `models/training_metrics.json` conserva rutas absolutas de otro checkout (`/home/emi/Documentos/GeoForest-AI`). Los SHA-256 coinciden con los artefactos actuales, pero el manifiesto debe regenerarse en esta ruta.

### 3.2 Motor de Inferencia (scores no calibrados) (`src/infer.py`)
- `ONNXForestPredictor`: Inferencia optimizada sobre CPU con configuración explícita de hilos (`intra_op=4`, `inter_op=1`).
- **Umbralización Adaptativa (Otsu):** Implementación vectorizada (`compute_otsu_threshold`) para corte bimodal dinámico entre fondo y deforestación.
- **Normalización de Scores:** Función `calibrate_probabilities` que reescala percentiles 5–99.5 a [0,1]. **No es calibración estadística** ni convierte el score en probabilidad de acierto.
- **Filtro Biofísico Multitemporal:** Función `apply_biophysical_filter` que exige caída de dosel ($\Delta\text{NDVI} \ge 0.10$) o pérdida de humedad/quema ($\Delta\text{NBR} \ge 0.15$), con el objetivo de reducir falsos positivos en bordes.
- **Extracción Vectorial:** `extract_alerts` genera polígonos GeoJSON con áreas métricas en hectáreas y severidades (`Alta`, `Media`, `Baja`).

### 3.3 Dictamen de la Fase 3: U-Net y Entrenamiento Expandido
- **Estado:** **Ciclo expandido ejecutado (12 épocas; mejor época 8 por `combined_score`).** La validación independiente OSCD sigue por debajo del gate operacional; la métrica HCO usa pseudo-label y no es independiente.
- **Métricas del ciclo expandido (mejor época 8 por `combined_score = 0.5*OSCD IoU + 0.5*HCO IoU`; el componente HCO es pseudo-label):**
  - Dominio Huánuco Sentinel-2 L2A (pseudo-label): `HCO_F1 = 78.84%` (hasta 80.17% en época 12), `HCO_IoU = 65.07%` (hasta 66.90%) (`models/training_metrics.json`).
  - Dominio OSCD (validación por ciudades no vistas; usada para seleccionar la época, no test ciego): `OSCD_IoU = 12.70%`, `OSCD_F1 = 22.54%`, precisión 39.69%, recall 15.74%.
- **Calibración medida en OSCD val (52 parches, 3.407.872 píxeles-muestra; solape del 50%):**
  - Brier `0.0332` vs baseline `0.0348` (BSS `4.65%`), ECE-15 `0.0251`, MCE-15 `0.4485`, AUROC `0.8022`, AP `0.2262`.
  - El ECE está dominado por el 96.4% de píxeles en el bin `[0.0, 0.1)`. En bins altos hay sobreconfianza: `p >= 0.80` → precisión empírica `44.49%` (score medio `0.8806`); `p >= 0.90` → `50.55%` (score medio `0.9365`).
  - **Lectura:** el score ordena mejor que azar (AUROC 0.80) pero **no es una probabilidad calibrada**. Los niveles ALTA/MEDIA/BAJA de la UI son umbrales heurísticos, no probabilidades de acierto.
- **Calibración aparente en pseudo-label Huánuco (no independiente):** Brier `0.0197` (BSS `65.15%`), ECE-15 `0.0062`, AUROC `0.9776`, AP `0.8501`; `p >= 0.80` → `89.12%` y `p >= 0.90` → `93.69%`. Esta cercanía es esperable porque el pseudo-label se construye con las mismas reglas ΔNDVI/ΔNBR/ATD usadas en la adaptación de dominio; **no demuestra confianza real**.
- **Sobre las cifras 98.1%–99.6%:** no son calibración del modelo. Provienen de la fórmula heurística `confianza_pct` (en `app/services/alerts_data.py`: `clip(85 + puntos*1.5, 82, 98.5)`) y/o de la mezcla `p90/max` de scores en la UI; no son probabilidades empíricas ni una validación independiente. No usarlas como claim de confianza.
- **Criterio del Gate de Calidad Operacional:**
  - Exige `IoU >= 85.0%` y `F1 >= 85.0%` en **validación independiente** y sin depender de reglas de descarte espectral heurístico.
  - Para la certificación final se requerirán series multitemporales de estaciones opuestas (estiaje vs. lluvias) y máscaras ráster multianuales de cobertura boscosa continua.

## 4. Visualizador y Consola C2 Streamlit

### 4.1 Datos de Entrada y Series Demostrativas
- **PNCB Geobosques (MINAM):** 2,853 alertas tempranas reales de deforestación en Huánuco cargadas desde `data/processed/vectores/atd_huanuco.geojson` mediante `obtener_alertas_reales_geobosques`.
- **NASA FIRMS VIIRS:** Consulta en tiempo real de anomalías térmicas y focos de calor vía API.
- **Alcance de datos reales:** Las vistas operativas usan alertas y escenas reales; el Modo Pitch conserva una serie demostrativa de 8 semanas. La columna interna `confianza_pct` (UI: «Score heurístico de persistencia espacial (%)») es una fórmula heurística basada en el número de puntos; **no es una confianza oficial del PNCB ni una probabilidad calibrada**.

### 4.2 Consola de Misión Satelital (`app/components/tab_inferencia.py`)
- **Catálogo de escenas:** `CATALOGO_ESCENAS_REALES` en `app/services/copernicus.py` ofrece 5 escenas multitemporales reales Sentinel-2 L2A (Huánuco Tile 18LUQ y ESA OSCD).
- **Telemetría Orbital:** Ficha interactiva con resolución (10 m/px), sensor MSI, fechas $T_0$ vs $T_1$ y cuadrículas ráster.
- **Previsualización Dual en Vivo:** Comparativa visual en falso color infrarrojo (B08/B04/B11) de $T_0$ vs $T_1$.
- **Expediente técnico:** Generación automatizada de informes con hash SHA-256 que permite verificar integridad y trazabilidad; **no sustituye una pericia oficial** de FEMA/SERFOR.
