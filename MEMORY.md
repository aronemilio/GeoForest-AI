# Memoria Técnica del Proyecto: GeoForest-AI Huánuco

## 1. Contexto y Objetivos Operativos
- **Proyecto:** GeoForest-AI Huánuco.
- **Objetivo:** Monitoreo y detección temprana de cambios en la cobertura forestal, deforestación y quemas agrícolas en el departamento de Huánuco, Perú, priorizando dos zonas clave:
  1. **Bosque Montano de Carpish (Chinchao):** Ecosistema de bosque nuboso de alta fragilidad y endemismo.
  2. **Valle del Huallaga (Chinchao, Churubamba, Acomayo):** Corredor agroforestal de transición hacia selva alta.

## 2. Pila Tecnológica
- **Lenguaje:** Python 3.14.8 (`.venv`; `venv/` solo contiene `pip`)
- **Frameworks de ML/DL:** PyTorch 2.14.1+cpu, ONNX 1.23.2, ONNX Runtime 1.31.0
- **Librerías Geoespaciales:** Rasterio 1.5.2, GeoPandas 1.2.0, Shapely 2.2.0, PyProj 3.8.0
- **Interfaz y Cartografía:** Streamlit 1.66.0, Folium 0.20.0, Streamlit-Folium 0.27.4
- **Pruebas:** Pytest 9.1 (**188 pruebas**; 178 pasan sin datos/modelos locales y 10 se omiten si faltan los artefactos no versionados de `data/` y `models/`)

## 3. Estado del Entrenamiento y Modelos

### 3.1 Pesos y Grafos ONNX
- **Dataset de Base:** OSCD (14 ciudades) + adaptación de dominio con par multitemporal real Sentinel-2 L2A Huánuco (Tile `18LUQ`: 2024-07-18 vs 2024-09-11) y alertas oficiales ATD de Geobosques (MINAM).
- **Checkpoints y Grafos Exportados:**
  - `models/unet_best.pth` (31.1 MB, FP32, `base_features=32`) — **ausente en este checkout**.
  - `models/unet_geoforest.onnx` (31.1 MB, FP32, opset 17, dynamic axes) — **ausente en este checkout**.
  - Presentes solo como archivos independientes (no enlaces simbólicos): `unet_sentinel2_huanuco.onnx` (`7a0e60d7…`) y `unet_sentinel2_multitemporal.onnx` (`5ecf644e…`), usados por `app/services/onnx_demo.py` y `tab_inferencia.py`.
  - `models/manifest.json` registra hashes de artefactos ausentes (`19929711…` para el `.pth`, `5df72419…` para el ONNX), por lo que **no verifica** los archivos presentes.
  - **Crítico:** un smoke test con entrada aleatoria (`scripts/verificar_pipeline.py`) muestra que ambos ONNX servibles devuelven una salida **casi constante** (rango 0.0146 y 0.0189, σ ≈ 0.002): no discriminan cambio. Las probabilidades quedan en ≈0.465, por debajo de 0.5 (0 detecciones) y por encima de 0.40 (toda la escena como alerta si se baja el umbral). No son los artefactos que produjeron las métricas de `reports/benchmark_operational_report.json`.
  - Regeneración: `python scripts/run_phase1_and_2.py` (datos → entrenamiento) y `python scripts/export_onnx.py` (checkpoint → ONNX + manifiesto). Ambos scripts fallaban con `ModuleNotFoundError: No module named 'src'` al ejecutarse desde la raíz; corregido con bootstrap de `sys.path`.
  - **Trazabilidad pendiente:** `models/training_metrics.json` conserva rutas absolutas de otro checkout (`/home/emi/Documentos/GeoForest-AI`). Los SHA-256 no coinciden con los artefactos actuales; regenerar el manifiesto tras reexportar.

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
- **PNCB Geobosques (MINAM):** 2,865 puntos oficiales en `data/processed/vectores/atd_huanuco.geojson`, agrupados en 2,842 eventos por fecha y celda de ~100 m (`obtener_alertas_reales_geobosques`).
- **Atributos derivados (no oficiales):** en `alerts_data.py`, `area_ha = puntos × 0.12`, `confianza_pct` proviene de un logit constante `3.2` (≈99.8 % para toda alerta), `ndvi_cambio = -0.30`, deltas fijos y `estado = «Alerta Temprana Validada (MINAM)»`. `expediente.py` los reproduce y afirma «registros oficiales validados por PNCB Geobosques»: **no son datos oficiales ni una validación del PNCB**.
- **NASA FIRMS VIIRS:** Consulta en tiempo real de anomalías térmicas y focos de calor vía API.
- **Alcance de datos reales:** Las vistas operativas usan alertas y escenas reales; el Modo Pitch conserva una serie demostrativa de 8 semanas. La columna `confianza_pct` (UI: «Score heurístico de persistencia espacial (%)») **no depende de los puntos ni del modelo**: es un logit constante convertido con la temperatura, por lo que vale ≈99.8 % en todas las alertas; no es una confianza oficial del PNCB ni una probabilidad calibrada.

### 4.2 Consola de Misión Satelital (`app/components/tab_inferencia.py`)
- **Catálogo de escenas:** `CATALOGO_ESCENAS_REALES` en `app/services/copernicus.py` ofrece 5 escenas multitemporales reales Sentinel-2 L2A (Huánuco Tile 18LUQ y ESA OSCD).
- **Telemetría Orbital:** Ficha interactiva con resolución (10 m/px), sensor MSI, fechas $T_0$ vs $T_1$ y cuadrículas ráster.
- **Previsualización Dual en Vivo:** Comparativa visual en falso color infrarrojo (B08/B04/B11) de $T_0$ vs $T_1$.
- **Expediente técnico:** Generación automatizada de informes con hash SHA-256 que permite verificar integridad y trazabilidad; **no sustituye una pericia oficial** de FEMA/SERFOR.
