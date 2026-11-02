# Guía de exposición — GeoForest-AI Huánuco

> Documento de preparación para la exposición ante jurados, SERFOR y FEMA. Esta guía detalla **el recorrido narrativo, qué destacar y cómo demostrar la plataforma**.
> **Advertencia de comunicación:** el score del modelo pasa por un escalado de temperatura post-entrenamiento (`p = σ(z/T)`) con `T = 0.3277` en `models/calibration.json`. Al ser `T < 1`, el escalado **agudiza** las probabilidades; no la describas como suavizado ni como mitigación de sobreconfianza. Checkpoint reportado: época 8 de 12 (el archivo `models/unet_best.pth` no está en este checkout). Validación OSCD por ciudades no vistas (usada para seleccionar la época; no es un test ciego): **F1 22.54 %, IoU 12.70 %, precisión 39.69 %, recall 15.74 %**. Calibración medida en OSCD: Brier **0.0332** (baseline 0.0348), ECE-15 **0.0251**, MCE-15 **0.4485**; `p ≥ 0.80` → **44.49 %** de precisión empírica y `p ≥ 0.90` → **50.55 %**. La métrica de Huánuco (F1 78.84 %, IoU 65.07 %) usa un pseudo-label generado con las mismas reglas del entrenamiento y **no es validación independiente**. Decir «prototipo en validación», no «detector validado». La serie de 8 semanas del Modo Pitch es **demostrativa** (bins fijos de 2026), y la columna «Score heurístico de persistencia espacial (%)» de las alertas es **heurística, no oficial ni calibrada**. No presentar cifras de 98–99 % como confianza del modelo. Las alertas PNCB son puntos oficiales, pero `area_ha` (`puntos × 0.12`), `confianza_pct` (logit constante 3.2), `ndvi_cambio` (`-0.30`), los deltas y el estado «Alerta Temprana Validada (MINAM)» son atributos derivados en `app/services/alerts_data.py`; el expediente no debe presentarse como validación oficial del PNCB.

## Checklist de arranque (10 min antes)

1. **Terminal — panel principal Streamlit:**
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   source .venv/bin/activate
   python scripts/verificar_pipeline.py   # debe reportar OK antes de la demo
   streamlit run app/main.py --server.headless true --server.port 8501
   ```
   Verificar que `http://127.0.0.1:8501` responda con HTTP 200.

2. **Navegador:** abrir `http://127.0.0.1:8501` en modo incógnito/limpio. Dejar que concluya la pantalla de bienvenida interactiva (~3 s).

3. **Verificar que no haya procesos huérfanos:**
   ```bash
   lsof -i :8501 || ss -tulpn | grep 8501
   ```

---

## Recorrido propuesto (≈ 10–12 min)

### A. Primeras 60 s — Pantallazo visual y narrativa territorial

- La **pantalla de bienvenida (splash)** presenta el isotipo forestal animado en CSS puro. Luego se despliega la **portada animada** (`app/assets/portada-bosque.mp4`) en bucle continuo de 8 segundos.
- Explicar el ámbito territorial prioritario:
  - **Bosque Montano de Carpish (Chinchao):** Ecosistema de neblina de alta fragilidad y endemismo.
  - **Valle del Huallaga (Chinchao, Churubamba, Acomayo):** Frente activo de expansión agroforestal y quemas.
- Destacar el origen de datos: **PNCB Geobosques (MINAM)** a partir de **Sentinel-2 (10 m)** e integración en vivo de focos de calor **NASA FIRMS (sensor VIIRS)**.

### B. KPIs y Consola SIG (Mapa Multicapa)

- **Tarjetas de Indicadores:**
  - *Alertas Satelitales Activas:* 2,842 eventos agrupados (2,865 puntos PNCB Geobosques).
  - *Área Afectada Registrada:* **no es una cifra oficial**: `app/services/alerts_data.py` calcula `area_ha = puntos × 0.12` (supuesto local).
  - *Severidad Alta:* Eventos clasificados por tamaño y prioridad para inspección FEMA.
  - *Score de Consistencia:* **no sale del modelo**: `alerts_data.py` usa un logit constante `3.2` (≈99.8 % en todas las alertas). La app lo rotula «Score heurístico (sin calibrar)»; no es una probabilidad calibrada ni una precisión validada. Medición OSCD aparte: `p ≥ 0.80` → 44.49 % de precisión empírica; `p ≥ 0.90` → 50.55 %.
- **Cartografía Folium Abierta (Sin API Keys bloqueantes):**
  - Mostrar la conmutación de capas: *Satélite real (Esri World Imagery)*, *Sentinel-2 Cloudless EOX* y *Focos Térmicos NASA VIIRS*. La capa «Copernicus CDSE» no existe en `map_view.py` (el parámetro `mostrar_sentinel_copernicus` no se usa); no anunciarla.
  - Inspeccionar un polígono de alerta sobre el mapa para mostrar el popup con identificador único (e.g. `PNCB-HCO-0001`), coordenadas UTM 18S y área.

### C. Consola de Misión Satelital (Laboratorio de Inferencia U-Net)

- Abrir la pestaña **🔬 Laboratorio de Inferencia**:
  - **Selector de Misiones Satelitales Reales:** Demostrar cómo conmutar entre las 5 escenas reales de Sentinel-2 L2A (`Huánuco Completa Tile 18LUQ`, `Carpish Norte`, `Valle Huallaga Sur`, `Aguas Claras` y `Brasilia`).
  - **Ficha de Telemetría Orbital:** Explicar los metadatos de la misión (resolución de 10 m/px, bandas B04 Rojo, B08 NIR y B11 SWIR, temporalidad multitemporal $T_0$ vs $T_1$).
  - **Previsualización Dual en Vivo:** Comparar las imágenes en falso color infrarrojo pre y post-evento.
- **Ejecución en Tiempo Real (CPU):**
  - Pulsar **«⚡ Ejecutar Inferencia U-Net en Tiempo Real»**.
  - Resaltar la **latencia de CPU**: ~40 ms para una inferencia ONNX de 256×256 en este equipo; el pipeline completo puede tardar más por E/S y post-proceso.
  - Explicar la **Umbralización Adaptativa de Otsu**: el algoritmo calcula automáticamente el corte bimodal óptimo entre fondo y cambio sin depender de umbrales manuales arbitrarios.
  - Destacar el **Filtro Biofísico Multitemporal**: filtra candidatos que no cumplen con caída de dosel ($\Delta\text{NDVI} \ge 0.10$) o incremento térmico/desecación ($\Delta\text{NBR} \ge 0.15$), con el objetivo de reducir falsos positivos.
  - Mostrar la cuadrícula de 4 vistas: Falso Color Infrarrojo, Índice NDVI, Heatmap de score normalizado (no calibrado; no interpretar como precisión) y Máscara Binaria de candidatos.

### D. Modo Pitch: Línea de Tiempo Semanal

- Cambiar a la vista **🌟 Presentación Ejecutiva Unificada**.
- Recorrer el deslizador temporal **demostrativo** de 8 semanas (bins fijos de agosto a septiembre de 2026; no son observaciones semanales reales), ilustrando cómo evoluciona la tasa semanal de deforestación y la respuesta de fiscalización.

### E. Cierre y Cadena de Custodia Digital

- Descargar la capa vectorial en **GeoJSON** para interoperabilidad con QGIS o ArcGIS.
- Descargar el **Expediente Técnico Digital**: destacar el bloque criptográfico con **hash SHA-256**, que permite verificar la integridad técnica y la trazabilidad; **no certifica la inmutabilidad de la evidencia ni sustituye una pericia oficial** de FEMA/SERFOR, y **no valida las alertas ante el PNCB** (los campos `area_ha`, `confianza_pct`, `ndvi_cambio` y `estado` son derivados locales).
- Respaldar la solidez del código: **188 pruebas automatizadas**; sin datos/modelos locales pasan **178** y las 10 restantes se **omiten** (`skipped`) con el motivo y el comando de regeneración (no afirmar «100 % de éxito» sin ejecutar el gate real).

---

## Puntos clave a destacar ante el jurado

- ✅ **Datos de entrada reales:** Las capas operativas usan Copernicus Sentinel-2 L2A, Geobosques MINAM y NASA FIRMS. El Modo Pitch conserva una serie demostrativa, y la columna «Score heurístico de persistencia espacial (%)» de las alertas es heurística, no oficial ni calibrada. No usar las cifras de 98–99 % como si fueran calibración del modelo.
- ✅ **Inferencia en CPU (prototipo):** El motor U-Net corre en ONNX Runtime optimizado (~40 ms por parche de 256×256 en este equipo), apto para servidores de borde o estaciones de campo sin GPU costosa.
- ✅ **Rigor Matemático:** Binarización adaptativa mediante el método de Otsu y filtrado biofísico multitemporal ($\Delta\text{NBR} / \Delta\text{NDVI}$).
- ✅ **Interoperabilidad GIS Completa:** Proyecciones UTM WGS84 EPSG:32718 y formatos estándar GeoTIFF / GeoJSON.

---

## Diagnóstico rápido si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| `No module named pytest` / `No module named rasterio` | Se activó `venv/`, que solo contiene `pip` | `source .venv/bin/activate` y repetir el comando. |
| `ModuleNotFoundError: No module named 'src'` | Script ejecutado sin la raíz en `sys.path` (ya corregido en `scripts/`) | Actualizar a la versión actual de `scripts/*.py` y ejecutar desde la raíz del repo. |
| `Falta el checkpoint real en models/unet_best.pth` | Artefactos de modelo no versionados | `python scripts/run_phase1_and_2.py` y luego `python scripts/export_onnx.py`. |
| `Falta manifiesto: data/processed/benchmark_scenes.json` | Fase 1 no ejecutada en este checkout | `python scripts/run_phase1_and_2.py`. |
| `0 >= 103` en pruebas de `gee_dataset` | `data/raw/sentinel2/` vacío (hoy esas pruebas se **omiten**, no fallan) | Descargar los pares Sentinel-2 con autorización y reejecutar; el mensaje del `skip` indica qué falta. |
| La previsualización Sentinel-2 «real» parece ruido | Fallback sintético de `cargar_escena_sentinel2_real` cuando faltan los GeoTIFF | No presentarla como real; cargar `data/raw/sentinel2/2024-07-18_*` y `2024-09-11_*`. El servicio ya la marca `es_simulado: True` (cambio en curso); verificar que la UI lo respete. |
| La inferencia devuelve 0 ha o un heatmap uniforme | Los ONNX servibles tienen salida casi constante (rango < 0.02): no están entrenados o el export está roto | No presentar máscaras como detección; ejecutar `python scripts/verificar_pipeline.py` y restaurar `models/unet_best.pth` / `models/unet_geoforest.onnx`. |
| Con umbral Manual 0.40 toda la escena queda como alerta | Las probabilidades rondan 0.465 (salida constante), por encima de 0.40 | No bajar el umbral por debajo de 0.5 ni interpretar ese resultado; el problema es el modelo. |
| «Otsu», «Filtro biofísico» y «Reescalar score» no cambian nada | En `tab_inferencia.py` son controles de UI sin llamadas a `compute_otsu_threshold`, `apply_biophysical_filter` ni `calibrate_probabilities` | No afirmar que están activos; pendiente del responsable de `app/components/**`. |
| El expediente dice «Alerta Temprana Validada (MINAM)» y 99.8 % | Campos derivados en `alerts_data.py` (logit constante, `puntos × 0.12`) | Presentarlos como estimación local, nunca como validación oficial del PNCB. |
| `pytest` reporta 10 `skipped` | Faltan datos/modelos no versionados; las pruebas del gate se omiten a propósito | No es un fallo: regenerar con `scripts/run_phase1_and_2.py` y `scripts/export_onnx.py`, y repetir. |
| Panel no carga / Connection refused | Servidor apagado | Arrancarlo con `streamlit run app/main.py --server.port 8501`. |
| Puerto 8501 ocupado | Proceso previo colgado | Matar proceso con `kill -9 $(lsof -t -i:8501)`. |
| Advertencia de caché | Recarga en curso | Recargar la pestaña del navegador con `Ctrl+Shift+R`. |
| Focos NASA no cargan | Sin conexión externa | El sistema usa la capa satelital local y caché automática sin detenerse. |

## Archivos clave de consulta

- `app/main.py` y `app/components/` (interfaz, Consola de Misión y modo Pitch).
- `app/services/` (`copernicus.py`, `alerts_data.py`, `expediente.py`, `nasa_firms.py`).
- `src/infer.py` (motor ONNX, Otsu, normalización de scores y filtro biofísico).
- `src/preprocess.py` (validación de rásteres, remuestreo explícito e índices espectrales).
- `tests/` (188 pruebas; 10 se omiten si faltan datos y modelos no versionados).
