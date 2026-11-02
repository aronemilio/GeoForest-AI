# 🌲 GeoForest-AI Huánuco

Sistema de alerta temprana e inferencia geoespacial de aprendizaje profundo para identificar pérdidas en la cobertura forestal y quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga**, departamento de Huánuco, Perú.

> **Procedencia de datos y estado del modelo.** La plataforma se fundamenta **exclusivamente en datos satelitales y vectoriales reales**:
> - **Alertas Tempranas de Deforestación (ATD):** 2,842 eventos agrupados a partir de los 2,865 puntos oficiales del **Programa Nacional de Conservación de Bosques (PNCB — Geobosques MINAM)** mediante Sentinel-2 (10 m) en el corredor Carpish–Huallaga. La app agrupa puntos por fecha y celda de ~100 m; `area_ha`, `confianza_pct`, `ndvi_cambio` y el estado son atributos **derivados localmente**, no cifras oficiales.
> - **Imágenes Satelitales Multitemporales:** Coberturas orbitales reales Sentinel-2 L2A de la Agencia Espacial Europea (**ESA / Copernicus CDSE**) sobre Huánuco (Tile MGRS `18LUQ`: 2024-07-18 vs 2024-09-11) y pares reales del dataset **ESA OSCD** (Aguas Claras, Brasilia, París, Nantes, etc.). La validación independiente emplea las ciudades OSCD Hong Kong, Mumbai y Rennes.
> - **Focos de Calor en Tiempo Real:** Detecciones térmicas activas integradas vía API directa de **NASA FIRMS (sensor VIIRS)**.
> - **Motor de Inferencia U-Net:** Ejecución sobre CPU con **ONNX Runtime**, umbralización adaptativa bimodal (**método de Otsu**), calibración post-entrenamiento (**Temperature Scaling**, `T = 0.3277` en `models/calibration.json`), normalización de scores y filtro biofísico multitemporal ($\Delta\text{NBR} \ge 0.15$ / $\Delta\text{NDVI} \ge 0.10$). Las piezas están implementadas en `src/infer.py`; la consola de inferencia aún no las conecta (ver «Consideraciones operativas»). La suite tiene **188 pruebas**; sin los datos y modelos locales no versionados pasan **178** y las **10 restantes se omiten** con un mensaje que indica qué regenerar (ver paso 5 de «Instalación y ejecución»).

> **Estrategias Matemáticas y Pipeline de Entrenamiento (Fase 5):**
> - **Focal Tversky Loss (`FocalTverskyLoss`):** Función de pérdida asimétrica en PyTorch ($\alpha=0.3, \beta=0.7, \gamma=1.333$) que penaliza masivamente los Falsos Negativos para forzar un alto Recall ante el severo desbalance de la clase deforestación (<4% de los píxeles).
> - **Aumento de Datos Geométrico Agresivo y Radiométrico:** Transformaciones espaciales profundas (`albumentations`: `RandomRotate90`, volteos H/V, `RandomResizedCrop`, `ElasticTransform` / `GridDistortion`) y jitter radiométrico multitemporal para evitar que la red aprenda atajos espectrales directos y forzar el aprendizaje de patrones geométricos (trochas, claros de parcelas).
> - **Calibración con Temperature Scaling (`src/infer.py`):** Reescalado de logits $p = \sigma(z/T)$ ajustado por optimización sobre validación. El artefacto vigente (`models/calibration.json`) registra `T = 0.3277`: al ser $T < 1$ el escalado **agudiza** las probabilidades, no las suaviza; no debe describirse como mitigación de sobreconfianza sin una medición Brier/ECE independiente.

> **Estado del modelo y métricas de validación:** El modelo está entrenado sobre pares Sentinel-2 y adaptado con la escena multitemporal de Huánuco. Las métricas de validación independiente en ciudades OSCD no vistas reportan IoU y F1 representativos para detección de cambio. El escalado de temperatura del artefacto vigente ($T = 0.3277$) agudiza las salidas; no es un suavizado ni una garantía de calibración. La columna «Score heurístico de persistencia espacial (%)» de las alertas no mide persistencia ni densidad: es un logit constante (≈99.8 %) calculado en `app/services/alerts_data.py`.

---

## 🚀 Estado de los módulos desarrollados

### 1. Panel de monitoreo e interfaz modular (`app/`)

Dashboard interactivo en Streamlit con tema «Niebla de dosel» (verde forestal táctico + dark glassmorphism). Arquitectura modular desacoplada:

- **🌟 Presentación Ejecutiva Unificada (Pitch Interactivo)**: Flujo ejecutivo para jurados, SERFOR o fiscalías ambientales con narrativa territorial, línea de tiempo semanal multitemporal de 8 semanas sobre el corredor Carpish–Huallaga, métricas de área afectada y exportación de expedientes técnicos con hash criptográfico SHA-256.
- **🗺️ Consola Integral SIG & Fiscalización**: Visualización cartográfica con Folium multicapa (Esri World Imagery, Sentinel-2 Cloudless EOX, Copernicus CDSE, NASA FIRMS VIIRS, CartoDB, OSM), polígonos de alerta georreferenciados en EPSG:32718 y auditoría de parámetros.
- **🛰️ Consola de Misión Satelital (Laboratorio de Inferencia)**: Selector visual de misiones satelitales reales con ficha de telemetría orbital en vivo, previsualización dual multitemporal ($T_0$ vs $T_1$) en falso color infrarrojo (B08/B04/B11) y ejecución de inferencia U-Net en tiempo real con diagnóstico de latencia en milisegundos.

Animaciones: pantalla de bienvenida (splash decorativo CSS puro) y portada animada (`app/assets/portada-bosque.mp4`, bucle sin costura).

### 2. Preprocesamiento geoespacial GIS (`src/preprocess.py`)

- Validación rigurosa de GeoTIFFs (CRS, resolución espacial, consistencia de cuadrícula `GridSpec`).
- Alineación y remuestreo explícito de bandas Sentinel-2 (B04 rojo, B08 NIR, B11 SWIR), distinguiendo datos continuos (bilineal) de categóricos (vecino más cercano).
- Cálculo vectorizado de índices espectrales: **NDVI** (vegetación) y **NBR** (humedad y quemas).
- Rasterización de vectores de referencia bajo reglas de etiquetado explícitas (`LabelRule`).
- Metadatos de procedencia reproducibles en formato JSON.

### 3. Series temporales multitemporales y métricas (`src/sim/`)

- `src/sim/simulador.py`: generador determinista de series temporales multitemporales con PRNG Mulberry32.
- `src/sim/metricas.py`: benchmark de referencia y especificaciones técnicas del motor (IoU, F1, latencia, matriz de confusión en hectáreas).
- Hub de paquete `src/__init__.py` reexporta `preprocess`, `unet`, `infer` y `sim`.

### 4. Arquitectura U-Net (`src/unet.py`)

- Arquitectura convolucional U-Net en PyTorch para datos multiespectrales de 6 canales ($T_0$: B04, B08, B11; $T_1$: B04, B08, B11).
- Skip connections, normalización por lotes (`BatchNorm2d`) y capa de salida de 1 canal que produce **logits**; la sigmoide se aplica en inferencia tras el escalado de temperatura.
- Exportador a formato estándar **ONNX** con dimensiones dinámicas en batch, alto y ancho (`export_unet_to_onnx`).
- Checkpoint: **7.75 M parámetros**, ~31 MB FP32; ONNX **opset 17** con ejes dinámicos (`base_features=32`).

### 5. Motor de inferencia ONNX (`src/infer.py`)

- Predictor `ONNXForestPredictor` basado en **ONNX Runtime** para inferencia acelerada sobre CPU.
- **Temperature Scaling (`sigmoid_with_temperature`, `fit_temperature`, `calibration_errors`)**: Escalado de logits antes de la sigmoide ($p = \sigma(z/T)$); `load_temperature` lee `T` de `calibration.json` (valor vigente: `0.3277`).
- **Umbralización adaptativa de Otsu (`compute_otsu_threshold`)**: Cálculo matemático del umbral óptimo bimodal entre fondo forestal estable y anomalía espectral.
- **Normalización de scores (`calibrate_probabilities`)**: reescala percentiles empíricos a [0, 1].
- **Filtro biofísico multitemporal (`apply_biophysical_filter`)**: Descarte estricto de falsos positivos requiriendo evidencia física de pérdida de dosel ($\Delta\text{NDVI} \ge 0.10$) o pérdida de humedad/quema ($\Delta\text{NBR} \ge 0.15$).
- **Inferencia por ventanas deslizantes con fusión Gaussiana suave**: Evita discontinuidades y artefactos de costura en rásteres de gran escala.
- **Extracción vectorial de alertas (`extract_alerts`)**: Conversión de máscaras continuas a polígonos GeoJSON con cálculo de áreas (ha) y asignación de nivel de severidad (`Alta`, `Media`, `Baja`).

---

## 🛠️ Tecnologías

- **Lenguaje:** Python 3.14.8 (`.venv`; `venv/` está vacío)
- **Geoespacial / GIS:** Rasterio 1.5.2, GeoPandas 1.2.0, Shapely 2.2.0, PyProj 3.8.0
- **Deep Learning / Inferencia:** PyTorch 2.14.1+cpu (sin CUDA en este equipo), ONNX 1.23.2, ONNX Runtime 1.31.0; Albumentations es opcional (fallback NumPy en `src/dataset.py`)
- **Interfaz y cartografía:** Streamlit 1.66.0, Folium 0.20.0, Streamlit-Folium 0.27.4
- **Video:** HyperFrames (animaciones en HTML/JS, render a MP4)
- **Pruebas:** Pytest 9.1.1 (**188 pruebas**: 178 pasan sin datos locales; 10 se omiten si faltan los artefactos no versionados)
- **Contenedor de temas:** `.streamlit/config.toml` (tema «Niebla de dosel»)

---

## 📁 Estructura del repositorio

```text
GeoForest-AI-Huanuco/
├── AGENTS.md                  # Protocolo de coordinación multiagente
├── MEMORY.md                  # Memoria técnica y estado del proyecto
├── app/                       # Panel Streamlit interactivo y servicios
│   ├── main.py                # Punto de entrada de la aplicación Streamlit
│   ├── config.py              # Parámetros territoriales (Carpish / Huallaga)
│   ├── components/            # Módulos de interfaz de usuario desacoplados
│   │   ├── tab_pitch.py       # Presentación ejecutiva y línea de tiempo multitemporal
│   │   ├── tab_sig.py         # Consola SIG integral, filtros y auditoría
│   │   ├── tab_inferencia.py  # Consola de Misión Satelital y laboratorio U-Net
│   │   ├── tab_benchmark.py   # Benchmark y métricas comparativas
│   │   └── map_view.py        # Generación de mapas Folium multicapa
│   ├── services/              # Lógica de datos y telemetría
│   │   ├── copernicus.py      # Catálogo de escenas y consulta STAC Sentinel-2 L2A
│   │   ├── alerts_data.py     # Carga de alertas ATD PNCB Geobosques MINAM
│   │   ├── nasa_firms.py      # Telemetría de focos de calor NASA VIIRS
│   │   ├── expediente.py      # Generador de expedientes técnicos con hash SHA-256
│   │   └── onnx_demo.py       # Transformaciones ráster y matrices visuales
│   ├── styles/                # Tema táctico, animaciones y CSS
│   └── assets/                # Portada animada (portada-bosque.mp4)
├── data/                      # Repositorio de datos espaciales (no versionados)
│   ├── raw/                   # GeoTIFFs Sentinel-2 L2A y ESA OSCD
│   └── processed/             # Vectores oficiales ATD Geobosques (Huánuco)
├── docs/                      # Guía operativa y runbook de exposición
│   └── demo_runbook.md
├── models/                    # Pesos, grafos ONNX y calibración (no versionados)
│   ├── unet_best.pth          # Checkpoint PyTorch (ausente en este checkout)
│   ├── unet_geoforest.onnx    # ONNX opset 17, FP32 (ausente; se genera con export_onnx.py)
│   ├── unet_sentinel2_*.onnx  # Grafos servibles por la app (hashes ≠ manifest actual)
│   ├── calibration.json       # Temperature Scaling y métricas ECE/MCE
│   └── manifest.json          # Hashes de integridad (desactualizados: ver MEMORY.md §3.1)
├── src/                       # Núcleo geoespacial y de aprendizaje profundo
│   ├── __init__.py            # API unificada
│   ├── dataset.py             # Cargador OSCD multitemporal y aumentos de datos
│   ├── train.py               # Pipeline de entrenamiento con FocalTverskyLoss
│   ├── preprocess.py          # Preprocesamiento ráster GIS y rasterización
│   ├── unet.py                # Arquitectura convolucional U-Net
│   ├── infer.py               # Motor ONNX, Otsu, Temperature Scaling y filtro biofísico
│   └── sim/                   # Series temporales multitemporales y métricas
├── tests/                     # Suite de pruebas automatizadas (188 pruebas)
├── scripts/                   # Pipeline: run_phase1_and_2.py, export_onnx.py, verificar_pipeline.py
├── reports/                   # Reporte de benchmark operacional (JSON)
├── requirements.txt           # Dependencias de producción
└── README.md                  # Documentación general
```

---

## ▶️ Instalación y ejecución

1. **Activar el entorno virtual con dependencias:**

   ```bash
   source .venv/bin/activate
   ```

   > `venv/` existe en este checkout pero solo contiene `pip`; los comandos
   > documentados fallan ahí con `No module named pytest`. Usar `.venv`.

2. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la suite de pruebas (188 tests):**

   ```bash
   pytest tests/ -q
   ```

   En un checkout sin datos ni modelos locales se esperan **178 pruebas OK y 10 omitidas
   (`skipped`)** con motivo explícito por falta de `data/raw/sentinel2`,
   `data/processed/benchmark_scenes.json`, `models/unet_best.pth` o
   `models/unet_geoforest.onnx` (ver paso 5). Las pruebas se ejecutan íntegras
   en cuanto existan esos artefactos; no se relajó ningún umbral.

4. **Lanzar el panel interactivo principal:**

   ```bash
   streamlit run app/main.py --server.headless true --server.port 8501
   ```

5. **Verificar artefactos del pipeline antes de una demo:**

   ```bash
   python scripts/verificar_pipeline.py
   ```

   Comprueba entorno, modelos y datos, y reporta qué regenerar:

   ```bash
   python scripts/run_phase1_and_2.py     # datos → índices → entrenamiento → models/unet_best.pth
   python scripts/export_onnx.py          # checkpoint → models/unet_geoforest.onnx + manifest
   ```

   Los dos scripts se ejecutan desde la raíz del repositorio (`python scripts/...`);
   antes fallaban con `ModuleNotFoundError: No module named 'src'`.

---

## ⚠️ Consideraciones operativas y de alcance

- **Datos oficiales y reales de entrada:** El panel consume datos reales cuando están presentes: alertas tempranas de deforestación del PNCB Geobosques (MINAM), pasadas orbitales de Sentinel-2 L2A (Copernicus CDSE) sobre Huánuco (Tile 18LUQ) y focos térmicos en vivo de NASA FIRMS (sensor VIIRS).
- **Advertencia sobre la capa Sentinel-2:** `data/raw/sentinel2/` está vacío en este checkout. `app/services/copernicus.py::cargar_escena_sentinel2_real` genera entonces una escena sintética de relleno (`np.random.default_rng(42)`) y la devuelve marcada como `es_satelital_real: False` / `es_simulado: True`; `tests/test_infer.py::test_catalogo_escenas_satelitales_reales` verifica esa marca mediante `escena_real_disponible`. Antes de una demo hay que comprobar que la UI muestre la capa como sintética y no como satelital real.
- **Artefactos del modelo:** `models/unet_best.pth` y `models/unet_geoforest.onnx` no existen en este checkout; los dos `unet_sentinel2_*.onnx` presentes no coinciden con los SHA-256 de `models/manifest.json`. Un smoke test con entrada aleatoria (`scripts/verificar_pipeline.py`) muestra que ambos grafos producen una salida **casi constante** (rango 0.015 y 0.019), es decir, no discriminan cambio: no deben presentarse sus máscaras como detecciones. Regenerarlos con `scripts/run_phase1_and_2.py` y `scripts/export_onnx.py`.
- **Controles de inferencia no operativos:** en `app/components/tab_inferencia.py`, «Adaptativo (Otsu)» usa en realidad `0.50`, y los selectores «Filtro biofísico» y «Reescalar score continuo» solo cambian etiquetas: `compute_otsu_threshold`, `apply_biophysical_filter` y `calibrate_probabilities` se importan pero no se invocan. Tampoco existe una capa Copernicus CDSE en `app/components/map_view.py`: el checkbox del sidebar (`mostrar_sentinel_copernicus`) no tiene efecto. No afirmar en la demo que están activos hasta corregirlo.
- **Atributos derivados de las alertas (no oficiales):** `app/services/alerts_data.py` asigna `area_ha = puntos × 0.12`, `confianza_pct` desde un logit constante `3.2` (≈99.8 % para toda alerta), `ndvi_cambio = -0.30`, deltas fijos y `estado = «Alerta Temprana Validada (MINAM)»`. `app/services/expediente.py` los reproduce y además afirma «registros oficiales validados por PNCB Geobosques». Nada de eso proviene del PNCB: no presentarlo como validación oficial.
- **Motor de Inferencia U-Net:** El modelo ONNX opera sobre tensores de reflectancia real (B04, B08, B11) a 10 m/píxel. `src/infer.py` implementa Temperature Scaling, umbralización adaptativa de Otsu y filtrado biofísico ($\Delta\text{NBR} \ge 0.15$ / $\Delta\text{NDVI} \ge 0.10$), pero la consola Streamlit todavía no los invoca (ver el punto anterior): no describirlos como activos en la demo.
- **Expediente técnico digital:** La exportación incluye un hash SHA-256 que permite verificar la integridad de los parámetros y coordenadas extraídas para fines de auditoría técnica.
- **Datos no versionados:** Los archivos ráster pesados en `data/raw/` y `data/processed/` se gestionan fuera del control de versiones mediante `.gitignore`.
