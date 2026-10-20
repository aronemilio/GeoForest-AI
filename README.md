# 🌲 GeoForest-AI Huánuco

Sistema experimental de alerta temprana para identificar posibles cambios en la cobertura forestal y quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga**, departamento de Huánuco, Perú.

> **Nota sobre los datos del panel.** El panel visualiza **alertas demostrativas identificadas como simuladas (`SIMULADO`)** para validar la interfaz, los indicadores y el visor cartográfico antes del despliegue en producción con imágenes satelitales en vivo. No se descargan ni se procesan imágenes satelitales reales todavía (la carpeta `data/` está vacía / en andamiaje). Los módulos de backend GIS, arquitectura U-Net e inferencia ONNX están **implementados y cubiertos por pruebas automatizadas**; las cifras de precisión/benchmark que se muestran en el panel son **de referencia simuladas, no resultados de una evaluación medida**.
>
> **Estado del modelo ONNX.** Los archivos `models/unet_sentinel2_*.onnx` (≈ 31,1 MB, FP32) son **modelos de referencia sin entrenar** (salida constante). El motor de inferencia (ONNX Runtime, ventanas deslizantes) está integrado y verificado, pero el modelo servido **no detecta** pérdida ni quemas; las métricas de IoU/F1/latencia del panel son una plantilla de maquetación. No se cuantiza a Int8 (no hay PTQ).

---

## 🚀 Estado de los módulos desarrollados

### 1. Panel de monitoreo e interfaz (`app/main.py`)

Dashboard en Streamlit con tema «Niebla de dosel» (verde forestal + mist). Pestañas:

- **🗺️ Panel de Monitoreo GIS**: KPIs (alertas simuladas activas, superficie estimada simulada, severidad alta, confianza promedio simulada), filtros dinámicos por zona, tipo de evento y severidad, mapa folium (capas base satélite/OSM/topográfico), polígonos de alerta, exportación CSV, GeoJSON y expediente demostrativo.
- **⏱️ Modo Pitch (Simulador Temporal)**: simulación temporal determinista (PRNG Mulberry32, bit-a-bit coherente con el visor React).
- **🧠 Benchmark U-Net ONNX (métricas simuladas)**: plantilla del reporte de IoU, F1, latencia y matriz de confusión; marcada como no medida. Peso real del `.onnx` servido (≈ 31,1 MB, FP32, sin cuantizar).
- **🌐 Visor Táctico Web (React :5174)**: integra (si responde) un subproyecto externo en `/home/emi/Proyectos/GeoForest-AI-Web`; degrada con aviso claro si el servidor no está en marcha.

Animaciones: pantalla de bienvenida (splash, CSS puro, ~3 s) y portada animada (`app/assets/portada-bosque.mp4`, bucle sin costura, 8 s, generada con HyperFrames).

### 2. Preprocesamiento geoespacial GIS (`src/preprocess.py`)

- Validación rigurosa de GeoTIFFs (CRS, resolución espacial, consistencia de cuadrícula `GridSpec`).
- Alineación y remuestreo explícito de bandas Sentinel-2 (B4 rojo, B8 NIR, B11 SWIR), distinguiendo datos continuos (bilineal) de categóricos (vecino más cercano).
- Cálculo vectorizado de índices espectrales: **NDVI** (vegetación) y **NBR** (quemas / áreas quemadas).
- Rasterización de vectores de referencia bajo reglas de etiquetado explícitas (`LabelRule`).
- Metadatos de procedencia reproducibles en formato JSON.

### 3. Simulación temporal y métricas (`src/sim/`)

- `src/sim/simulador.py`: réplica determinista del PRNG Mulberry32 y datos de simulación temporal.
- `src/sim/metricas.py`: plantilla de métricas y benchmark (IoU, F1, latencia, confusión por clase). Valores de referencia, no medidos.
- Hub de paquete `src/__init__.py` reexporta `preprocess`, `unet`, `infer` y `sim`.

### 4. Arquitectura U-Net (`src/unet.py`)

- Arquitectura convolucional U-Net en PyTorch para datos multiespectrales (5 canales: B4, B8, B11, NDVI, NBR).
- Skip connections, normalización por lotes (`BatchNorm2d`) y proyección a máscaras de probabilidad.
- Exportador a **ONNX** con dimensiones dinámicas en batch, alto y ancho (`export_unet_to_onnx`).
- Los archivos `.onnx` entregados son **de referencia (sin entrenar)**, FP32.

### 5. Motor de inferencia ONNX (`src/infer.py`)

- Predictor `ONNXForestPredictor` basado en **ONNX Runtime**.
- **Inferencia por ventanas deslizantes** para rásteres grandes sin saturar memoria.
- **Fusión ponderada suave** (Gaussiana, lineal o constante) para evitar artefactos de costura entre parches.
- Post-procesamiento geoespacial: binarización por umbral, cálculo de superficie afectada en ha (10 m/píxel → 0,01 ha) y extracción de alertas vectorizadas con severidad (`Alta`, `Media`, `Baja`).

---

## 🛠️ Tecnologías

- **Lenguaje:** Python 3.14.8
- **Geoespacial / GIS:** Rasterio 1.5.2, GeoPandas 1.2.0, Shapely, PyProj
- **Deep Learning / Inferencia:** PyTorch 2.14.1+cpu, ONNX 1.23.2, ONNX Runtime 1.31.0
- **Interfaz y cartografía:** Streamlit 1.66.0, Folium 0.20.0, Streamlit-Folium 0.27.4
- **Video:** HyperFrames (animaciones en HTML/JS, render a MP4)
- **Pruebas:** Pytest (77 pruebas automatizadas: 46 preprocesamiento, 11 U-Net, 16 inferencia, 4 simulación)
- **Contenedor de temas:** `.streamlit/config.toml` (tema «Niebla de dosel»)

---

## 📁 Estructura del repositorio

```text
GeoForest-AI-Huanuco/
├── AGENTS.md                  # Protocolo de coordinación multiagente
├── app/                       # Panel Streamlit y assets animados
│   ├── main.py                # Panel principal en Streamlit (4 pestañas)
│   └── assets/
│       └── portada-bosque.mp4 # Portada animada del encabezado (bucle, 8 s)
├── data/                      # Andamiaje: sin datos reales todavía
│   ├── raw/                   # geobosques/, sentinel2/
│   ├── masks/
│   └── processed/
├── docs/                      # Guía de operación de la demo
│   └── demo_runbook.md
├── models/
│   ├── .gitkeep
│   ├── unet_sentinel2_huanuco.onnx        # Modelo de referencia ONNX (FP32, sin entrenar)
│   └── unet_sentinel2_multitemporal.onnx  # Ídem (multitemporal)
├── src/
│   ├── __init__.py            # API unificada
│   ├── preprocess.py          # Preprocesamiento GIS, NDVI/NBR
│   ├── unet.py                # U-Net y exportador ONNX
│   ├── infer.py               # Inferencia ONNX por ventanas deslizantes
│   └── sim/
│       ├── __init__.py
│       ├── metricas.py        # Referencias simuladas de benchmark
│       └── simulador.py       # Simulación temporal determinista
├── tests/
│   ├── test_preprocess.py     # 46 pruebas GIS
│   ├── test_unet.py           # 11 pruebas de U-Net y exportación
│   ├── test_infer.py          # 16 pruebas de inferencia ONNX
│   └── test_sim.py            # 4 pruebas de simulación
├── videos/
│   └── geoforest-portada/     # Composición HyperFrames (fuente y parámetros)
├── .streamlit/
│   └── config.toml            # Tema «Niebla de dosel»
├── requirements.txt           # Dependencias (Python 3.14)
└── README.md                  # Documentación general
```

> La carpeta `docs/` ahora contiene `demo_runbook.md` (guía operativa de la exposición); el resto de la documentación técnica está en `AGENTS.md` y en los docstrings del código.

---

## ▶️ Instalación y ejecución

1. **Activar el entorno virtual:**

   ```bash
   source .venv/bin/activate
   ```

2. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar la suite de pruebas (77 tests):**

   ```bash
   pytest tests/ -q
   ```

4. **Lanzar el panel interactivo (usar Python como módulo, para que `src` quede en `sys.path`):**

   ```bash
   python -m streamlit run app/main.py --server.headless true --server.port 8501 --server.fileWatcherType none
   ```

5. **(Opcional) Lanzar el visor web externo (React + Vite):** la pestaña «🌐 Visor Táctico Web» consume `http://localhost:5174/`. Si está disponible, se embebe; si no, se muestra un aviso.

   ```bash
   cd /home/emi/Proyectos/GeoForest-AI-Web
   npm run dev -- --port 5174 --host
   ```

---

## ⚠️ Limitaciones importantes (para exposición / demo)

- **Todo lo que se visualiza en el panel es simulado.** Alertas, áreas en ha, confianza IA, métricas de benchmark (IoU, F1, latencia, matriz de confusión) y el expediente FEMA/SERFOR son datos de maquetación marcados como `SIMULADO`. No representan una detección real.
- **El modelo ONNX está sin entrenar.** Su salida es prácticamente constante (~0,465) y el botón «🔧 Verificar pipeline ONNX» mostrará 0,00 ha para que quede evidente que no detecta. Sirve para comprobar que el motor funciona y cuánto tarda, **no** para decidir.
- **No hay cuantización Int8.** El `.onnx` servido es FP32 (~31,1 MB). No aplicar claims de PTQ ni de throughput ficticio.
- **Sin datos satelitales reales.** `data/` está vacía; no se descargaron escenas Sentinel-2 ni vectores de Geobosques (requiere autorización). Los recortes del pipeline son sintéticos deterministas.
- **El visor web es un proyecto aparte** (`GeoForest-AI-Web`) y depende de que ese servidor esté en marcha; sin él, su pestaña degrada a un aviso claro.
- **Nada de lo aquí publicado es evidencia jurídica.** El hash SHA-256 calculado es una huella de integridad reproducible, no una cadena de custodia acreditada.
