# 🌲 GeoForest-AI Huánuco

Sistema experimental de alerta temprana para identificar posibles cambios en la cobertura forestal y quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga**, departamento de Huánuco, Perú.

> **Nota sobre los datos actuales:** En esta primera etapa, el panel visualiza **alertas demostrativas explícitamente identificadas como simuladas (`SIMULADO`)** con el objetivo de validar la interfaz, los indicadores y el visor cartográfico antes de conectar los datos satelitales reales y la inferencia del modelo.

---

## 🚀 Características actuales (Grupo 1)

- **Panel interactivo en español (`Streamlit`):**
  - Indicadores clave (KPIs): alertas simuladas activas, superficie afectada estimada (ha), alertas de severidad alta y confianza promedio simulada.
  - Filtros dinámicos por zona de estudio (**Bosque de Carpish** y **Valle del Huallaga**), tipo de evento (*Deforestación*, *Quema agrícola*, *Degradación de bosque nuboso*) y nivel de severidad (*Alta*, *Media*, *Baja*).
- **Mapa geoespacial interactivo (`Folium` / `Leaflet`):**
  - Capas base intercambiables: Satélite (*Esri World Imagery*), *OpenStreetMap* y Topográfico (*OpenTopoMap*).
  - Polígonos delimitadores referenciales de las zonas de estudio en Huánuco.
  - Polígonos y centroides de áreas de alerta simuladas coloreados por nivel de severidad y con fichas informativas emergentes.
- **Resumen analítico y exportación:**
  - Gráficos y tablas de distribución por tipo de evento, zona y severidad.
  - Descarga directa en formato CSV de los registros demostrativos filtrados.

---

## 🛠️ Tecnologías

- **Lenguaje:** Python 3
- **Datos satelitales y referencia (fases siguientes):** Sentinel-2 y Geobosques (MINAM)
- **Modelado e inferencia (fases siguientes):** PyTorch (U-Net) y ONNX Runtime
- **Interfaz y visualización:** Streamlit, Folium (`streamlit-folium`) y Leaflet

---

## 📁 Estructura del repositorio

```text
GeoForest-AI-Huanuco/
├── app/
│   └── main.py          # Panel principal en Streamlit con mapa e indicadores
├── data/
│   ├── raw/
│   │   ├── geobosques/  # Datos de referencia de pérdida de bosque
│   │   └── sentinel2/   # Escenas multiespectrales Sentinel-2
│   ├── masks/           # Máscaras binarias de entrenamiento
│   └── processed/       # Parches procesados e índices espectrales
├── docs/                # Documentación técnica
├── models/              # Pesos entrenados y modelos exportados (.onnx)
├── src/
│   ├── preprocess.py    # Preprocesamiento de bandas e índices (NDVI, NBR)
│   ├── unet.py          # Arquitectura de segmentación U-Net
│   └── infer.py         # Motor de inferencia con ONNX Runtime
├── tests/               # Pruebas automatizadas
├── requirements.txt     # Dependencias del entorno virtual
└── README.md            # Documentación general
```

---

## ▶️ Instalación y ejecución

1. **Activar el entorno virtual:**

   ```bash
   source .venv/bin/activate
   ```

2. **Instalar dependencias (si aún no están instaladas):**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el panel Streamlit:**

   ```bash
   streamlit run app/main.py
   ```

---

## 🗺️ Orden de trabajo (Hoja de ruta)

1. **Fase 1 (Completada):** Estructura base del repositorio, panel visual en español, mapa interactivo de la zona de estudio y alertas demostrativas simuladas.
2. **Fase 2:** Mapa interactivo avanzado y datos reales de referencia.
3. **Fase 3:** Preprocesamiento de imágenes Sentinel-2 y generación de máscaras de Geobosques.
4. **Fase 4:** Entrenamiento del modelo U-Net, exportación e inferencia con ONNX Runtime y evaluación de métricas.
