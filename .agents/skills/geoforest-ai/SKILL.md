---
name: geoforest-ai-huanuco
description: >-
  Contexto y restricciones de dominio del proyecto GeoForest-AI Huánuco.
  Actívalo cuando trabajes en este repositorio (GeoForest-AI-Huanuco): panel
  Streamlit, preprocesamiento GIS, U-Net, ONNX Runtime, simulación, temas y
  demo. Cubre el stack, las reglas de dominio (no presentar datos simulados
  como reales, modelo sin entrenar, no SCL), el mapa de propiedad multiagente,
  comandos de ejecución y la guía de exposición.
metadata:
  origin: GeoForest-AI
---

# GeoForest-AI Huánuco — contexto del proyecto

Prototipo de alerta temprana para detectar posibles cambios de cobertura forestal y
quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga**
(Huánuco, Perú). Stack: Python, Rasterio, GeoPandas, Sentinel-2, Geobosques (MINAM),
U-Net (PyTorch), ONNX Runtime, Streamlit.

## Estructura clave

- `app/main.py` — panel Streamlit (4 pestañas: GIS, Pitch, Benchmark, Visor React).
- `src/preprocess.py` — preprocesamiento GIS, NDVI/NBR, validación ráster.
- `src/unet.py` — arquitectura U-Net + exportador ONNX.
- `src/infer.py` — motor ONNX Runtime (sliding window, fusión suave, alertas).
- `src/sim/` — simulación temporal determinista (Mulberry32) compartida con el visor React.
- `tests/` — 77 pruebas (46 preprocess, 11 unet, 16 infer, 4 sim).
- `models/*.onnx` — modelos de **referencia, sin entrenar** (FP32, ~31,1 MB).
- `videos/geoforest-portada/` + `app/assets/portada-bosque.mp4` — portada animada (HyperFrames).
- `.streamlit/config.toml` — tema «Niebla de dosel».
- `AGENTS.md` — protocolo multiagente y mapa de propiedad (fuente de verdad).
- `docs/demo_runbook.md` — guía de operación de la demo.

## Comandos

```bash
source .venv/bin/activate
python -m pytest tests/ -q                          # 77 tests
python -m streamlit run app/main.py --server.headless true --server.port 8501 --server.fileWatcherType none
cd /home/emi/Proyectos/GeoForest-AI-Web && npm run dev -- --port 5174 --host   # visor React (opcional)
```

> Lanzar con `python -m streamlit`, no con el binario de consola (da `ModuleNotFoundError: No module named 'src'`).

## Reglas de dominio (no negociables)

1. **No presentar datos simulados como detecciones reales.** Las alertas del panel son demostrativas (`origen_dato = SIMULADO`). El modelo ONNX es *de referencia sin entrenar* (salida constante ~0,465). No afirmar IoU/F1/latencia como medidas ni cuantización Int8 (es FP32).
2. **No inventar etiquetas.** Toda máscara requiere una `LabelRule` explícita; un cambio de cobertura forestal NO implica tala ilegal.
3. **Toda reproyección/remuestreo es explícita** (continuo vs. categórico) y documentada.
4. **No descargar imágenes satelitales ni datos de referencia sin autorización.**
5. `src/preprocess.py` **no** gestiona máscaras de nubes/SCL ni corrección atmosférica (declarado en línea 57). No afirmes lo contrario.
6. `data/` está vacía: no hay escenas Sentinel-2 ni vectores de Geobosques todavía.

## Mapa de propiedad multiagente (ownership)

| Archivo | Responsable |
|---|---|
| `app/main.py` | Interfaz |
| `src/preprocess.py`, `tests/test_preprocess.py` | Preprocesamiento GIS |
| `src/unet.py`, `tests/test_unet.py`, `models/**` | Modelado (U-Net) |
| `src/infer.py`, `tests/test_infer.py` | Inferencia (ONNX) |
| `requirements.txt`, `README.md` | Coordinación (autorización) |
| `docs/` | Documentación |

Antes de editar: `git status --short` y `git log --oneline -5`; relee el archivo; declara ownership; no sobrescribir trabajo ajeno; nunca `git add .`/`-A`, `reset`, `stash`, `checkout`, `restore`, `clean`, `switch`; no commit/push sin autorización; no cambiar de rama.

## Guía de exposición (resumen)

- Presentar el panel, no cifras del modelo. No pulsar el botón de inferencia sin contexto (sale 0,00 ha).
- No afirmar: detección real, OSCD entrenado, calibración con Geobosques, SCL, 45/5 días, cadenas de custodia.
- Arranque: `python -m streamlit run app/main.py ...` (no el binario de consola). Visor React opcional en `:5174`.
- Guion completo: `docs/demo_runbook.md`.

## Archivos de referencia

- `AGENTS.md` — protocolo completo (fuente de verdad).
- `docs/demo_runbook.md` — guion de demo y diagnóstico.
- `README.md` — estado real del proyecto y limitaciones.
