# Inventario verificado — GeoForest-AI

Verificado el 2026-10-20 en el checkout local.

## Entorno

| Dato | Valor real |
| --- | --- |
| Intérprete | Python 3.10.20 |
| Entorno virtual | `venv/` |
| Pytest | 9.1.1 |
| Streamlit | 1.65.0 |
| Streamlit-Folium | 0.26.2 |
| PyTorch | 2.5.1+cu121 |
| ONNX | 1.23.2 |
| ONNX Runtime | 1.23.2 |
| Rasterio | 1.4.4 |
| GeoPandas | 1.1.4 |
| Shapely | 2.1.2 |
| PyProj | 3.7.1 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |

## Pruebas

```text
77 tests collected
46 tests/test_preprocess.py
16 tests/test_infer.py
11 tests/test_unet.py
 4 tests/test_sim.py
```

## Modelo

| Dato | Valor real |
| --- | --- |
| `models/unet_geoforest.onnx` | 7.713.376 bytes (≈ 7,7 MB / 7,4 MiB) |
| `models/unet_best.pth` | 7.753.538 bytes |
| Entrada ONNX | `['batch', 6, 'height', 'width']`, float32 |
| Salida ONNX | `[batch, 1, height, width]` |
| Estado | Modelo de referencia **sin entrenar**; no es una detección |
| Metadatos del checkpoint | Ninguno (sin época, pérdida ni métricas) |
| `src/train.py` | **No existe** en este checkout |

## Datos

| Dato | Valor real |
| --- | --- |
| `data/` | ≈ 842 MB |
| `data/raw/sentinel2/` | 2 escenas × 3 bandas (B04, B08, B11) |
| `data/raw/geobosques/` | Shapefile PNCB y ZIP de alertas |
| `data/processed/vectores/` | `atd_huanuco.geojson` |
| Versionado | Excluido por `.gitignore` (`data/raw/*`, `data/processed/*`) |

## Inconsistencias conocidas

- `requirements.txt` declara mínimos superiores a lo instalado (`streamlit>=1.66.0`, `pandas>=3.0.0`, `numpy>=2.5.0`, `torch>=2.14.0`, `onnxruntime>=1.31.0`). No es una fuente de verdad del entorno actual; usar `requirements.lock` o `pip freeze`.
- `docs/demo_runbook.md` decía “35+ pruebas GIS”; el número real es 46.
- El README antiguo citaba 31,1 MB y 5 canales; el artefacto real pesa 7,7 MB y espera 6 canales.
- `app/services/copernicus.py` consulta Earth Search (AWS), no la API CDSE de Copernicus; los nombres “Copernicus” se refieren a la constelación, no al endpoint.

## Comandos de verificación

```bash
source venv/bin/activate
python --version
python -m pytest tests/ --collect-only -q | tail -1
python - <<'PY'
import onnxruntime as ort
s = ort.InferenceSession("models/unet_geoforest.onnx", providers=["CPUExecutionProvider"])
print(s.get_inputs()[0].name, s.get_inputs()[0].shape)
PY
stat -c '%s %n' models/unet_geoforest.onnx models/unet_best.pth
du -sh data
```
