# GeoForest-AI Huánuco — Instrucciones del proyecto

Prototipo de alerta temprana para identificar posibles cambios de cobertura forestal
y quemas agrícolas en el **Bosque Montano de Carpish** y el **Valle del Huallaga**
(Huánuco, Perú). Stack: Python, Rasterio, GeoPandas, Sentinel-2, Geobosques (MINAM),
U-Net, ONNX Runtime y Streamlit.

> ⚠️ **Trabajo multiagente concurrente.** Varios agentes (sesiones primarias y
> subagentes, en primer o segundo plano) pueden estar editando este mismo
> repositorio al mismo tiempo. Asume siempre concurrencia y sigue el protocolo
> para no interferir con el trabajo de los demás.

## Protocolo de coordinación multiagente

### Antes de tocar un archivo
- Ejecuta `git status --short` y `git log --oneline -5` para ver el estado real.
- Relee el archivo justo antes de editarlo: puede haber cambiado.
- Declara qué archivos vas a editar (**ownership**) antes de empezar.
- Si un archivo ya tiene cambios que no son tuyos, **detente y repórtalo**; no lo
  sobrescribas ni lo reviertas.

### Git (reglas estrictas)
- **Nunca** `git add .` ni `git add -A`: usa rutas explícitas
  (`git add ruta/archivo1 ruta/archivo2`).
- **Nunca** `git reset`, `git stash`, `git checkout`, `git restore`, `git clean`
  ni `git switch` sobre trabajo ajeno.
- **No** hagas `commit` ni `push` sin autorización explícita del usuario.
- **No** cambies de rama.
- Verifica `git diff --cached --name-only` justo antes de confirmar.

### Edición
- Toca **solo** los archivos de tu responsabilidad (ver mapa de propiedad).
- No borres ni sobrescribas trabajo existente sin justificación.
- No dejes marcadores de conflicto (`<<<<<<<`, `=======`, `>>>>>>>`).
- Usa directorios temporales únicos bajo `/tmp/opencode`.

## Mapa de propiedad de archivos

| Ruta | Responsable | Notas |
| --- | --- | --- |
| `app/main.py`, `app/components/**` | Interfaz (Frontend) | Panel Streamlit modular (modo Pitch y Consola SIG). Las vistas operativas usan datos reales PNCB Geobosques y Sentinel-2; el Modo Pitch es una serie demostrativa. |
| `app/services/**` | Servicios & Datos | Carga de ATD Geobosques, Copernicus CDSE, NASA FIRMS y expedientes SHA-256. |
| `src/preprocess.py` | Preprocesamiento GIS | Validación ráster, bandas S2 (B04/B08/B11), vectores de referencia y máscaras. |
| `tests/test_preprocess.py` | Preprocesamiento GIS | Pruebas con datos sintéticos y rásteres reales. |
| `src/unet.py` | Modelado | Arquitectura U-Net y exportador ONNX. |
| `src/infer.py` | Inferencia | Motor ONNX Runtime, umbralización Otsu, normalización de scores y filtro biofísico. |
| `requirements.txt`, `README.md` | Coordinación | Cambiar solo con autorización y anunciándolo para no pisar a otros. |
| `docs/` | Documentación | Documentación técnica y runbook de exposición. |

## Subagentes por rol (ownership impuesto por permisos)

| Subagente | Puede editar | Notas |
| --- | --- | --- |
| `gis-worker` | `src/preprocess.py`, `tests/test_preprocess.py` | Preprocesamiento GIS. |
| `ml-worker` | `src/unet.py`, `tests/test_unet.py`, `models/**` | Arquitectura U-Net y exports ONNX. |
| `inference-worker` | `src/infer.py`, `tests/test_infer.py` | Motor ONNX Runtime. |

Cada worker tiene `edit` **denegado** en el resto del repositorio, `git` destructivo
bloqueado y `git commit`/`push` en `ask`. Para cada cambio de código, lanza el worker
de su rol; así el ownership queda garantizado por permisos y no solo por instrucciones.

## Entorno y comandos

```bash
source venv/bin/activate                  # entorno virtual del proyecto
python -m pytest tests/ -v                # pruebas
streamlit run app/main.py                 # interfaz
```

## Reglas de dominio del proyecto
- No descargar imágenes satelitales ni datos de referencia sin autorización.
- No inventar etiquetas: toda máscara requiere una `LabelRule` **explícita**; un
  cambio de cobertura forestal **no** implica tala ilegal.
- Trabajar rigurosamente con datos y detecciones reales.
- No rellenar áreas, confianzas ni métricas con valores ficticios cuando el
  modelo no produce detecciones: mostrar `0` o «sin detecciones».
- La salida del modelo no es una confianza calibrada; etiquetarla
  como «probabilidad media (sin calibrar)».
- No llamar «calibrada» a una mezcla de percentiles/máximos, a `confianza_pct`
  ni a una fórmula basada en número de puntos. Solo Brier/ECE/reliability
  medidos sobre una referencia independiente pueden respaldar esa etiqueta.
- Distinguir **imagen real** de **detección**: una escena Copernicus real sigue
  siendo solo entrada si el modelo no está validado o no produce detecciones.
- Toda reproyección o remuestreo debe ser **explícita** y quedar documentada
  (continuo vs. categórico).

## Reporte final
- Indica archivos creados/modificados, comandos ejecutados, resultados de las
  pruebas y siguientes pasos. No hagas commit ni push sin autorización.
