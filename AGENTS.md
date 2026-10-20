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
| `app/main.py` | Interfaz (Grupo 1) | Panel Streamlit. Datos actuales marcados como `SIMULADO`. No modificar sin autorización. |
| `src/preprocess.py` | Preprocesamiento GIS | Validación ráster, bandas S2 (B4/B8/B11), vectores de referencia y máscaras. |
| `tests/test_preprocess.py` | Preprocesamiento GIS | Pruebas con datos sintéticos. |
| `src/unet.py` | Modelado | Arquitectura U-Net. |
| `src/infer.py` | Inferencia | Motor ONNX Runtime. |
| `requirements.txt`, `README.md` | Coordinación | Cambiar solo con autorización y anunciándolo para no pisar a otros. |
| `docs/` | Documentación | Documentación técnica. |

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
source .venv/bin/activate                 # entorno virtual del proyecto
python -m pytest tests/ -v                # pruebas
streamlit run app/main.py                 # interfaz
```

## Reglas de dominio del proyecto
- No descargar imágenes satelitales ni datos de referencia sin autorización.
- No inventar etiquetas: toda máscara requiere una `LabelRule` **explícita**; un
  cambio de cobertura forestal **no** implica tala ilegal.
- No presentar datos simulados como detecciones reales.
- Toda reproyección o remuestreo debe ser **explícita** y quedar documentada
  (continuo vs. categórico).

## Reporte final
- Indica archivos creados/modificados, comandos ejecutados, resultados de las
  pruebas y siguientes pasos. No hagas commit ni push sin autorización.
