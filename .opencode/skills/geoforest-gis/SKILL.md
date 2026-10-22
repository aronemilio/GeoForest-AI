---
name: GeoForest GIS
description: Guía para cambios en src/preprocess.py y tests/test_preprocess.py de GeoForest-AI (validación de GeoTIFF, bandas Sentinel-2 B4/B8/B11, alineación de cuadrícula, remuestreo explícito y LabelRule).
---

# GeoForest GIS

## Cuándo usar

- Vas a modificar `src/preprocess.py` o `tests/test_preprocess.py`.
- Necesitas validar GeoTIFF, alinear bandas B4/B8/B11, calcular NDVI/NBR o rasterizar vectores de Geobosques.
- Debes revisar una `LabelRule` o una máscara categórica.

## Ownership

- Edita solo `src/preprocess.py` y `tests/test_preprocess.py`.
- Antes de editar: `git status --short` y `git log --oneline -5`.
- Si el archivo tiene cambios que no son tuyos, detente y repórtalo.

## Reglas duras

- **No descargar** imágenes satelitales ni vectores sin autorización explícita.
- **No inventar etiquetas**: toda máscara requiere una `LabelRule` explícita (`field`, `mapping`, `background_value`).
- Un cambio de cobertura forestal **no** implica tala ilegal; la semántica de clases es externa al módulo.
- Reproyección y remuestreo siempre explícitos: continuo → `Resampling.bilinear`; categórico → `Resampling.nearest`.
- No silenciar errores de CRS, cuadrícula o resolución: usa las excepciones de `src/preprocess.py`.

## Flujo de trabajo

1. Lee el archivo completo y localiza la función/clase afectada.
2. Añade o actualiza pruebas sintéticas con `tmp_path` (sin red ni datos reales).
3. Ejecuta:
   ```bash
   source venv/bin/activate
   python -m pytest tests/test_preprocess.py -q
   ```
4. Reporta archivos, comandos y resultados.

## Referencias

- `references/domain-rules.md` — bandas, remuestreo y contrato de `LabelRule`.
