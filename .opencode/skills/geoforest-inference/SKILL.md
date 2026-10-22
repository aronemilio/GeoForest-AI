---
name: GeoForest Inference
description: Guía para src/infer.py y tests/test_infer.py de GeoForest-AI (ONNX Runtime, ventanas deslizantes, umbral, extracción de alertas y semántica correcta de la confianza).
---

# GeoForest Inference

## Cuándo usar

- Vas a modificar `src/infer.py` o `tests/test_infer.py`.
- Necesitas ejecutar o revisar `ONNXForestPredictor`, `predict_sliding_window`, `binarize`, `calculate_affected_area` o `extract_alerts`.
- Debes decidir cómo mostrar la “confianza” de una alerta.

## Ownership

- Edita solo `src/infer.py` y `tests/test_infer.py`.
- No cambies el contrato de 6 canales ni el significado de `confidence_score` sin actualizar pruebas y UI.
- Si hay cambios ajenos, detente y reporta.

## Reglas duras

- `confidence_score` es la **media de la probabilidad sigmoidea** dentro del polígono; **no** es una probabilidad calibrada.
- `confidence_level` usa umbrales heurísticos: ALTA ≥ 0,80; MEDIA ≥ 0,60; BAJA < 0,60.
- Con un modelo sin entrenar, **no** llames a eso “confianza” ni “detección”: usa “probabilidad media (sin calibrar)”.
- **Nunca** rellenes área o confianza con valores ficticios cuando no hay píxeles sobre el umbral. Devuelve `0.0` o “sin detecciones”.
- Mantén el umbral explícito y documentado; no lo ocultes en la UI.
- Valida forma, canales (6), tamaño máximo y valores finitos antes de inferir.

## Flujo de trabajo

1. Verifica el modelo y el contrato de entrada (ver skill `geoforest-model`).
2. Haz el cambio con pruebas sintéticas; no dependas de red ni de datos reales.
3. Ejecuta:
   ```bash
   source venv/bin/activate
   python -m pytest tests/test_infer.py -q
   ```
4. Si tocas la UI, aplica también el skill `geoforest-ui`.

## Referencias

- `references/confidence-policy.md` — fórmula, umbrales, calibración ausente y textos permitidos.
