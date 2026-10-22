---
name: GeoForest Model
description: Guía para src/unet.py, tests/test_unet.py y los artefactos de models/ de GeoForest-AI (arquitectura U-Net, exportación ONNX, manifiesto y estado real del modelo).
---

# GeoForest Model

## Cuándo usar

- Vas a modificar `src/unet.py` o `tests/test_unet.py`.
- Necesitas exportar un `.onnx`, actualizar `models/manifest.json` o revisar un checkpoint.
- Debes verificar los canales de entrada y las dimensiones dinámicas del modelo.

## Ownership

- Edita solo `src/unet.py`, `tests/test_unet.py` y `models/**`.
- No sobrescribas un `.onnx` o `.pth` existente sin respaldarlo y anunciarlo.
- Si hay cambios ajenos en esos archivos, detente y reporta.

## Reglas duras

- **No afirmes que un modelo está entrenado** si el checkpoint no contiene evidencia (época, pérdida, métricas, historial).
- El modelo servido es de **referencia y sin entrenar**; su salida no es una detección.
- No publiques métricas (IoU, F1, latencia) que no hayas medido.
- No apliques claims de cuantización Int8/PTQ sin un artefacto cuantizado y una medición reproducible.
- Mantén el contrato de entrada documentado: el ONNX servido espera **6 canales**; `src/unet.py` permite `in_channels` configurable.

## Flujo de trabajo

1. Verifica la arquitectura y el contrato real:
   ```bash
   source venv/bin/activate
   python -c "import onnxruntime as ort; s=ort.InferenceSession('models/unet_geoforest.onnx'); print(s.get_inputs()[0].shape)"
   ```
2. Cambia la arquitectura o el exportador con pruebas.
3. Si generas un artefacto, actualiza `models/manifest.json` con su SHA-256.
4. Ejecuta:
   ```bash
   python -m pytest tests/test_unet.py -q
   ```

## Referencias

- `references/model-inventory.md` — tamaño, hashes, entradas/salidas y estado del checkpoint.
