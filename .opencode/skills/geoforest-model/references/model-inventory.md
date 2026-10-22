# Inventario del modelo — GeoForest-AI

Verificado el 2026-10-20 en el entorno local.

## Artefactos

| Archivo | Tamaño | Estado |
| --- | ---: | --- |
| `models/unet_geoforest.onnx` | 7.713.376 bytes (≈ 7,7 MB / 7,4 MiB) | FP32, modelo de referencia **sin entrenar** |
| `models/unet_best.pth` | 7.753.538 bytes | `state_dict` de PyTorch, sin metadatos de entrenamiento |
| `models/unet_sentinel2_huanuco.onnx` | symlink a `unet_geoforest.onnx` | alias de compatibilidad |
| `models/unet_sentinel2_multitemporal.onnx` | symlink a `unet_geoforest.onnx` | alias de compatibilidad |

## Contrato ONNX

- Entrada: `input_6ch`, forma `['batch', 6, 'height', 'width']`, tipo `tensor(float)`.
- Salida: `[batch, 1, height, width]`, probabilidades sigmoideas.
- Dimensiones dinámicas en batch, alto y ancho.

## Evidencia de estado

- El checkpoint no contiene `epoch`, `best_iou`, `loss` ni historial.
- No existe `src/train.py` en el repositorio a esta fecha.
- La salida del ONNX cambia con la entrada, pero no tiene señal aprendida: con entrada aleatoria la media es ≈ 0,435 y la desviación ≈ 0,098; con ceros ≈ 0,320 y con unos ≈ 0,574.
- `models/manifest.json` solo registra hashes SHA-256 de los dos artefactos principales.

## Implicación

El modelo sirve para verificar la cadena técnica (tensor → ONNX Runtime → post-procesamiento), **no** para detectar pérdida de cobertura ni quemas. Cualquier cifra de confianza, IoU o F1 debe presentarse como simulada o pendiente de medición.
