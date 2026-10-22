# Política de confianza — GeoForest-AI

## Qué calcula el código

En `src/infer.py`, `extract_alerts()` hace:

```python
conf_score = float(np.mean(sub_prob[valid_in_shape]))
```

Es decir, la **media de la probabilidad sigmoidea** de los píxeles del polígono. Luego asigna:

| Nivel | Condición |
| --- | --- |
| `ALTA` | `conf_score >= 0.80` |
| `MEDIA` | `conf_score >= 0.60` |
| `BAJA` | `conf_score < 0.60` |

Estos umbrales son heurísticos; no son intervalos de confianza ni probabilidades calibradas.

## Por qué no es “confianza” hoy

- El modelo servido (`models/unet_geoforest.onnx`) es de referencia y **no está entrenado**.
- No hay calibración (temperature scaling, isotonic regression) ni ECE medido.
- La probabilidad sigmoidea de una red sin entrenar no tiene interpretación predictiva.
- Evidencia empírica: entrada aleatoria → media ≈ 0,435, desv. ≈ 0,098; ceros → ≈ 0,320; unos → ≈ 0,574. No hay señal aprendida.

## Textos permitidos en UI

| Situación | Texto correcto |
| --- | --- |
| Salida del modelo sin entrenar | “Probabilidad media (sin calibrar)” |
| Nivel cualitativo | “Nivel de confianza (heurístico)” |
| Alertas simuladas | “Confianza simulada” |
| Sin píxeles sobre umbral | “0,00 ha” o “Sin detecciones” |

## Textos prohibidos

- “Confianza 94,2 %” para una salida no calibrada.
- “Certeza IA”.
- “Detección real” o “Detección validada”.
- Cualquier valor de relleno (por ejemplo, 1,24 ha) cuando el modelo no produce detecciones.

## Cómo obtener una confianza real (trabajo futuro)

1. Etiquetas validadas de Geobosques con `LabelRule` explícita.
2. Entrenamiento y split espacial/temporal.
3. Métricas en conjunto retenido: IoU, F1, precisión, recall, PR-AUC.
4. Calibración: reliability diagram, ECE, temperature scaling.
5. Reportar intervalos por bootstrap y validación externa en Carpish/Huallaga.
