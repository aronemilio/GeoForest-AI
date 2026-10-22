# Reglas de dominio GIS — GeoForest-AI

## Bandas y productos

- Bandas espectrales de entrada: **B4 (rojo)**, **B8 (NIR)**, **B11 (SWIR1)**.
- Índices: **NDVI** (vegetación) y **NBR** (quemas / áreas quemadas).
- El modelo servido usa **6 canales**: T0 (B4, B8, B11) + T1 (B4, B8, B11).
- No confundir el valor por defecto de `src/unet.py` (5 canales con NDVI/NBR) con el contrato del ONNX servido (6 canales). Verifica siempre la entrada real del modelo.

## Remuestreo explícito

`src/preprocess.py` define `DataKind` y `ResamplingPolicy`:

| Naturaleza | Método | Uso |
| --- | --- | --- |
| `DataKind.CONTINUO` | `Resampling.bilinear` | reflectancia, índices espectrales |
| `DataKind.CATEGORICO` | `Resampling.nearest` | máscaras, etiquetas, clases |

No elijas el método de forma implícita. Si hay reproyección, documéntala en los metadatos de procedencia.

## Contrato de `LabelRule`

```python
LabelRule(
    field="clase",              # columna del vector
    mapping={"bosque": 1, "pérdida": 2},
    background_value=0,
    nodata=None,
    allow_unmapped=False,       # por defecto, un valor no mapeado es error
    unmapped_value=-1,
)
```

- El módulo **no** asume que un código signifique deforestación.
- Si `allow_unmapped=False`, cualquier valor ausente del `mapping` lanza `LabelRuleError`.
- Toda máscara debe conservar la `LabelRule` aplicada y la `GridSpec` de alineación.

## Prohibiciones

- No descargar datos sin autorización.
- No inventar etiquetas ni clases.
- No usar remuestreo bilineal sobre máscaras categóricas.
- No presentar una máscara simulada como una detección real.
