"""
Pruebas unitarias para el módulo de inferencia ONNX (``src/infer.py``).

Verifica:
- Inicialización y validación del predictor ONNXForestPredictor.
- Manejo de excepciones ante rutas inexistentes o formas de entrada inválidas.
- Inferencia directa por parches unitarios y lotes (batches).
- Inferencia por ventanas deslizantes con mezcla suave (Gaussian, Linear, Constant).
- Reconstrucción perfecta sin artefactos de borde y soporte de imágenes de cualquier tamaño.
- Post-procesamiento: binarización, cálculo de área afectada en hectáreas.
- Extracción de alertas georreferenciadas con niveles de confianza (ALTA, MEDIA, BAJA).
- Integración completa con estructuras PreparedBands del módulo de preprocesamiento GIS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.infer import (
    ConfidenceLevel,
    ForestAlert,
    ForestPredictionResult,
    InferenceError,
    InvalidInputShapeError,
    ModelNotFoundError,
    ONNXForestPredictor,
    WindowConfigError,
    apply_biophysical_filter,
    binarize,
    calculate_affected_area,
    calibrate_probabilities,
    calibration_errors,
    compute_otsu_threshold,
    create_blend_window,
    extract_alerts,
    fit_temperature,
    load_temperature,
    sigmoid_with_temperature,
)
from src.preprocess import GridSpec, PreparedBands, RasterSource
from src.unet import generate_reference_model

CRS_UTM = CRS.from_epsg(32718)  # UTM 18S (Huánuco)
TRANSFORM_10M = from_origin(400000.0, 8900000.0, 10.0, 10.0)


@pytest.fixture(scope="module")
def sample_onnx_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Crea un modelo ONNX ligero de referencia reutilizable para las pruebas."""
    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "unet_test_small.onnx"
    return generate_reference_model(
        output_path=model_path,
        in_channels=5,
        num_classes=1,
        base_features=16,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Pruebas de Inicialización y Validación
# ---------------------------------------------------------------------------
def test_predictor_initialization(sample_onnx_model: Path) -> None:
    """Verifica que el predictor cargue correctamente el grafo ONNX y sus metadatos."""
    predictor = ONNXForestPredictor(sample_onnx_model, threshold=0.6, pixel_size_m=10.0)
    assert predictor.model_path == sample_onnx_model.resolve()
    assert predictor.threshold == 0.6
    assert predictor.pixel_size_m == 10.0
    assert predictor.input_name == "input"
    assert predictor.output_name == "output"
    assert predictor.expected_channels == 5


def test_predictor_model_not_found(tmp_path: Path) -> None:
    """Lanza ModelNotFoundError si la ruta al archivo ONNX no existe."""
    non_existent = tmp_path / "inexistente.onnx"
    with pytest.raises(ModelNotFoundError, match="No existe el modelo ONNX"):
        ONNXForestPredictor(non_existent)


def test_blend_window_creation() -> None:
    """Verifica la generación de ventanas de mezcla ponderada."""
    w_gauss = create_blend_window(64, mode="gaussian")
    assert w_gauss.shape == (64, 64)
    assert np.isclose(w_gauss.max(), 1.0)
    assert w_gauss.min() > 0.0

    w_lin = create_blend_window(32, mode="linear")
    assert w_lin.shape == (32, 32)
    assert np.isclose(w_lin.max(), 1.0)

    w_const = create_blend_window(16, mode="constant")
    assert w_const.shape == (16, 16)
    assert np.all(w_const == 1.0)

    with pytest.raises(WindowConfigError, match="desconocido"):
        create_blend_window(32, mode="invalido")

    with pytest.raises(WindowConfigError, match=">= 1"):
        create_blend_window(0)


# ---------------------------------------------------------------------------
# Pruebas de Inferencia por Parches (Patch Inference)
# ---------------------------------------------------------------------------
def test_patch_inference(sample_onnx_model: Path) -> None:
    """Verifica inferencia sobre parche único (C, H, W) y por lotes (B, C, H, W)."""
    predictor = ONNXForestPredictor(sample_onnx_model, threshold=0.5)

    # Parche individual 3D (5 canales, 64x64)
    patch_3d = np.random.randn(5, 64, 64).astype(np.float32)
    proba_single = predictor.predict_proba(patch_3d, squeeze_single=True)
    assert proba_single.shape == (64, 64)
    assert 0.0 <= proba_single.min() and proba_single.max() <= 1.0

    # Máscara binaria
    mask_single = predictor.predict_mask(patch_3d, threshold=0.5)
    assert mask_single.shape == (64, 64)
    assert mask_single.dtype == bool

    # Lote 4D (2 parches, 5 canales, 64x64)
    batch_4d = np.random.randn(2, 5, 64, 64).astype(np.float32)
    proba_batch = predictor.predict_proba(batch_4d)
    assert proba_batch.shape == (2, 1, 64, 64)
    assert 0.0 <= proba_batch.min() and proba_batch.max() <= 1.0

    # Retorno de logits crudos
    logits = predictor.predict_raw(batch_4d)
    assert logits.shape == (2, 1, 64, 64)


def test_patch_inference_invalid_inputs(sample_onnx_model: Path) -> None:
    """Verifica control de errores ante canales o dimensiones incorrectas."""
    predictor = ONNXForestPredictor(sample_onnx_model)

    # Canal incorrecto (3 canales en vez de 5)
    with pytest.raises(InvalidInputShapeError, match="canales"):
        predictor.predict_proba(np.zeros((3, 64, 64), dtype=np.float32))

    # Dimensión inválida (2D en lugar de 3D/4D)
    with pytest.raises(InvalidInputShapeError, match="dimensiones"):
        predictor.predict_proba(np.zeros((64, 64), dtype=np.float32))

    # Manejo de NaNs en la entrada (no debe fallar ni propagar NaNs)
    x_nan = np.random.randn(5, 32, 32).astype(np.float32)
    x_nan[:, 5:10, 5:10] = np.nan
    out = predictor.predict_proba(x_nan)
    assert not np.isnan(out).any()


# ---------------------------------------------------------------------------
# Pruebas de Inferencia por Ventana Deslizante (Sliding Window)
# ---------------------------------------------------------------------------
def test_sliding_window_reconstruction(sample_onnx_model: Path) -> None:
    """Verifica que la ventana deslizante procese rásteres arbitrarios sin discontinuidades."""
    predictor = ONNXForestPredictor(sample_onnx_model)

    # Ráster grande de dimensiones no múltiplos de patch_size (300 x 350)
    image = np.random.randn(5, 300, 350).astype(np.float32)

    # Probar con los tres modos de mezcla
    for mode in ["gaussian", "linear", "constant"]:
        prob_map = predictor.predict_sliding_window(
            image=image,
            patch_size=128,
            stride=64,
            batch_size=2,
            blend_mode=mode,
        )
        assert prob_map.shape == (300, 350)
        assert 0.0 <= np.nanmin(prob_map) and np.nanmax(prob_map) <= 1.0


def test_sliding_window_smaller_than_patch(sample_onnx_model: Path) -> None:
    """Verifica el comportamiento cuando la imagen es menor al tamaño de ventana."""
    predictor = ONNXForestPredictor(sample_onnx_model)

    # Imagen pequeña (80 x 90), menor que patch_size=128
    small_img = np.random.randn(5, 80, 90).astype(np.float32)
    out = predictor.predict_sliding_window(
        image=small_img,
        patch_size=128,
        stride=64,
    )
    assert out.shape == (80, 90)


def test_sliding_window_valid_mask(sample_onnx_model: Path) -> None:
    """Verifica que la máscara de validez asigne NaN a los píxeles inválidos."""
    predictor = ONNXForestPredictor(sample_onnx_model)
    image = np.random.randn(5, 100, 100).astype(np.float32)

    valid_mask = np.ones((100, 100), dtype=bool)
    valid_mask[0:20, :] = False  # Zona inválida (p. ej. sin cobertura)

    out = predictor.predict_sliding_window(
        image=image,
        patch_size=64,
        stride=32,
        valid_mask=valid_mask,
    )
    assert np.isnan(out[0:20, :]).all()
    assert not np.isnan(out[20:, :]).any()


def test_sliding_window_invalid_config(sample_onnx_model: Path) -> None:
    """Verifica validación de parámetros de ventana deslizante."""
    predictor = ONNXForestPredictor(sample_onnx_model)
    img = np.zeros((5, 100, 100), dtype=np.float32)

    with pytest.raises(WindowConfigError, match="patch_size"):
        predictor.predict_sliding_window(img, patch_size=8)

    with pytest.raises(WindowConfigError, match="stride"):
        predictor.predict_sliding_window(img, patch_size=64, stride=100)

    with pytest.raises(WindowConfigError, match="batch_size"):
        predictor.predict_sliding_window(img, patch_size=64, stride=32, batch_size=0)


# ---------------------------------------------------------------------------
# Pruebas de Post-procesamiento y Alertas
# ---------------------------------------------------------------------------
def test_binarize() -> None:
    """Verifica binarización de probabilidades con manejo de NaN."""
    probs = np.array([[0.2, 0.5, 0.8], [np.nan, 0.4, 0.9]], dtype=np.float32)
    mask = binarize(probs, threshold=0.5)
    expected = np.array([[False, True, True], [False, False, True]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_area_calculation() -> None:
    """Verifica cálculo de superficie afectada en hectáreas según tamaño de píxel."""
    # 100 píxeles de 10m x 10m (100 m² c/u) = 10,000 m² = 1.0 ha
    mask_100 = np.zeros((20, 20), dtype=bool)
    mask_100[:10, :10] = True
    area_10m = calculate_affected_area(mask_100, pixel_size_m=10.0)
    assert np.isclose(area_10m, 1.0)

    # 100 píxeles a resolución de 20m (400 m² c/u) = 40,000 m² = 4.0 ha
    area_20m = calculate_affected_area(mask_100, pixel_size_m=20.0)
    assert np.isclose(area_20m, 4.0)

    # Máscara vacía = 0 ha
    assert calculate_affected_area(np.zeros((10, 10), dtype=bool)) == 0.0

    # Error si pixel_size_m <= 0
    with pytest.raises(ValueError, match="> 0"):
        calculate_affected_area(mask_100, pixel_size_m=0)


def test_extract_alerts_without_georef() -> None:
    """Verifica extracción de alertas poligonales en coordenadas de imagen."""
    prob_map = np.zeros((100, 100), dtype=np.float32)

    # Anomalía 1: 100 píxeles con confianza alta (0.95)
    prob_map[10:20, 10:20] = 0.95

    # Anomalía 2: 25 píxeles con confianza media (0.70)
    prob_map[50:55, 50:55] = 0.70

    alerts = extract_alerts(
        prob_map=prob_map,
        threshold=0.5,
        min_area_ha=0.0,
        pixel_size_m=10.0,
    )

    assert len(alerts) == 2

    # Orden o identificación de las alertas
    a1 = next(a for a in alerts if a.pixel_count == 100)
    assert a1.area_ha == 1.0
    assert a1.confidence_level == ConfidenceLevel.ALTA
    assert np.isclose(a1.confidence_score, 0.95)
    assert a1.centroid_geo is None
    assert a1.geometry["type"] == "Polygon"

    a2 = next(a for a in alerts if a.pixel_count == 25)
    assert a2.area_ha == 0.25
    assert a2.confidence_level == ConfidenceLevel.MEDIA
    assert np.isclose(a2.confidence_score, 0.70)


def test_extract_alerts_with_georef() -> None:
    """Verifica georreferenciación de alertas con Transform y CRS."""
    prob_map = np.zeros((50, 50), dtype=np.float32)
    prob_map[10:20, 10:20] = 0.85

    alerts = extract_alerts(
        prob_map=prob_map,
        threshold=0.5,
        transform=TRANSFORM_10M,
        crs=CRS_UTM,
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.centroid_geo is not None
    # Verificar que las coordenadas geográficas correspondan a UTM 18S
    x_geo, y_geo = alert.centroid_geo
    assert 400000.0 <= x_geo <= 401000.0
    assert 8899000.0 <= y_geo <= 8900000.0


def test_extract_alerts_min_area_filter() -> None:
    """Verifica filtrado de anomalías según umbral de superficie mínima."""
    prob_map = np.zeros((100, 100), dtype=np.float32)
    # 25 píxeles = 0.25 ha
    prob_map[10:15, 10:15] = 0.9
    # 200 píxeles = 2.0 ha
    prob_map[30:50, 30:40] = 0.9

    # Exigir al menos 1.0 ha
    alerts = extract_alerts(prob_map, threshold=0.5, min_area_ha=1.0)
    assert len(alerts) == 1
    assert alerts[0].area_ha == 2.0


def test_extract_alerts_empty() -> None:
    """Verifica que un mapa sin detecciones retorne una lista vacía."""
    prob_map = np.zeros((50, 50), dtype=np.float32)
    assert extract_alerts(prob_map, threshold=0.5) == []


# ---------------------------------------------------------------------------
# Pruebas de Integración con PreparedBands
# ---------------------------------------------------------------------------
def test_predict_prepared_integration(sample_onnx_model: Path) -> None:
    """Verifica integración directa entre PreparedBands y ONNXForestPredictor."""
    grid = GridSpec(width=64, height=64, transform=TRANSFORM_10M, crs=CRS_UTM)
    valid_mask = np.ones((64, 64), dtype=bool)

    # Proveer bandas base B4, B8, B11 como reflectancias sintéticas típicas
    b4 = np.full((64, 64), 0.05, dtype=np.float32)  # Rojo bajo en bosque
    b8 = np.full((64, 64), 0.40, dtype=np.float32)  # NIR alto en vegetación
    b11 = np.full((64, 64), 0.15, dtype=np.float32)  # SWIR moderado

    # Introducir un parche de deforestación sintética (rojo sube, NIR baja)
    b4[20:30, 20:30] = 0.25
    b8[20:30, 20:30] = 0.10

    dummy_source = RasterSource(path=Path("/tmp/s2.tif"), band=1)
    prepared = PreparedBands(
        arrays={"B4": b4, "B8": b8, "B11": b11},
        grid=grid,
        valid_mask=valid_mask,
        sources={"B4": dummy_source, "B8": dummy_source, "B11": dummy_source},
        resampling_applied={"B4": "none", "B8": "none", "B11": "none"},
    )

    predictor = ONNXForestPredictor(sample_onnx_model, threshold=0.5)

    # Inferencia con cálculo automático de NDVI y NBR
    result = predictor.predict_prepared(
        prepared=prepared,
        band_order=("B4", "B8", "B11", "NDVI", "NBR"),
        patch_size=32,
        stride=16,
    )

    assert isinstance(result, ForestPredictionResult)
    assert result.probability_map.shape == (64, 64)
    assert result.mask.shape == (64, 64)
    assert result.transform == TRANSFORM_10M
    assert result.crs == CRS_UTM
    assert isinstance(result.affected_area_ha, float)
    assert isinstance(result.alerts, list)

    summary = result.to_dict()
    assert "affected_area_ha" in summary
    assert "total_alerts" in summary


# ---------------------------------------------------------------------------
# Pruebas de Umbral Adaptativo, Calibración y Filtro Biofísico
# ---------------------------------------------------------------------------
def test_compute_otsu_threshold_bimodal() -> None:
    """Verifica que Otsu encuentre el punto de corte óptimo en una distribución bimodal."""
    # Fondo con media ~0.2, anomalías con media ~0.8
    rng = np.random.default_rng(42)
    fondo = rng.normal(loc=0.2, scale=0.05, size=(100, 100))
    anomalia = rng.normal(loc=0.8, scale=0.05, size=(20, 20))
    prob_map = fondo.copy()
    prob_map[10:30, 10:30] = anomalia
    prob_map = np.clip(prob_map, 0.0, 1.0)

    th = compute_otsu_threshold(prob_map)
    assert 0.35 <= th <= 0.65


def test_compute_otsu_threshold_edge_cases() -> None:
    """Verifica casos extremos como matrices constantes o con NaNs."""
    const_map = np.full((50, 50), 0.7, dtype=np.float32)
    assert compute_otsu_threshold(const_map) == 0.7

    nan_map = np.full((10, 10), np.nan, dtype=np.float32)
    assert compute_otsu_threshold(nan_map) == 0.5


def test_calibrate_probabilities() -> None:
    """Verifica que la calibración expanda el intervalo [vmin, vmax] a [0.0, 1.0]."""
    raw_probs = np.linspace(0.55, 0.75, num=100, dtype=np.float32).reshape(10, 10)
    calibrated = calibrate_probabilities(raw_probs, vmin=0.55, vmax=0.75)

    assert calibrated.shape == (10, 10)
    assert np.isclose(calibrated.min(), 0.0, atol=1e-3)
    assert np.isclose(calibrated.max(), 1.0, atol=1e-3)


def test_apply_biophysical_filter() -> None:
    """Verifica que el filtro descarte píxeles que no presentan caída de NBR ni NDVI."""
    # Máscara binaria con 4 píxeles positivos
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:4, 2:4] = True  # 4 píxeles

    # Tensor de 6 bandas (C, H, W):
    # T0: B4=0.05, B8=0.40, B11=0.15 (Bosque sano)
    # T1 píxeles (2,2) y (2,3): deforestación real (B4=0.20, B8=0.10, B11=0.30) -> dNDVI y dNBR altos
    # T1 píxeles (3,2) y (3,3): falso positivo sin cambio (B4=0.05, B8=0.40, B11=0.15)
    tensor = np.zeros((6, 10, 10), dtype=np.float32)
    tensor[0, :, :] = 0.05  # B4 T0
    tensor[1, :, :] = 0.40  # B8 T0
    tensor[2, :, :] = 0.15  # B11 T0

    tensor[3, :, :] = 0.05  # B4 T1 default
    tensor[4, :, :] = 0.40  # B8 T1 default
    tensor[5, :, :] = 0.15  # B11 T1 default

    # Introducir cambio real solo en (2, 2) y (2, 3)
    tensor[3, 2, 2:4] = 0.20
    tensor[4, 2, 2:4] = 0.10
    tensor[5, 2, 2:4] = 0.30

    filtered = apply_biophysical_filter(mask, tensor, min_dnbr=0.15, min_dndvi=0.10)
    assert np.count_nonzero(filtered) == 2
    assert filtered[2, 2] is np.True_
    assert filtered[2, 3] is np.True_
    assert filtered[3, 2] is np.False_
    assert filtered[3, 3] is np.False_


def test_extract_alerts_with_calibrated_map() -> None:
    """Verifica que extract_alerts calcule scores cualitativos usando el mapa calibrado."""
    raw_prob = np.full((20, 20), 0.58, dtype=np.float32)
    raw_prob[5:10, 5:10] = 0.72  # Valor raw moderado (~72%)

    calib_prob = np.full((20, 20), 0.10, dtype=np.float32)
    calib_prob[5:10, 5:10] = 0.92  # Valor calibrado alto (>80% -> ALTA)

    alerts = extract_alerts(
        prob_map=raw_prob,
        threshold=0.65,
        calibrated_prob_map=calib_prob,
    )
    assert len(alerts) == 1
    assert alerts[0].confidence_level == ConfidenceLevel.ALTA
    assert alerts[0].confidence_score >= 0.90


def test_catalogo_escenas_satelitales_reales() -> None:
    """Verifica que el catálogo de escenas satelitales reales cargue tensores válidos."""
    from app.services.copernicus import CATALOGO_ESCENAS_REALES, cargar_escena_satelital_seleccionada

    assert len(CATALOGO_ESCENAS_REALES) >= 5
    for clave, meta in CATALOGO_ESCENAS_REALES.items():
        assert "titulo" in meta
        assert "satelite" in meta
        assert "resolucion" in meta
        tensor, meta_escena, nombre = cargar_escena_satelital_seleccionada(clave)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 6  # 6 canales multitemporales (T0: B4,B8,B11; T1: B4,B8,B11)
        assert tensor.dtype == np.float32
        assert meta_escena["es_satelital_real"] is True
        assert -90.0 <= meta_escena["lat"] <= 90.0
        assert -180.0 <= meta_escena["lon"] <= 180.0
        assert len(nombre) > 0


def test_alertas_reales_geobosques_cargadas() -> None:
    """Verifica que las alertas de Geobosques MINAM carguen correctamente como datos reales."""
    from app.services.alerts_data import obtener_alertas_reales_geobosques

    df = obtener_alertas_reales_geobosques()
    assert not df.empty
    assert "PNCB-HCO-" in df["id_alerta"].iloc[0]
    assert (df["origen_dato"] == "PNCB_GEOBOSQUES_MINAM").all()
    assert (df["area_ha"] > 0).all()
    assert (df["confianza_pct"] >= 80.0).all()
    assert "zona" in df.columns
    assert "sector" in df.columns


def test_sigmoid_with_temperature_softens_probabilities() -> None:
    logits = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    p_raw = sigmoid_with_temperature(logits, temperature=1.0)
    p_soft = sigmoid_with_temperature(logits, temperature=2.5)

    # 0.0 se mantiene en 0.5
    assert np.isclose(p_raw[2], 0.5)
    assert np.isclose(p_soft[2], 0.5)

    # Las probabilidades extremas se suavizan hacia 0.5
    assert p_soft[0] > p_raw[0]
    assert p_soft[4] < p_raw[4]

    # El orden relativo (ranking) se preserva estrictamente
    assert (np.diff(p_raw) > 0).all()
    assert (np.diff(p_soft) > 0).all()


def test_calibration_errors_calculation() -> None:
    # Caso perfectamente calibrado
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    targets = np.array([0, 0, 1, 1])
    ece, mce = calibration_errors(probs, targets, n_bins=10)
    assert ece < 0.15
    assert mce < 0.15


def test_fit_temperature_finds_optimal_t() -> None:
    # Datos sobreconfiados: logits altos (+6 y -6) pero con 20% de errores
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 2, size=1000)
    # Logits que exageran la certeza (+6 / -6) con ruido
    logits = np.where(targets == 1, 4.0, -4.0) + rng.normal(0, 1.5, size=1000)
    # Inyectar 15% de ruido/errores
    flip = rng.random(size=1000) < 0.15
    targets[flip] = 1 - targets[flip]

    t_opt = fit_temperature(logits, targets, t_min=1.0, t_max=10.0)
    assert t_opt > 1.2  # Debe haber encontrado T > 1 para suavizar sobreconfianza

    p_raw = sigmoid_with_temperature(logits, temperature=1.0)
    p_cal = sigmoid_with_temperature(logits, temperature=t_opt)
    ece_raw, mce_raw = calibration_errors(p_raw, targets)
    ece_cal, mce_cal = calibration_errors(p_cal, targets)
    assert ece_cal <= ece_raw


def test_predictor_temperature_integration(sample_onnx_model: Path, tmp_path: Path) -> None:
    # Predictor con T=1.0 vs T=2.5
    pred_1 = ONNXForestPredictor(sample_onnx_model, temperature=1.0)
    pred_cal = ONNXForestPredictor(sample_onnx_model, temperature=2.5)

    x = np.random.randn(5, 64, 64).astype(np.float32)
    p1 = pred_1.predict_proba(x)
    p2 = pred_cal.predict_proba(x)

    # Probabilidades de pred_cal deben estar más cerca de 0.5
    dev1 = np.abs(p1 - 0.5)
    dev2 = np.abs(p2 - 0.5)
    assert (dev2 <= dev1 + 1e-5).all()

    # Carga desde calibration.json junto al modelo
    cal_file = tmp_path / "calibration.json"
    import json
    cal_file.write_text(json.dumps({"temperature": 3.14}))
    assert load_temperature(cal_file.parent / "dummy_model.onnx") == 3.14



