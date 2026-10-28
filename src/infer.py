"""
Módulo de Inferencia ONNX y Análisis Geoespacial para GeoForest-AI Huánuco.

Implementa:
- Predicción optimizada sobre CPU mediante ONNX Runtime (InferenceSession).
- Inferencia por ventanas deslizantes (sliding window) con agregación ponderada
  (Gaussian, Linear o Constant) para mitigar artefactos en bordes de parches.
- Soporte para arreglos NumPy (C, H, W) o (B, C, H, W), así como integración directa
  con estructuras `PreparedBands` del módulo GIS de preprocesamiento.
- Post-procesamiento geoespacial: umbralización, cálculo de superficie afectada en
  hectáreas según la resolución de píxel y extracción de alertas poligonales con
  nivel de confianza (ALTA, MEDIA, BAJA) y coordenadas espaciales.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.affinity import affine_transform
from shapely.geometry import mapping, shape

from src.preprocess import PreparedBands, compute_nbr, compute_ndvi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------
class InferenceError(Exception):
    """Excepción base para errores durante la inferencia ONNX."""


class ModelNotFoundError(InferenceError, FileNotFoundError):
    """El archivo del modelo ONNX no existe en la ruta indicada."""


class InvalidInputShapeError(InferenceError, ValueError):
    """La forma o el número de canales del tensor de entrada no es compatible."""


class WindowConfigError(InferenceError, ValueError):
    """Parámetros de ventana deslizante inválidos (stride, patch_size)."""


# ---------------------------------------------------------------------------
# Estructuras de Datos y Clases de Confianza
# ---------------------------------------------------------------------------
class ConfidenceLevel(str, Enum):
    """Nivel cualitativo de confianza asignado a una alerta forestal."""

    ALTA = "ALTA"
    MEDIA = "MEDIA"
    BAJA = "BAJA"


@dataclass(frozen=True)
class ForestAlert:
    """
    Alerta geoespacial individual de anomalía o deforestación detectada.

    Attributes:
        alert_id: Identificador único secuencial de la alerta.
        area_ha: Superficie afectada calculada en hectáreas.
        pixel_count: Cantidad total de píxeles comprendidos en la alerta.
        confidence_score: Probabilidad media de los píxeles de la anomalía [0.0 - 1.0].
        confidence_level: Nivel cualitativo (ALTA, MEDIA, BAJA).
        bbox: Límites en píxeles de la envolvente ``(min_row, min_col, max_row, max_col)``.
        centroid_pixel: Centroide en coordenadas de imagen ``(row, col)``.
        centroid_geo: Centroide en coordenadas espaciales ``(x, y)`` si existe CRS.
        geometry: Geometría GeoJSON dict (polígono o multipolígono) en coordenadas espaciales.
    """

    alert_id: int
    area_ha: float
    pixel_count: int
    confidence_score: float
    confidence_level: ConfidenceLevel
    bbox: tuple[int, int, int, int]
    centroid_pixel: tuple[float, float]
    centroid_geo: tuple[float, float] | None = None
    geometry: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convierte la alerta a un diccionario serializable."""
        data = asdict(self)
        data["confidence_level"] = self.confidence_level.value
        return data


@dataclass
class ForestPredictionResult:
    """
    Resultado global de la inferencia y análisis post-procesamiento.

    Attributes:
        probability_map: Arreglo 2D float32 con probabilidades continuas en [0, 1].
        mask: Máscara booleana 2D de alerta tras aplicar el umbral de decisión.
        affected_area_ha: Superficie total acumulada afectada en hectáreas.
        alerts: Lista de anomalías individuales extraídas como :class:`ForestAlert`.
        pixel_size_m: Tamaño de píxel en metros (por defecto 10.0 m para Sentinel-2).
        threshold: Umbral de probabilidad empleado para la binarización.
        transform: Transform afín rasterio si la entrada estaba georreferenciada.
        crs: CRS del ráster original o None.
    """

    probability_map: np.ndarray
    mask: np.ndarray
    affected_area_ha: float
    alerts: list[ForestAlert]
    pixel_size_m: float = 10.0
    threshold: float = 0.5
    transform: Affine | None = None
    crs: CRS | str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Resumen estadístico del resultado."""
        return {
            "affected_area_ha": self.affected_area_ha,
            "total_alerts": len(self.alerts),
            "pixel_size_m": self.pixel_size_m,
            "threshold": self.threshold,
            "crs": str(self.crs) if self.crs else None,
            "alerts": [alert.to_dict() for alert in self.alerts],
        }


# ---------------------------------------------------------------------------
# Funciones de Post-procesamiento y Ventana de Mezcla
# ---------------------------------------------------------------------------
def create_blend_window(size: int, mode: str = "gaussian") -> np.ndarray:
    """
    Crea una matriz 2D de ponderación de tamaño (size, size) para suavizar
    bordes en inferencia por ventanas deslizantes.

    Args:
        size: Dimensión del parche cuadrado.
        mode: Tipo de ponderación: ``"gaussian"`` (recomendado), ``"linear"`` o ``"constant"``.

    Returns:
        Matriz 2D de ponderaciones normalizadas de tipo float32.
    """
    if size < 1:
        raise WindowConfigError(f"El tamaño del parche debe ser >= 1, recibido: {size}")

    mode_lower = mode.lower()
    if mode_lower == "constant":
        return np.ones((size, size), dtype=np.float32)

    if mode_lower == "linear":
        w1d = np.bartlett(size)
        w1d = np.maximum(w1d, 0.05)
        w2d = np.outer(w1d, w1d)
        return (w2d / w2d.max()).astype(np.float32)

    if mode_lower == "gaussian":
        x = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        sigma = size * 0.25
        g1d = np.exp(-0.5 * (x / sigma) ** 2)
        g2d = np.outer(g1d, g1d)
        return (g2d / g2d.max()).astype(np.float32)

    raise WindowConfigError(
        f"Modo de mezcla desconocido: '{mode}'. Opciones válidas: 'gaussian', 'linear', 'constant'."
    )


def binarize(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binariza un mapa de probabilidades aplicando un umbral de corte.

    Args:
        prob_map: Arreglo con valores numéricos continuos.
        threshold: Umbral de decisión en [0, 1].

    Returns:
        Arreglo booleano con True donde ``prob_map >= threshold`` (excluyendo NaNs).
    """
    valid = ~np.isnan(prob_map)
    return np.where(valid, prob_map >= threshold, False).astype(bool)


def calculate_affected_area(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    """
    Calcula la superficie total afectada en hectáreas a partir de una máscara binaria.

    Para Sentinel-2 con píxel de 10 m x 10 m (100 m²):
    1 píxel = 0.01 ha (100 m² / 10,000 m²/ha).

    Args:
        mask: Arreglo con la máscara booleana o de enteros (1/0).
        pixel_size_m: Resolución espacial del píxel en metros (debe ser > 0).

    Returns:
        Superficie total en hectáreas (float).
    """
    if pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m debe ser > 0, recibido: {pixel_size_m}")
    count = int(np.count_nonzero(mask))
    pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0
    return float(count * pixel_area_ha)


def compute_otsu_threshold(prob_map: np.ndarray, num_bins: int = 256) -> float:
    """Calcula el umbral óptimo de binarización bimodal según el método de Otsu."""
    valid = prob_map[~np.isnan(prob_map)]
    if valid.size == 0:
        return 0.50
    p_min, p_max = float(valid.min()), float(valid.max())
    if p_max - p_min < 1e-4:
        return round(float(p_min), 4)
    hist, bin_edges = np.histogram(valid, bins=num_bins, range=(p_min, p_max))
    hist = hist.astype(np.float64)
    total_pixels = hist.sum()
    if total_pixels == 0:
        return 0.50
    prob_bins = hist / total_pixels
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(prob_bins)
    weight2 = 1.0 - weight1
    mean1 = np.cumsum(prob_bins * bin_centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.sum(prob_bins * bin_centers) - np.cumsum(prob_bins * bin_centers)) / np.maximum(weight2, 1e-12)
    inter_class_variance = weight1 * weight2 * (mean1 - mean2) ** 2
    idx_max = int(np.argmax(inter_class_variance))
    optimal_thresh = float(bin_centers[idx_max])
    return float(np.clip(optimal_thresh, 0.10, 0.90))


def calibrate_probabilities(
    prob_map: np.ndarray,
    p_min: float | None = None,
    p_max: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Normaliza y calibra las probabilidades predichas al rango 0% a 100%."""
    valid = prob_map[~np.isnan(prob_map)]
    if valid.size == 0:
        return prob_map.astype(np.float32)

    eff_min = vmin if vmin is not None else p_min
    eff_max = vmax if vmax is not None else p_max

    if eff_min is None:
        eff_min = float(np.percentile(valid, 5))
    if eff_max is None:
        eff_max = float(np.percentile(valid, 99.5))
    if eff_max - eff_min < 1e-5:
        return np.clip(prob_map, 0.0, 1.0).astype(np.float32)
    calib = (prob_map - eff_min) / (eff_max - eff_min)
    return np.clip(calib, 0.0, 1.0).astype(np.float32)


def sigmoid_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature Scaling (Guo et al., 2017): p = sigmoid(z / T).

    Con ``T > 1`` las probabilidades extremas (0.99 / 0.01) se suavizan hacia 0.5
    sin cambiar el orden de los píxeles (el ranking y el AUC se conservan).
    """
    if temperature <= 0:
        raise ValueError(f"temperature debe ser > 0; recibido {temperature}")
    scaled = np.clip(logits / float(temperature), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)


def calibration_errors(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float]:
    """Expected y Maximum Calibration Error (ECE, MCE) binarios con ``n_bins`` bins."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        gap = abs(float(y[sel].mean()) - float(p[sel].mean()))
        ece += gap * n / p.size
        mce = max(mce, gap)
    return float(ece), float(mce)


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    t_min: float = 1.0,
    t_max: float = 10.0,
    iters: int = 60,
) -> float:
    """Ajusta T minimizando la NLL binaria sobre un conjunto de validación.

    Búsqueda de sección áurea en ``log T`` (la NLL es unimodal en T). Por defecto
    ``t_min = 1.0`` para solo suavizar (corregir sobreconfianza), nunca afilar.
    Debe ajustarse sobre datos **no vistos en entrenamiento**.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    def nll(log_t: float) -> float:
        s = z / np.exp(log_t)
        # log(1 + exp(-|s|)) estable numéricamente
        return float(np.mean(np.maximum(s, 0) - s * y + np.log1p(np.exp(-np.abs(s)))))

    lo, hi = np.log(t_min), np.log(t_max)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = nll(d)
    return float(np.exp((lo + hi) / 2.0))


def load_temperature(model_path: str | Path) -> float:
    """Lee la temperatura ajustada de ``calibration.json`` junto al modelo (1.0 si no existe)."""
    import json

    cal = Path(model_path).resolve().parent / "calibration.json"
    if not cal.is_file():
        return 1.0
    try:
        t = float(json.loads(cal.read_text(encoding="utf-8")).get("temperature", 1.0))
        return t if t > 0 else 1.0
    except (ValueError, OSError):
        return 1.0


def apply_biophysical_filter(
    mask: np.ndarray,
    tensor_6ch: np.ndarray,
    min_dnbr: float = 0.15,
    min_dndvi: float = 0.10,
) -> np.ndarray:
    """Aplica verificación biofísica multitemporal (Delta NBR o Delta NDVI)."""
    if tensor_6ch.ndim != 3 or tensor_6ch.shape[0] < 6:
        return mask
    red_t0, nir_t0, swir_t0 = tensor_6ch[0], tensor_6ch[1], tensor_6ch[2]
    ndvi_t0 = (nir_t0 - red_t0) / np.maximum(nir_t0 + red_t0, 1e-6)
    nbr_t0 = (nir_t0 - swir_t0) / np.maximum(nir_t0 + swir_t0, 1e-6)

    red_t1, nir_t1, swir_t1 = tensor_6ch[3], tensor_6ch[4], tensor_6ch[5]
    ndvi_t1 = (nir_t1 - red_t1) / np.maximum(nir_t1 + red_t1, 1e-6)
    nbr_t1 = (nir_t1 - swir_t1) / np.maximum(nir_t1 + swir_t1, 1e-6)

    delta_ndvi = ndvi_t0 - ndvi_t1
    delta_nbr = nbr_t0 - nbr_t1
    evidencia_biofisica = (delta_nbr >= min_dnbr) | (delta_ndvi >= min_dndvi)
    return (mask.astype(bool) & evidencia_biofisica)


def extract_alerts(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_area_ha: float = 0.0,
    pixel_size_m: float = 10.0,
    transform: Affine | None = None,
    crs: CRS | str | None = None,
    high_conf_threshold: float = 0.8,
    med_conf_threshold: float = 0.6,
    mask: np.ndarray | None = None,
    calibrated_prob_map: np.ndarray | None = None,
) -> list[ForestAlert]:
    """
    Extrae polígonos de alerta individuales a partir del mapa de probabilidades.

    Utiliza vectorización geoespacial para agrupar píxeles contiguos, calculando
    estadísticas clave (superficie, conteo de píxeles, confianza media, nivel cualitativo,
    centroide y geometría).

    Args:
        prob_map: Arreglo 2D de probabilidades en [0, 1].
        threshold: Umbral de binarización.
        min_area_ha: Área mínima en hectáreas para reportar una alerta.
        pixel_size_m: Tamaño del píxel en metros.
        transform: Transform afín para georreferenciar las coordenadas.
        crs: Sistema de referencia de coordenadas.
        high_conf_threshold: Umbral para clasificar como confianza ALTA.
        med_conf_threshold: Umbral para clasificar como confianza MEDIA.

    Returns:
        Lista ordenada de alertas :class:`ForestAlert`.
    """
    if prob_map.ndim != 2:
        raise InvalidInputShapeError(
            f"prob_map debe ser 2D (alto, ancho); recibido shape={prob_map.shape}"
        )

    if mask is not None:
        bin_mask = mask.astype(np.uint8)
    else:
        bin_mask = binarize(prob_map, threshold=threshold).astype(np.uint8)
    if np.count_nonzero(bin_mask) == 0:
        return []

    pixel_area_ha = (pixel_size_m * pixel_size_m) / 10000.0
    alerts: list[ForestAlert] = []
    alert_idx = 1

    # Extraer formas contiguas en coordenadas de píxel (GDALPolygonize)
    for geom_dict, val in rasterio.features.shapes(bin_mask, mask=(bin_mask == 1)):
        poly_px = shape(geom_dict)
        if poly_px.is_empty or not poly_px.is_valid:
            poly_px = poly_px.buffer(0)
            if poly_px.is_empty:
                continue

        min_col, min_row, max_col, max_row = poly_px.bounds
        r0 = max(0, int(np.floor(min_row)))
        c0 = max(0, int(np.floor(min_col)))
        r1 = min(prob_map.shape[0], int(np.ceil(max_row)))
        c1 = min(prob_map.shape[1], int(np.ceil(max_col)))

        sub_shape = (r1 - r0, c1 - c0)
        if sub_shape[0] <= 0 or sub_shape[1] <= 0:
            continue

        local_transform = Affine.translation(c0, r0)
        local_mask = rasterio.features.rasterize(
            [geom_dict],
            out_shape=sub_shape,
            transform=local_transform,
        )

        px_count = int(np.sum(local_mask == 1))
        if px_count == 0:
            continue

        area_ha = float(px_count * pixel_area_ha)
        if area_ha < min_area_ha:
            continue

        eval_map = calibrated_prob_map if calibrated_prob_map is not None else prob_map
        sub_prob = eval_map[r0:r1, c0:c1]
        valid_in_shape = (local_mask == 1) & ~np.isnan(sub_prob)
        if np.any(valid_in_shape):
            vals = sub_prob[valid_in_shape]
            core_conf = float(np.percentile(vals, 90))
            max_conf = float(np.max(vals))
            mean_conf = float(np.mean(vals))
            conf_score = float(np.clip(0.60 * core_conf + 0.25 * max_conf + 0.15 * mean_conf, 0.0, 1.0))
        else:
            conf_score = float(threshold)

        if conf_score >= high_conf_threshold:
            conf_level = ConfidenceLevel.ALTA
        elif conf_score >= med_conf_threshold:
            conf_level = ConfidenceLevel.MEDIA
        else:
            conf_level = ConfidenceLevel.BAJA

        centroid_px = (float(poly_px.centroid.y), float(poly_px.centroid.x))

        if transform is not None:
            geo_x, geo_y = transform @ (poly_px.centroid.x, poly_px.centroid.y)
            centroid_geo = (float(geo_x), float(geo_y))
            # Transformar geometría completa a coordenadas geográficas
            poly_geo = affine_transform(
                poly_px,
                [
                    transform.a,
                    transform.b,
                    transform.d,
                    transform.e,
                    transform.c,
                    transform.f,
                ],
            )
            geom_out = mapping(poly_geo)
        else:
            centroid_geo = None
            geom_out = mapping(poly_px)

        alerts.append(
            ForestAlert(
                alert_id=alert_idx,
                area_ha=round(area_ha, 4),
                pixel_count=px_count,
                confidence_score=round(conf_score, 4),
                confidence_level=conf_level,
                bbox=(r0, c0, r1, c1),
                centroid_pixel=(round(centroid_px[0], 2), round(centroid_px[1], 2)),
                centroid_geo=(
                    (round(centroid_geo[0], 4), round(centroid_geo[1], 4))
                    if centroid_geo
                    else None
                ),
                geometry=geom_out,
            )
        )
        alert_idx += 1

    return alerts


# ---------------------------------------------------------------------------
# Predictor ONNX para Detección Forestal
# ---------------------------------------------------------------------------
class ONNXForestPredictor:
    """
    Motor de inferencia sobre modelos ONNX para detección y monitoreo forestal.

    Ejecuta modelos U-Net exportados usando ONNX Runtime, ofreciendo soporte para
    imágenes ráster grandes mediante ventanas deslizantes con mezcla ponderada,
    evitando discontinuidades o artefactos en las costuras entre parches.

    Attributes:
        model_path: Ruta al modelo ONNX cargado.
        session: Instancia de :class:`ort.InferenceSession`.
        input_name: Nombre del nodo de entrada en el grafo ONNX.
        output_name: Nombre del nodo de salida en el grafo ONNX.
        expected_channels: Cantidad esperada de canales si el grafo lo fija.
        threshold: Umbral por defecto para binarización.
        pixel_size_m: Tamaño del píxel en metros para estimación de áreas.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Sequence[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        threshold: float = 0.5,
        pixel_size_m: float = 10.0,
        temperature: float | None = None,
    ) -> None:
        p = Path(model_path).resolve()
        if not p.is_file():
            raise ModelNotFoundError(f"No existe el modelo ONNX en: {p}")

        self.model_path = p
        self.threshold = threshold
        self.pixel_size_m = pixel_size_m
        self.temperature = float(temperature) if temperature is not None else load_temperature(p)

        if providers is None:
            available = set(ort.get_available_providers())
            preferred = [
                prov
                for prov in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
                if prov in available
            ]
            providers = [*preferred, "CPUExecutionProvider"]

        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 1

        try:
            self.session = ort.InferenceSession(
                str(p),
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            if providers != ["CPUExecutionProvider"]:
                try:
                    self.session = ort.InferenceSession(
                        str(p),
                        sess_options=session_options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as inner_exc:
                    raise InferenceError(
                        f"Error al inicializar sesión ONNX Runtime para {p}: {inner_exc}"
                    ) from inner_exc
            else:
                raise InferenceError(
                    f"Error al inicializar sesión ONNX Runtime para {p}: {exc}"
                ) from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs or not outputs:
            raise InferenceError("El modelo ONNX no declara entradas o salidas válidas.")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        # Detectar número de canales esperado si está fijado en la firma de entrada
        in_shape = inputs[0].shape
        if len(in_shape) >= 2 and isinstance(in_shape[1], int) and in_shape[1] > 0:
            self.expected_channels: int | None = in_shape[1]
        else:
            self.expected_channels = None

        logger.info(
            "Predictor ONNX cargado: %s (entrada: '%s', salida: '%s', canales esperados: %s)",
            self.model_path.name,
            self.input_name,
            self.output_name,
            self.expected_channels,
        )

    def _prepare_input(self, x: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Valida y normaliza las dimensiones del arreglo de entrada a (B, C, H, W).

        Returns:
            Tupla (tensor_normalizado, is_single_image).
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        if x.ndim == 3:
            # (C, H, W) -> expandir a (1, C, H, W)
            x_batch = np.expand_dims(x, axis=0)
            is_single = True
        elif x.ndim == 4:
            x_batch = x
            is_single = False
        else:
            raise InvalidInputShapeError(
                f"El arreglo de entrada debe tener 3 (C, H, W) o 4 (B, C, H, W) dimensiones; "
                f"recibidas {x.ndim} dimensiones (forma {x.shape})."
            )

        if self.expected_channels is not None and x_batch.shape[1] != self.expected_channels:
            raise InvalidInputShapeError(
                f"Se esperaban {self.expected_channels} canales pero la entrada contiene "
                f"{x_batch.shape[1]} canales."
            )

        x_batch = x_batch.astype(np.float32, copy=False)

        # Tratar NaNs para evitar propagación descontrolada en la convolución
        if np.isnan(x_batch).any():
            x_batch = np.nan_to_num(x_batch, nan=0.0)

        return x_batch, is_single

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """
        Ejecuta la inferencia directa retornando los logits crudos del modelo.

        Args:
            x: Arreglo NumPy con forma ``(C, H, W)`` o ``(B, C, H, W)``.

        Returns:
            Logits predichos con forma ``(num_classes, H, W)`` o ``(B, num_classes, H, W)``.
        """
        x_batch, is_single = self._prepare_input(x)
        raw_out = self.session.run([self.output_name], {self.input_name: x_batch})[0]

        if is_single:
            return raw_out[0]
        return raw_out

    def predict_proba(
        self,
        x: np.ndarray,
        apply_sigmoid: bool = True,
        squeeze_single: bool = True,
        temperature: float | None = None,
    ) -> np.ndarray:
        """
        Ejecuta la inferencia y transforma los logits en probabilidades en [0, 1].

        Aplica Temperature Scaling (Guo et al., 2017) seguido de sigmoide para
        salida uniclase o softmax escalado para multiclase.

        Args:
            x: Entrada ``(C, H, W)`` o ``(B, C, H, W)``.
            apply_sigmoid: Si False retorna los logits directamente.
            squeeze_single: Si True y la entrada era 3D con 1 clase, retorna 2D ``(H, W)``.
            temperature: Factor de temperatura T > 0 (por defecto self.temperature).

        Returns:
            Arreglo de probabilidades en [0, 1].
        """
        logits = self.predict_raw(x)

        if not apply_sigmoid:
            if squeeze_single and x.ndim == 3 and logits.shape[0] == 1:
                return logits[0]
            return logits

        temp = self.temperature if temperature is None else float(temperature)

        # Inferencia con activación de salida y escalado de temperatura
        if logits.ndim >= 3:
            channel_dim = 0 if x.ndim == 3 else 1
            num_classes = logits.shape[channel_dim]

            if num_classes == 1:
                probs = sigmoid_with_temperature(logits, temperature=temp)
                if squeeze_single and x.ndim == 3:
                    probs = probs[0]
                return probs
            else:
                # Softmax escalado por temperatura a lo largo de la dimensión de clases
                scaled = logits / max(temp, 1e-4)
                shift = scaled - np.max(scaled, axis=channel_dim, keepdims=True)
                exp_vals = np.exp(np.clip(shift, -50.0, 50.0))
                probs = exp_vals / np.sum(exp_vals, axis=channel_dim, keepdims=True)
                return probs

        # Fallback genérico sigmoide con temperatura
        return sigmoid_with_temperature(logits, temperature=temp)

    def predict_mask(
        self,
        x: np.ndarray,
        threshold: float | None = None,
        squeeze_single: bool = True,
    ) -> np.ndarray:
        """
        Calcula la máscara binaria aplicando el umbral de decisión.

        Args:
            x: Entrada ``(C, H, W)`` o ``(B, C, H, W)``.
            threshold: Umbral a aplicar (por defecto el configurado en la clase).
            squeeze_single: Comprimir canales unitarios.

        Returns:
            Máscara booleana.
        """
        th = self.threshold if threshold is None else threshold
        probs = self.predict_proba(x, apply_sigmoid=True, squeeze_single=squeeze_single)
        return (probs >= th).astype(bool)

    def predict_sliding_window(
        self,
        image: np.ndarray,
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 4,
        blend_mode: str = "gaussian",
        apply_sigmoid: bool = True,
        valid_mask: np.ndarray | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        """
        Inferencia por parches deslizantes con agregación ponderada continua.

        Permite segmentar rásteres grandes que superan el tamaño de memoria,
        eliminando líneas de discontinuidad y artefactos perimetrales.

        Args:
            image: Arreglo de entrada con dimensiones ``(C, H, W)``.
            patch_size: Dimensión de cada ventana cuadrada (por defecto 256).
            stride: Paso entre ventanas sucesivas (por defecto 128, 50% solapamiento).
            batch_size: Cantidad de parches procesados en paralelo por inferencia.
            blend_mode: Tipo de ponderación: ``"gaussian"``, ``"linear"`` o ``"constant"``.
            apply_sigmoid: Si True aplica activación para retornar probabilidades en [0, 1].
            valid_mask: Máscara booleana opcional de píxeles válidos (H, W).

        Returns:
            Mapa de predicción continuo con forma ``(H, W)`` (o ``(num_classes, H, W)``).
        """
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if image.ndim != 3:
            raise InvalidInputShapeError(
                f"predict_sliding_window requiere imagen 3D (C, H, W); recibida forma {image.shape}"
            )

        if patch_size < 16:
            raise WindowConfigError(f"patch_size debe ser >= 16; recibido: {patch_size}")
        if stride < 1 or stride > patch_size:
            raise WindowConfigError(
                f"stride debe estar entre 1 y patch_size ({patch_size}); recibido: {stride}"
            )
        if batch_size < 1:
            raise WindowConfigError(f"batch_size debe ser >= 1; recibido: {batch_size}")

        c, h, w = image.shape

        # Manejo de imágenes con dimensiones menores a patch_size mediante relleno reflejado
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)

        if pad_h > 0 or pad_w > 0:
            padded = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="edge",
            )
        else:
            padded = image

        _, h_pad, w_pad = padded.shape

        # Coordenadas de corte a lo largo de cada eje asegurando cobertura de los bordes
        def _get_coords(dim_len: int) -> list[int]:
            if dim_len <= patch_size:
                return [0]
            coords = list(range(0, dim_len - patch_size + 1, stride))
            if coords[-1] != dim_len - patch_size:
                coords.append(dim_len - patch_size)
            return coords

        y_coords = _get_coords(h_pad)
        x_coords = _get_coords(w_pad)
        patch_coords = [(y, x) for y in y_coords for x in x_coords]

        weight_win = create_blend_window(patch_size, mode=blend_mode)

        # Realizar un pase preliminar para determinar el número de canales de salida
        sample_patch = padded[:, 0:patch_size, 0:patch_size]
        sample_out = self.predict_proba(
            sample_patch,
            apply_sigmoid=apply_sigmoid,
            squeeze_single=False,
            temperature=temperature,
        )
        num_classes = sample_out.shape[0]

        # Acumuladores con soporte para múltiples clases
        output_acc = np.zeros((num_classes, h_pad, w_pad), dtype=np.float32)
        weight_acc = np.zeros((h_pad, w_pad), dtype=np.float32)

        # Procesar parches en lotes de tamaño batch_size
        for idx in range(0, len(patch_coords), batch_size):
            batch_slice = patch_coords[idx : idx + batch_size]
            patches = [padded[:, y : y + patch_size, x : x + patch_size] for y, x in batch_slice]
            batch_tensor = np.stack(patches, axis=0)

            batch_preds = self.predict_proba(
                batch_tensor,
                apply_sigmoid=apply_sigmoid,
                squeeze_single=False,
                temperature=temperature,
            )

            for pred, (y, x) in zip(batch_preds, batch_slice):
                output_acc[:, y : y + patch_size, x : x + patch_size] += pred * weight_win
                weight_acc[y : y + patch_size, x : x + patch_size] += weight_win

        # Normalización por el mapa de pesos acumulados
        denom = np.maximum(weight_acc[np.newaxis, ...], 1e-7)
        result = output_acc / denom

        # Recortar de vuelta al tamaño original
        result = result[:, :h, :w]

        # Aplicar máscara de píxeles válidos si se suministra o se deriva
        if valid_mask is not None:
            if valid_mask.shape != (h, w):
                raise InvalidInputShapeError(
                    f"valid_mask tiene forma {valid_mask.shape}, se esperaba {(h, w)}"
                )
            for ch in range(num_classes):
                result[ch, ~valid_mask] = np.nan

        # Si num_classes es 1, aplanar a 2D (h, w) para conveniencia
        if num_classes == 1:
            return result[0]
        return result

    def predict_prepared(
        self,
        prepared: PreparedBands,
        band_order: Sequence[str] = ("B4", "B8", "B11", "NDVI", "NBR"),
        patch_size: int = 256,
        stride: int = 128,
        batch_size: int = 4,
        blend_mode: str = "gaussian",
        threshold: float | None = None,
        min_area_ha: float = 0.0,
        temperature: float | None = None,
    ) -> ForestPredictionResult:
        """
        Ejecuta la predicción integral sobre un objeto :class:`PreparedBands`.

        Calcula automáticamente índices espectrales faltantes (NDVI, NBR), apila
        las bandas solicitadas, ejecuta la inferencia por parches continuos,
        binariza el resultado y extrae alertas con georreferenciación completa.

        Args:
            prepared: Bandas alineadas del módulo de preprocesamiento.
            band_order: Secuencia de bandas e índices a proveer a la red.
            patch_size: Tamaño de cada parche espacial.
            stride: Paso entre parches consecutivos.
            batch_size: Parches procesados en lote.
            blend_mode: Tipo de función de mezcla perimetral.
            threshold: Umbral de decisión.
            min_area_ha: Superficie mínima para registrar alertas.

        Returns:
            Instancia completa de :class:`ForestPredictionResult`.
        """
        th = self.threshold if threshold is None else threshold

        # Asegurar disponibilidad de índices derivados requeridos
        available_arrays = dict(prepared.arrays)
        if "NDVI" in band_order and "NDVI" not in available_arrays:
            available_arrays["NDVI"] = compute_ndvi(prepared)
        if "NBR" in band_order and "NBR" not in available_arrays:
            available_arrays["NBR"] = compute_nbr(prepared)

        # Validar y apilar bandas
        stacked_bands: list[np.ndarray] = []
        for name in band_order:
            if name not in available_arrays:
                raise InferenceError(
                    f"La banda o índice '{name}' no está disponible ni pudo calcularse. "
                    f"Disponibles: {sorted(available_arrays.keys())}"
                )
            stacked_bands.append(available_arrays[name])

        image = np.stack(stacked_bands, axis=0)

        # Inferencia por ventana deslizante
        prob_map = self.predict_sliding_window(
            image=image,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            blend_mode=blend_mode,
            apply_sigmoid=True,
            valid_mask=prepared.valid_mask,
            temperature=temperature,
        )

        binary_mask = binarize(prob_map, threshold=th)
        affected_area_ha = calculate_affected_area(
            binary_mask,
            pixel_size_m=self.pixel_size_m,
        )

        # Extraer alertas georreferenciadas
        alerts = extract_alerts(
            prob_map=prob_map,
            threshold=th,
            min_area_ha=min_area_ha,
            pixel_size_m=self.pixel_size_m,
            transform=prepared.grid.transform,
            crs=prepared.grid.crs,
        )

        return ForestPredictionResult(
            probability_map=prob_map,
            mask=binary_mask,
            affected_area_ha=affected_area_ha,
            alerts=alerts,
            pixel_size_m=self.pixel_size_m,
            threshold=th,
            transform=prepared.grid.transform,
            crs=prepared.grid.crs,
        )
