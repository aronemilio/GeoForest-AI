"""Servicios y utilidades de procesamiento ráster y demostración ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from rasterio.transform import from_origin


def peso_modelo_onnx_mb() -> float:
    """Tamaño real (MB) del modelo ONNX servido; ``0.0`` si no está disponible."""
    ruta = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "unet_sentinel2_multitemporal.onnx"
    )
    return ruta.stat().st_size / 1_000_000 if ruta.is_file() else 0.0


def generar_parche_demostracion(sector_idx: int) -> tuple[np.ndarray, dict[str, Any], str]:
    """Genera un tensor multitemporal calibrado de 6 bandas Sentinel-2 (T0: B4, B8, B11; T1: B4, B8, B11)
    para el sector seleccionado de Huánuco (256x256 píxeles a 10m de resolución espacial).
    """
    np.random.seed(100 + sector_idx * 33)
    h, w = 256, 256
    yy, xx = np.mgrid[:h, :w]

    if sector_idx == 0:
        nombre = "Bosque Montano de Carpish — Quebrada Chinchao (Bosque Nuboso / Tala)"
        b4_base = 0.07 + 0.02 * np.sin(xx / 20.0) + np.random.normal(0, 0.01, (h, w))
        b8_base = 0.68 + 0.06 * np.cos(yy / 22.0) + np.random.normal(0, 0.02, (h, w))
        b11_base = 0.12 + 0.03 * np.sin((xx + yy) / 30.0) + np.random.normal(0, 0.015, (h, w))

        dist = np.sqrt((xx - 130) ** 2 + 1.6 * (yy - 120) ** 2)
        anomalia = dist < 42
        delta_b4, delta_b8, delta_b11 = 0.28, -0.44, 0.36
        lat_cen, lon_cen = -9.7080, -76.0920

    elif sector_idx == 1:
        nombre = "Valle del Huallaga — Acomayo (Quema Agrícola y Rastrojo)"
        b4_base = 0.15 + 0.03 * np.cos(xx / 18.0) + np.random.normal(0, 0.015, (h, w))
        b8_base = 0.46 + 0.05 * np.sin(yy / 25.0) + np.random.normal(0, 0.02, (h, w))
        b11_base = 0.26 + 0.04 * np.cos((xx - yy) / 26.0) + np.random.normal(0, 0.02, (h, w))

        dist = np.sqrt(2.2 * (xx - 110) ** 2 + (yy - 140) ** 2)
        anomalia = dist < 50
        delta_b4, delta_b8, delta_b11 = 0.22, -0.32, 0.44
        lat_cen, lon_cen = -9.5850, -76.0450

    else:
        nombre = "Cuenca Media Huallaga — Churubamba (Tala Selectiva)"
        b4_base = 0.11 + 0.02 * np.sin(xx / 16.0) + np.random.normal(0, 0.01, (h, w))
        b8_base = 0.54 + 0.04 * np.cos(yy / 20.0) + np.random.normal(0, 0.02, (h, w))
        b11_base = 0.17 + 0.03 * np.sin((xx + yy) / 24.0) + np.random.normal(0, 0.015, (h, w))

        dist = np.sqrt((xx - 150) ** 2 + (yy - 95) ** 2)
        anomalia = dist < 38
        delta_b4, delta_b8, delta_b11 = 0.19, -0.30, 0.28
        lat_cen, lon_cen = -9.6250, -76.1150

    b4_t0 = np.clip(b4_base, 0.01, 0.98).astype(np.float32)
    b8_t0 = np.clip(b8_base, 0.01, 0.98).astype(np.float32)
    b11_t0 = np.clip(b11_base, 0.01, 0.98).astype(np.float32)

    b4_t1 = b4_t0.copy()
    b8_t1 = b8_t0.copy()
    b11_t1 = b11_t0.copy()

    b4_t1[anomalia] = np.clip(b4_t1[anomalia] + delta_b4, 0.01, 0.98)
    b8_t1[anomalia] = np.clip(b8_t1[anomalia] + delta_b8, 0.01, 0.98)
    b11_t1[anomalia] = np.clip(b11_t1[anomalia] + delta_b11, 0.01, 0.98)

    tensor = np.stack([b4_t0, b8_t0, b11_t0, b4_t1, b8_t1, b11_t1], axis=0)
    transform = from_origin(400000.0 + sector_idx * 6000, 8900000.0 - sector_idx * 6000, 10.0, 10.0)

    meta = {
        "crs": "EPSG:32718",
        "transform": transform,
        "lat": lat_cen,
        "lon": lon_cen,
        "pixel_size_m": 10.0,
        "anomalia_gt": anomalia,
    }
    return tensor, meta, nombre


def matriz_a_imagen_rgb(b8: np.ndarray, b4: np.ndarray, b11: np.ndarray) -> Image.Image:
    """Genera una imagen RGB en falso color infrarrojo (R=B8, G=B4, B=B11) normalizada."""
    norm_b8 = np.clip((b8 - b8.min()) / (b8.max() - b8.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    norm_b4 = np.clip((b4 - b4.min()) / (b4.max() - b4.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    norm_b11 = np.clip((b11 - b11.min()) / (b11.max() - b11.min() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([norm_b8, norm_b4, norm_b11], axis=-1), mode="RGB")


def matriz_a_imagen_ndvi(ndvi: np.ndarray) -> Image.Image:
    """Genera un mapa de color del índice NDVI (rojo = baja biomasa, verde = dosel denso)."""
    norm = np.clip((ndvi + 0.1) / 0.9, 0.0, 1.0)
    r = (np.clip(1.8 - 2.2 * norm, 0.0, 1.0) * 240).astype(np.uint8)
    g = (np.clip(2.0 * norm, 0.0, 1.0) * 220).astype(np.uint8)
    b = (np.clip(0.3 * (1.0 - norm), 0.0, 1.0) * 120).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def matriz_a_imagen_heatmap(prob: np.ndarray) -> Image.Image:
    """Genera un mapa térmico continuo de probabilidad U-Net (0 = verde, 1 = rojo alerta)."""
    p = np.clip(prob, 0.0, 1.0)
    r = (np.clip(p * 2.2, 0.0, 1.0) * 255).astype(np.uint8)
    g = (np.clip(2.0 * (1.0 - p), 0.0, 1.0) * 230).astype(np.uint8)
    b = (np.clip(0.25 * (1.0 - p), 0.0, 1.0) * 150).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def matriz_a_imagen_mascara(mask: np.ndarray) -> Image.Image:
    """Genera una imagen binaria (verde = bajo umbral, rojo = píxel sobre umbral; no es una detección validada)."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [12, 32, 18]
    rgb[mask > 0] = [255, 55, 55]
    return Image.fromarray(rgb, mode="RGB")
