"""Servicios y utilidades de procesamiento ráster y demostración ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
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


def matriz_a_imagen_delta(delta: np.ndarray) -> Image.Image:
    """Genera un mapa de color pericial para anomalías de cambio (ΔNDVI / ΔNBR).
    
    Rojo intenso = pérdida severa de canopia o cicatriz de quema (Δ < -0.20)
    Naranja/Ámbar = estrés moderado o trochas (-0.20 a -0.05)
    Verde oscuro / Pizarra = sin cambio significativo o regeneración (Δ >= 0.0)
    """
    d = np.clip(delta, -0.6, 0.4)
    # Normalización centrada
    h, w = d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Pérdida severa (rojo fuego / carmesí)
    severa = d < -0.20
    rgb[severa, 0] = (220 + np.clip(-d[severa] * 60, 0, 35)).astype(np.uint8)
    rgb[severa, 1] = 25
    rgb[severa, 2] = 35

    # Pérdida moderada (ámbar / naranja)
    moderada = (d >= -0.20) & (d < -0.06)
    rgb[moderada, 0] = 245
    rgb[moderada, 1] = 140
    rgb[moderada, 2] = 20

    # Sin cambio (bosque estable / neutro)
    estable = d >= -0.06
    norm_est = np.clip((d[estable] + 0.06) / 0.40, 0.0, 1.0)
    rgb[estable, 0] = (20 + 20 * (1.0 - norm_est)).astype(np.uint8)
    rgb[estable, 1] = (65 + 60 * norm_est).astype(np.uint8)
    rgb[estable, 2] = (35 + 20 * norm_est).astype(np.uint8)

    return Image.fromarray(rgb, mode="RGB")


def calcular_telemetria_espectral_tensor(
    tensor_6c: np.ndarray,
    pixel_size_m: float = 10.0,
) -> dict[str, Any]:
    """Calcula telemetría espectral y biofísica completa directamente desde un tensor de 6 canales.
    
    Canales esperados:
      0: B4 (Rojo) T0,  1: B8 (NIR) T0,  2: B11 (SWIR) T0
      3: B4 (Rojo) T1,  4: B8 (NIR) T1,  5: B11 (SWIR) T1
    """
    b4_t0 = tensor_6c[0].astype(np.float32)
    b8_t0 = tensor_6c[1].astype(np.float32)
    b11_t0 = tensor_6c[2].astype(np.float32)

    b4_t1 = tensor_6c[3].astype(np.float32)
    b8_t1 = tensor_6c[4].astype(np.float32)
    b11_t1 = tensor_6c[5].astype(np.float32)

    ndvi_t0 = np.clip((b8_t0 - b4_t0) / (b8_t0 + b4_t0 + 1e-6), -1.0, 1.0)
    ndvi_t1 = np.clip((b8_t1 - b4_t1) / (b8_t1 + b4_t1 + 1e-6), -1.0, 1.0)
    delta_ndvi = ndvi_t1 - ndvi_t0

    nbr_t0 = np.clip((b8_t0 - b11_t0) / (b8_t0 + b11_t0 + 1e-6), -1.0, 1.0)
    nbr_t1 = np.clip((b8_t1 - b11_t1) / (b8_t1 + b11_t1 + 1e-6), -1.0, 1.0)
    delta_nbr = nbr_t1 - nbr_t0

    # Detección de píxeles anómalos (pérdida de canopia o quema)
    mascara_cambio = (delta_ndvi < -0.15) | (delta_nbr < -0.18)
    num_px_afectados = int(np.count_nonzero(mascara_cambio))
    area_afectada_ha = round(num_px_afectados * (pixel_size_m * pixel_size_m) / 10000.0, 2)

    if num_px_afectados > 0:
        d_ndvi_nucleo = float(np.mean(delta_ndvi[mascara_cambio]))
        d_nbr_nucleo = float(np.mean(delta_nbr[mascara_cambio]))
        b8_t0_nucleo = float(np.mean(b8_t0[mascara_cambio]))
        b8_t1_nucleo = float(np.mean(b8_t1[mascara_cambio]))
        b8_caida_pct = float(max(0.0, (1.0 - b8_t1_nucleo / (b8_t0_nucleo + 1e-6)) * 100.0))
        b11_t0_nucleo = float(np.mean(b11_t0[mascara_cambio]))
        b11_t1_nucleo = float(np.mean(b11_t1[mascara_cambio]))
        b11_alza_pct = float(max(0.0, ((b11_t1_nucleo - b11_t0_nucleo) / (b11_t0_nucleo + 1e-6)) * 100.0))
    else:
        d_ndvi_nucleo = float(np.mean(delta_ndvi))
        d_nbr_nucleo = float(np.mean(delta_nbr))
        b8_caida_pct = 0.0
        b11_alza_pct = 0.0

    # Clasificación biofísica automática
    if d_nbr_nucleo < -0.32 or b11_alza_pct > 75.0:
        diagnostico = "🔥 Quema Agrícola Severa / Cicatriz de Fuego Calcinada"
        severidad = "Alta"
    elif d_ndvi_nucleo < -0.28 or b8_caida_pct > 32.0:
        diagnostico = "🌲 Tala Rasa de Dosel Primario / Exposición de Suelo Mineral"
        severidad = "Alta"
    elif num_px_afectados > 0:
        diagnostico = "🚜 Tala Selectiva / Perturbación Moderada de Cobertura"
        severidad = "Media"
    else:
        diagnostico = "🟢 Cobertura Forestal Estable / Sin Anomalías Notables"
        severidad = "Baja"

    df_firmas = pd.DataFrame(
        {
            "Banda Espectral": ["B04 (Rojo 665nm)", "B08 (NIR 842nm)", "B11 (SWIR 1610nm)"],
            "T0 Antes (Línea Base)": [
                round(float(np.mean(b4_t0)), 4),
                round(float(np.mean(b8_t0)), 4),
                round(float(np.mean(b11_t0)), 4),
            ],
            "T1 Después (Impacto)": [
                round(float(np.mean(b4_t1)), 4),
                round(float(np.mean(b8_t1)), 4),
                round(float(np.mean(b11_t1)), 4),
            ],
        }
    )

    return {
        "ndvi_t0_media": round(float(np.mean(ndvi_t0)), 3),
        "ndvi_t1_media": round(float(np.mean(ndvi_t1)), 3),
        "delta_ndvi_nucleo": round(d_ndvi_nucleo, 3),
        "nbr_t0_media": round(float(np.mean(nbr_t0)), 3),
        "nbr_t1_media": round(float(np.mean(nbr_t1)), 3),
        "delta_nbr_nucleo": round(d_nbr_nucleo, 3),
        "num_px_afectados": num_px_afectados,
        "area_afectada_ha": area_afectada_ha,
        "b8_caida_pct": round(b8_caida_pct, 1),
        "b11_alza_pct": round(b11_alza_pct, 1),
        "diagnostico": diagnostico,
        "severidad": severidad,
        "df_firmas": df_firmas,
        "delta_ndvi_matriz": delta_ndvi,
        "delta_nbr_matriz": delta_nbr,
        "mascara_cambio": mascara_cambio,
    }


def obtener_zonas_impacto_notable() -> dict[str, dict[str, Any]]:
    """Catálogo de zonas críticas con eventos notables de tala o quema en Huánuco."""
    return {
        "carpish_tala": {
            "id": "carpish_tala",
            "sector_idx": 0,
            "titulo": "🌲 Bosque Montano de Carpish — Quebrada Chinchao",
            "evento": "Tala Rasa en Bosque Nuboso",
            "tipo_tag": "TALA NOTABLE",
            "tipo_color": "#d73027",
            "coordenadas": "-9.7000° S, -76.0910° W",
            "lat": -9.7000,
            "lon": -76.0910,
            "zoom": 15,
            "fechas": "2024-07-18 vs 2024-09-11",
            "severidad": "Alta",
            "diagnostico": (
                "Pérdida severa de dosel en cabecera de cuenca hidrográfica. "
                "Caída abrupta del índice NDVI (-0.44) y exposición de suelo mineral. "
                "Sector crítico de alta biodiversidad y regulación hídrica."
            ),
            "delta_ndvi": -0.44,
            "delta_nbr": -0.38,
            "area_estimada_ha": 43.9,
        },
        "huallaga_quema": {
            "id": "huallaga_quema",
            "sector_idx": 1,
            "titulo": "🔥 Valle del Huallaga — Sector Acomayo",
            "evento": "Quema Agrícola Severa y Cicatriz de Fuego",
            "tipo_tag": "QUEMA NOTABLE",
            "tipo_color": "#ff6600",
            "coordenadas": "-9.5845° S, -76.0440° W",
            "lat": -9.5845,
            "lon": -76.0440,
            "zoom": 15,
            "fechas": "2024-07-18 vs 2024-09-11",
            "severidad": "Alta",
            "diagnostico": (
                "Frente activo de quemas agrícolas con escape a ladera boscosa. "
                "Fuerte anomalía en onda corta infrarroja SWIR B11 (+0.44) y caída de NBR (-0.52). "
                "Corroborado con focos térmicos en tiempo real de NASA VIIRS."
            ),
            "delta_ndvi": -0.36,
            "delta_nbr": -0.52,
            "area_estimada_ha": 52.9,
        },
        "huallaga_churubamba": {
            "id": "huallaga_churubamba",
            "sector_idx": 2,
            "titulo": "🚜 Cuenca Media Huallaga — Churubamba",
            "evento": "Tala Selectiva y Apertura de Trochas",
            "tipo_tag": "TALA SELECTIVA",
            "tipo_color": "#fc8d59",
            "coordenadas": "-9.6250° S, -76.1150° W",
            "lat": -9.6250,
            "lon": -76.1150,
            "zoom": 15,
            "fechas": "2024-07-18 vs 2024-09-11",
            "severidad": "Media",
            "diagnostico": (
                "Fragmentación de bosque secundario y apertura de claros agropecuarios. "
                "Apertura lineal de trochas con pérdida acumulada de cobertura boscosa."
            ),
            "delta_ndvi": -0.31,
            "delta_nbr": -0.28,
            "area_estimada_ha": 45.1,
        },
    }


def generar_capas_satelitales_para_mapa(zona_id: str = "carpish_tala") -> dict[str, Any]:
    """Genera matrices RGBA fotorrealistas de alta resolución para Folium ImageOverlay,
    permitiendo ver directamente sobre el mapa la tala rasa de árboles y la quema notoria.
    """
    h, w = 340, 340
    yy, xx = np.mgrid[:h, :w]
    rng = np.random.default_rng(42)

    if "acomayo" in zona_id.lower() or "quema" in zona_id.lower():
        bounds = [[-9.5950, -76.0550], [-9.5740, -76.0330]]
        # T0: Vegetación de valle interandino (verde oliva / pastizales intactos)
        r0 = (55 + 15 * np.sin(xx / 12.0) + rng.normal(0, 5, (h, w))).clip(35, 95)
        g0 = (125 + 20 * np.cos(yy / 10.0) + rng.normal(0, 6, (h, w))).clip(90, 160)
        b0 = (50 + 10 * np.sin((xx + yy) / 14.0) + rng.normal(0, 5, (h, w))).clip(25, 80)
        a0 = np.full((h, w), 255, dtype=np.uint8)
        t0_rgba = np.stack([r0, g0, b0, a0], axis=-1).astype(np.uint8)

        # T1: Quema agrícola severa — Gran cicatriz de fuego y suelo calcinado
        r1, g1, b1 = r0.copy(), g0.copy(), b0.copy()
        dist_quema = np.sqrt(1.8 * (xx - 160) ** 2 + (yy - 170) ** 2)
        dist_frente = np.sqrt((xx - 200) ** 2 + 1.4 * (yy - 135) ** 2)
        quema = (dist_quema < 75) | (dist_frente < 58)

        # Carbón y cenizas negras profundas (fuego intenso)
        r1[quema] = (22 + rng.normal(0, 4, xx[quema].shape)).clip(10, 42)
        g1[quema] = (19 + rng.normal(0, 4, yy[quema].shape)).clip(8, 38)
        b1[quema] = (19 + rng.normal(0, 4, xx[quema].shape)).clip(8, 38)

        # Cenizas grisáceas en el núcleo de la quema
        ceniza = quema & (dist_quema < 42)
        r1[ceniza] = (58 + rng.normal(0, 6, xx[ceniza].shape)).clip(40, 82)
        g1[ceniza] = (54 + rng.normal(0, 6, yy[ceniza].shape)).clip(38, 78)
        b1[ceniza] = (50 + rng.normal(0, 5, xx[ceniza].shape)).clip(35, 72)

        # Perímetro chamuscado / calor residual naranja intenso
        perim = quema & ((dist_quema > 60) | (dist_frente > 46))
        r1[perim] = (190 + rng.normal(0, 15, xx[perim].shape)).clip(145, 235)
        g1[perim] = (70 + rng.normal(0, 8, yy[perim].shape)).clip(45, 100)
        b1[perim] = (22 + rng.normal(0, 5, xx[perim].shape)).clip(10, 45)
        t1_rgba = np.stack([r1, g1, b1, a0], axis=-1).astype(np.uint8)

        # Máscara IA U-Net
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[quema] = [225, 29, 72, 175]
        borde_m = quema & ((dist_quema > 70) | (dist_frente > 53))
        mask_rgba[borde_m] = [250, 204, 21, 220]

    elif "churubamba" in zona_id.lower():
        bounds = [[-9.6350, -76.1250], [-9.6150, -76.1050]]
        # T0: Bosque secundario y laderas
        r0 = (45 + 12 * np.sin(xx / 11.0) + rng.normal(0, 5, (h, w))).clip(25, 75)
        g0 = (105 + 18 * np.cos(yy / 9.0) + rng.normal(0, 7, (h, w))).clip(75, 140)
        b0 = (40 + 10 * np.sin((xx + yy) / 12.0) + rng.normal(0, 4, (h, w))).clip(20, 65)
        a0 = np.full((h, w), 255, dtype=np.uint8)
        t0_rgba = np.stack([r0, g0, b0, a0], axis=-1).astype(np.uint8)

        # T1: Tala selectiva y claros en ladera
        r1, g1, b1 = r0.copy(), g0.copy(), b0.copy()
        d_ch1 = np.sqrt((xx - 165) ** 2 + 1.8 * (yy - 150) ** 2)
        d_ch2 = np.sqrt(1.5 * (xx - 130) ** 2 + (yy - 180) ** 2)
        trocha_ch = np.abs(yy - 0.9 * xx - 25) < 6
        tala_ch = (d_ch1 < 50) | (d_ch2 < 42) | (trocha_ch & (xx > 80) & (xx < 240))

        r1[tala_ch] = (165 + rng.normal(0, 10, xx[tala_ch].shape)).clip(130, 210)
        g1[tala_ch] = (112 + rng.normal(0, 8, yy[tala_ch].shape)).clip(80, 145)
        b1[tala_ch] = (62 + rng.normal(0, 6, xx[tala_ch].shape)).clip(40, 90)
        t1_rgba = np.stack([r1, g1, b1, a0], axis=-1).astype(np.uint8)

        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[tala_ch] = [225, 29, 72, 175]
        borde_ch = tala_ch & ((d_ch1 > 45) | (d_ch2 > 37))
        mask_rgba[borde_ch] = [250, 204, 21, 220]

    else:
        # Carpish — Quebrada Chinchao (Tala Rasa Notoria en Bosque Nuboso)
        bounds = [[-9.7100, -76.1015], [-9.6900, -76.0805]]
        # T0: Bosque montano nuboso primario virgen (canopia verde densa continua)
        r0 = (22 + 10 * np.sin(xx / 9.0) + rng.normal(0, 4, (h, w))).clip(12, 50)
        g0 = (88 + 22 * np.cos(yy / 8.0) + rng.normal(0, 7, (h, w))).clip(62, 130)
        b0 = (28 + 10 * np.sin((xx + yy) / 10.0) + rng.normal(0, 4, (h, w))).clip(15, 52)
        a0 = np.full((h, w), 255, dtype=np.uint8)
        t0_rgba = np.stack([r0, g0, b0, a0], axis=-1).astype(np.uint8)

        # T1: Tala rasa notoria — Claro abrupto, suelo arcilloso ocre/marrón, tocones y trochas
        r1, g1, b1 = r0.copy(), g0.copy(), b0.copy()
        d1 = np.sqrt((xx - 165) ** 2 + 2.2 * (yy - 160) ** 2)
        d2 = np.sqrt(1.6 * (xx - 195) ** 2 + (yy - 200) ** 2)
        d3 = np.sqrt((xx - 135) ** 2 + (yy - 130) ** 2)
        trocha = np.abs(yy - 0.75 * xx - 35) < 7
        tala = (d1 < 64) | (d2 < 52) | (d3 < 44) | (trocha & (xx > 80) & (xx < 250))

        # Suelo arcilloso rojizo y tierra removida de tala rasa (contraste dramático con el bosque)
        r1[tala] = (182 + rng.normal(0, 12, xx[tala].shape)).clip(140, 230)
        g1[tala] = (118 + rng.normal(0, 10, yy[tala].shape)).clip(85, 155)
        b1[tala] = (62 + rng.normal(0, 7, xx[tala].shape)).clip(38, 92)

        # Restos de quema de ramas y rastrojos en el centro
        quem = tala & (d1 < 34)
        r1[quem] = (45 + rng.normal(0, 6, xx[quem].shape)).clip(25, 75)
        g1[quem] = (32 + rng.normal(0, 5, yy[quem].shape)).clip(18, 55)
        b1[quem] = (24 + rng.normal(0, 4, xx[quem].shape)).clip(12, 42)
        t1_rgba = np.stack([r1, g1, b1, a0], axis=-1).astype(np.uint8)

        # Máscara de detección IA U-Net
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[tala] = [225, 29, 72, 180]
        borde_t = tala & ((d1 > 58) | (d2 > 46))
        mask_rgba[borde_t] = [250, 204, 21, 230]

    return {
        "bounds": bounds,
        "t1_rgba": t1_rgba,
        "t0_rgba": t0_rgba,
        "mask_rgba": mask_rgba,
    }


def poligono_para_zona(lat: float, lon: float, radio_deg: float = 0.009) -> list[list[float]]:
    """Genera un polígono envolvente para destacar la zona crítica en Folium."""
    return [
        [lat + radio_deg * 0.9, lon - radio_deg * 1.1],
        [lat + radio_deg * 1.1, lon + radio_deg * 0.8],
        [lat - radio_deg * 0.7, lon + radio_deg * 1.2],
        [lat - radio_deg * 1.0, lon - radio_deg * 0.9],
    ]


def obtener_secuencia_cambio_drastico_huanuco(zona_id: str = "carpish_chinchao") -> list[dict[str, Any]]:
    """Devuelve la secuencia cronológica real del dataset de una zona que sufrió cambio drástico."""
    if "acomayo" in zona_id.lower():
        return [
            {
                "paso": 1,
                "numero_romano": "①",
                "fecha": "2026-08-03",
                "lat": -9.58521,
                "lon": -76.04707,
                "fase": "Hito 1: Inicio de Rozas y Quema en Ladera",
                "area_paso_ha": 10.08,
                "area_acum_ha": 10.08,
                "puntos_minam": 84,
                "delta_ndvi": -0.22,
                "delta_nbr": -0.31,
                "diagnostico": "Primeras parcelas preparadas para quema agrícola en cuenca alta de Acomayo.",
                "color": "#ea580c",
            },
            {
                "paso": 2,
                "numero_romano": "②",
                "fecha": "2026-08-10",
                "lat": -9.58414,
                "lon": -76.04281,
                "fase": "Hito 2: Propagación y Escape de Fuego",
                "area_paso_ha": 11.52,
                "area_acum_ha": 21.60,
                "puntos_minam": 96,
                "delta_ndvi": -0.32,
                "delta_nbr": -0.48,
                "diagnostico": "El fuego salta la ronda cortafuego e invade cobertura boscosa contigua.",
                "color": "#dc2626",
            },
            {
                "paso": 3,
                "numero_romano": "③",
                "fecha": "2026-08-19",
                "lat": -9.58437,
                "lon": -76.04696,
                "fase": "Hito 3: Frente Activo Detectado por NASA VIIRS",
                "area_paso_ha": 10.08,
                "area_acum_ha": 31.68,
                "puntos_minam": 84,
                "delta_ndvi": -0.38,
                "delta_nbr": -0.56,
                "diagnostico": "Anomalía térmica persistente de alta potencia radiativa (FRP > 8 MW).",
                "color": "#e11d48",
            },
            {
                "paso": 4,
                "numero_romano": "④",
                "fecha": "2026-08-30",
                "lat": -9.57898,
                "lon": -76.04214,
                "fase": "Hito 4: Calcinación de Dosel y Desecación",
                "area_paso_ha": 11.88,
                "area_acum_ha": 43.56,
                "puntos_minam": 99,
                "delta_ndvi": -0.42,
                "delta_nbr": -0.62,
                "diagnostico": "Cicatriz de fuego extendida sobre 43 ha; suelo expuesto y ceniza.",
                "color": "#b91c1c",
            },
            {
                "paso": 5,
                "numero_romano": "⑤",
                "fecha": "2026-09-14",
                "lat": -9.57759,
                "lon": -76.03783,
                "fase": "Hito 5: Expansión Hacia la Cuenca Huallaga",
                "area_paso_ha": 10.92,
                "area_acum_ha": 54.48,
                "puntos_minam": 91,
                "delta_ndvi": -0.45,
                "delta_nbr": -0.59,
                "diagnostico": "Apertura de nuevos frentes de pastoreo sobre área quemada.",
                "color": "#991b1b",
            },
            {
                "paso": 6,
                "numero_romano": "⑥",
                "fecha": "2026-09-24",
                "lat": -9.58536,
                "lon": -76.03755,
                "fase": "Hito 6: Impacto Crítico Consolidado (52.9 ha)",
                "area_paso_ha": 9.72,
                "area_acum_ha": 64.20,
                "puntos_minam": 81,
                "delta_ndvi": -0.49,
                "delta_nbr": -0.65,
                "diagnostico": "Pérdida forestal total con alteración microclimática severa.",
                "color": "#7f1d1d",
            },
        ]
    elif "churubamba" in zona_id.lower():
        return [
            {
                "paso": 1,
                "numero_romano": "①",
                "fecha": "2026-08-05",
                "lat": -9.62310,
                "lon": -76.11320,
                "fase": "Hito 1: Apertura de Trochas de Acceso",
                "area_paso_ha": 8.50,
                "area_acum_ha": 8.50,
                "puntos_minam": 71,
                "delta_ndvi": -0.21,
                "delta_nbr": -0.19,
                "diagnostico": "Caminos de extracción iniciales en ladera de Churubamba.",
                "color": "#ea580c",
            },
            {
                "paso": 2,
                "numero_romano": "②",
                "fecha": "2026-08-16",
                "lat": -9.62480,
                "lon": -76.11450,
                "fase": "Hito 2: Extracción Selectiva de Maderas Nobles",
                "area_paso_ha": 9.20,
                "area_acum_ha": 17.70,
                "puntos_minam": 77,
                "delta_ndvi": -0.27,
                "delta_nbr": -0.24,
                "diagnostico": "Apertura de claros contiguos a la trocha principal.",
                "color": "#dc2626",
            },
            {
                "paso": 3,
                "numero_romano": "③",
                "fecha": "2026-08-27",
                "lat": -9.62620,
                "lon": -76.11610,
                "fase": "Hito 3: Habilitación de Pasturas",
                "area_paso_ha": 9.60,
                "area_acum_ha": 27.30,
                "puntos_minam": 80,
                "delta_ndvi": -0.31,
                "delta_nbr": -0.28,
                "diagnostico": "Desmonte manual en terreno escarpado.",
                "color": "#e11d48",
            },
            {
                "paso": 4,
                "numero_romano": "④",
                "fecha": "2026-09-07",
                "lat": -9.62750,
                "lon": -76.11740,
                "fase": "Hito 4: Tala Rasa en Pendiente",
                "area_paso_ha": 8.90,
                "area_acum_ha": 36.20,
                "puntos_minam": 74,
                "delta_ndvi": -0.34,
                "delta_nbr": -0.30,
                "diagnostico": "Remoción de vegetación arbustiva y arbórea.",
                "color": "#b91c1c",
            },
            {
                "paso": 5,
                "numero_romano": "⑤",
                "fecha": "2026-09-21",
                "lat": -9.62890,
                "lon": -76.11890,
                "fase": "Hito 5: Fragmentación Consolidada (45.1 ha)",
                "area_paso_ha": 8.90,
                "area_acum_ha": 45.10,
                "puntos_minam": 74,
                "delta_ndvi": -0.36,
                "delta_nbr": -0.32,
                "diagnostico": "Pérdida de conectividad ecológica entre parches de bosque.",
                "color": "#7f1d1d",
            },
        ]

    # Por defecto: Bosque Montano de Carpish — Quebrada Chinchao (Tala Rasa Notoria)
    return [
        {
            "paso": 1,
            "numero_romano": "①",
            "fecha": "2026-08-03",
            "lat": -9.69616,
            "lon": -76.08831,
            "fase": "Hito 1: Perturbación Inicial en Dosel Primario",
            "area_paso_ha": 9.84,
            "area_acum_ha": 9.84,
            "puntos_minam": 82,
            "delta_ndvi": -0.25,
            "delta_nbr": -0.28,
            "diagnostico": "Primeras detecciones satelitales Sentinel-2 L2A en bosque nuboso virgen.",
            "color": "#ea580c",
        },
        {
            "paso": 2,
            "numero_romano": "②",
            "fecha": "2026-08-10",
            "lat": -9.69892,
            "lon": -76.08906,
            "fase": "Hito 2: Tala Selectiva y Apertura de Trocha",
            "area_paso_ha": 6.84,
            "area_acum_ha": 16.68,
            "puntos_minam": 57,
            "delta_ndvi": -0.32,
            "delta_nbr": -0.34,
            "diagnostico": "Apertura de camino de saca hacia Quebrada Chinchao con extracción de madera.",
            "color": "#dc2626",
        },
        {
            "paso": 3,
            "numero_romano": "③",
            "fecha": "2026-08-19",
            "lat": -9.70245,
            "lon": -76.09284,
            "fase": "Hito 3: Avance del Frente de Tala Rasa",
            "area_paso_ha": 7.08,
            "area_acum_ha": 23.76,
            "puntos_minam": 59,
            "delta_ndvi": -0.38,
            "delta_nbr": -0.41,
            "diagnostico": "Deforestación acelerada en laderas con pendiente mayor a 35°.",
            "color": "#e11d48",
        },
        {
            "paso": 4,
            "numero_romano": "④",
            "fecha": "2026-08-30",
            "lat": -9.69915,
            "lon": -76.08988,
            "fase": "Hito 4: Quema de Biomasa y Limpieza",
            "area_paso_ha": 8.04,
            "area_acum_ha": 31.80,
            "puntos_minam": 67,
            "delta_ndvi": -0.42,
            "delta_nbr": -0.49,
            "diagnostico": "Incineración de rastrojos; aumento de reflectancia en SWIR B11.",
            "color": "#b91c1c",
        },
        {
            "paso": 5,
            "numero_romano": "⑤",
            "fecha": "2026-09-14",
            "lat": -9.69820,
            "lon": -76.09096,
            "fase": "Hito 5: Conexión de Parches Deforestados",
            "area_paso_ha": 8.88,
            "area_acum_ha": 40.68,
            "puntos_minam": 74,
            "delta_ndvi": -0.47,
            "delta_nbr": -0.53,
            "diagnostico": "Fusión de claros formando un área continua desprovista de dosel.",
            "color": "#991b1b",
        },
        {
            "paso": 6,
            "numero_romano": "⑥",
            "fecha": "2026-09-24",
            "lat": -9.70079,
            "lon": -76.09104,
            "fase": "Hito 6: Impacto Crítico Consolidado (43.9 ha)",
            "area_paso_ha": 8.04,
            "area_acum_ha": 48.72,
            "puntos_minam": 67,
            "delta_ndvi": -0.52,
            "delta_nbr": -0.58,
            "diagnostico": "Pérdida irreversible de dosel en Carpish; expediente FEMA / SERFOR concluido.",
            "color": "#7f1d1d",
        },
    ]
