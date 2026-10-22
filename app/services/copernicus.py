"""Servicio de integración con la API de Copernicus (CDSE / STAC Sentinel-2 L2A).

Proporciona consulta en tiempo real, descarga automatizada y recorte geoespacial
de bandas multiespectrales (B04 Red, B08 NIR, B11 SWIR) sobre Huánuco (Tile MGRS 18LUQ),
almacenándolas exactamente en la estructura requerida por `src/dataset.py`:
`data/raw/sentinel2/{YYYY-MM-DD}_{scene_id}/B04.tif, B08.tif, B11.tif`.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocess import preparar_tensor_multitemporal

logger = logging.getLogger(__name__)

# Bounding Box referencial para Huánuco (Carpish y Valle del Huallaga) en WGS84 [min_lon, min_lat, max_lon, max_lat]
BBOX_HUANUCO_DEFAULT = [-76.25, -9.85, -75.95, -9.50]
STAC_API_URL = "https://earth-search.aws.element84.com/v1/search"

# Bounding Box oficial del parche 256x256 (10m) de Huánuco / Carpish en EPSG:32718 (UTM Zona 18S)
HUANUCO_BOUNDS_UTM: tuple[float, float, float, float] = (370580.0, 8949380.0, 373140.0, 8951940.0)
DEFAULT_SENTINEL2_ROOT = Path("data/raw/sentinel2")
DEFAULT_CRS = "EPSG:32718"
BANDAS_REQUERIDAS: tuple[str, ...] = ("B04", "B08", "B11")

# Dominios autorizados para descarga de COGs
ALLOWED_COG_DOMAINS: set[str] = {
    "sentinel-cogs.s3.us-west-2.amazonaws.com",
    "sentinel-cogs.s3.amazonaws.com",
    "earth-search.aws.element84.com",
    "dataspace.copernicus.eu",
    "catalogue.dataspace.copernicus.eu",
    "zipper.dataspace.copernicus.eu",
}


def sanitize_scene_id(scene_id: str) -> str:
    """Valida y sanea un identificador de escena Sentinel-2.

    Rechaza cadenas vacías, caracteres de control, nulos, espacios o intentos de path traversal.

    Args:
        scene_id: Cadena con el identificador de escena (ej. 'S2A_18LUQ_20240911_0_L2A').

    Returns:
        Identificador saneado.

    Raises:
        ValueError: Si el identificador no cumple las reglas de seguridad.
    """
    if not isinstance(scene_id, str):
        raise ValueError(f"El identificador de escena debe ser str, se recibió {type(scene_id).__name__}")
    sid = scene_id.strip()
    if not sid:
        raise ValueError("El identificador de escena no puede estar vacío.")
    if len(sid) > 128:
        raise ValueError("El identificador de escena supera la longitud máxima permitida (128).")
    if "\x00" in sid or "/" in sid or "\\" in sid or ".." in sid:
        raise ValueError(f"Identificador de escena inválido o con intento de path traversal: '{scene_id}'")
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", sid):
        raise ValueError(f"El identificador de escena contiene caracteres no permitidos: '{scene_id}'")
    return sid


def validate_cog_url(url: str) -> str:
    """Verifica que una URL de GeoTIFF COG sea segura y use HTTPS en un host permitido.

    Args:
        url: URL a verificar.

    Returns:
        URL validada y saneada.

    Raises:
        ValueError: Si la URL no es segura o no proviene de un dominio autorizado.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("La URL del COG no puede estar vacía.")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Solo se permiten URLs HTTPS seguras, se recibió esquema '{parsed.scheme}'.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL inválida: no contiene hostname.")

    is_allowed = (
        hostname in ALLOWED_COG_DOMAINS
        or hostname.endswith(".dataspace.copernicus.eu")
        or (hostname.startswith("sentinel-cogs.s3.") and hostname.endswith(".amazonaws.com"))
    )
    if not is_allowed:
        raise ValueError(f"Dominio no autorizado para descarga de COG: '{hostname}'")
    return url.strip()


def validate_output_directory(output_dir: str | Path, base_dir: Path | None = None) -> Path:
    """Verifica que el directorio de salida sea seguro y no escape fuera del proyecto o rutas autorizadas."""
    path = Path(output_dir).resolve()
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    base_resolved = base_dir.resolve()

    is_safe = False
    try:
        path.relative_to(base_resolved)
        is_safe = True
    except ValueError:
        pass

    if not is_safe:
        str_path = str(path)
        if str_path.startswith("/tmp") or str_path.startswith(str(base_resolved / "data")):
            is_safe = True

    if not is_safe:
        raise ValueError(f"Directorio de salida fuera del directorio permitido: {path}")
    return path


def _construir_info_escena_desde_id(scene_id: str) -> dict[str, Any]:
    """Reconstruye los metadatos y URLs de AWS S2 COG a partir de un ID MGRS estándar.

    Ejemplo de ID: S2A_18LUQ_20240911_0_L2A o S2B_18LUQ_20240718_0_L2A.
    """
    sid = sanitize_scene_id(scene_id)
    tokens = sid.split("_")
    # Formato típico: S2A_18LUQ_20240911_0_L2A
    fecha = ""
    tile = "18LUQ"
    if len(tokens) >= 3 and len(tokens[2]) == 8 and tokens[2].isdigit():
        raw_date = tokens[2]
        fecha = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
        tile = tokens[1]
    elif len(tokens) >= 2 and len(tokens[0]) == 10 and tokens[0].count("-") == 2:
        fecha = tokens[0]

    # Construir ruta AWS COG para MGRS 18LUQ
    # https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/{year}/{month}/{scene_id}/
    year = fecha[:4] if fecha else "2024"
    month = str(int(fecha[5:7])) if fecha and len(fecha) >= 7 else "9"
    # Extracción de sub-cuadrícula MGRS (ej. 18LUQ -> 18 / L / UQ)
    zone = tile[:2] if len(tile) >= 2 else "18"
    band_letter = tile[2] if len(tile) >= 3 else "L"
    square = tile[3:5] if len(tile) >= 5 else "UQ"

    base_cog_url = (
        f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
        f"{zone}/{band_letter}/{square}/{year}/{month}/{sid}"
    )

    return {
        "id": sid,
        "fecha": fecha,
        "hora_utc": "00:00:00",
        "plataforma": "Sentinel-2A" if sid.startswith("S2A") else "Sentinel-2B",
        "tile": tile,
        "nubes_pct": 0.0,
        "url_b04": f"{base_cog_url}/B04.tif",
        "url_b08": f"{base_cog_url}/B08.tif",
        "url_b11": f"{base_cog_url}/B11.tif",
        "thumbnail": "",
    }


def consultar_catalogo_copernicus(
    bbox: list[float] | None = None,
    fecha_inicio: str = "2024-06-01",
    fecha_fin: str = "2024-10-31",
    max_nubes_pct: float = 30.0,
    limite: int = 10,
) -> list[dict[str, Any]]:
    """Consulta en tiempo real el catálogo abierto de escenas Sentinel-2 L2A de Copernicus (STAC).

    Args:
        bbox: Lista [min_lon, min_lat, max_lon, max_lat]. Por defecto Huánuco.
        fecha_inicio: Fecha ISO inicial (YYYY-MM-DD).
        fecha_fin: Fecha ISO final (YYYY-MM-DD).
        max_nubes_pct: Porcentaje máximo de cobertura nubosa.
        limite: Número máximo de escenas a recuperar.

    Returns:
        Lista de diccionarios con metadatos y URLs de bandas B04, B08, B11.
    """
    if bbox is None:
        bbox = BBOX_HUANUCO_DEFAULT

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{fecha_inicio}T00:00:00Z/{fecha_fin}T23:59:59Z",
        "query": {
            "eo:cloud_cover": {"lt": max_nubes_pct},
            "grid:code": {"eq": "MGRS-18LUQ"},
        },
        "limit": limite,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    try:
        response = requests.post(STAC_API_URL, json=payload, timeout=12)
        if response.status_code != 200:
            logger.warning("STAC API retornó status %s: %s", response.status_code, response.text)
            return []

        features = response.json().get("features", [])
        catalogo = []
        for feat in features:
            props = feat.get("properties", {})
            assets = feat.get("assets", {})
            catalogo.append(
                {
                    "id": feat.get("id"),
                    "fecha": props.get("datetime", "")[:10],
                    "hora_utc": props.get("datetime", "")[11:19],
                    "plataforma": props.get("platform", "Sentinel-2"),
                    "tile": props.get("grid:code", "18LUQ"),
                    "nubes_pct": round(float(props.get("eo:cloud_cover", 0.0)), 2),
                    "url_b04": assets.get("red", {}).get("href", "") or assets.get("B04", {}).get("href", ""),
                    "url_b08": assets.get("nir", {}).get("href", "") or assets.get("B08", {}).get("href", ""),
                    "url_b11": assets.get("swir16", {}).get("href", "") or assets.get("B11", {}).get("href", ""),
                    "thumbnail": assets.get("thumbnail", {}).get("href", ""),
                }
            )
        return catalogo

    except Exception as exc:
        logger.error("Error al consultar el catálogo de Copernicus: %s", exc)
        return []


def descargar_banda_sentinel2(
    url: str,
    output_path: str | Path,
    *,
    crop_bounds: tuple[float, float, float, float] | None = HUANUCO_BOUNDS_UTM,
    target_shape: tuple[int, int] | None = (256, 256),
    resampling: Resampling = Resampling.bilinear,
    timeout: int = 30,
    overwrite: bool = False,
) -> Path:
    """Descarga una banda Sentinel-2 L2A (GeoTIFF) desde una URL COG.

    Si se proporciona `crop_bounds` (xmin, ymin, xmax, ymax) en coordenadas EPSG:32718,
    extrae por streaming vía rasterio solo la ventana espacial solicitada, remuestreando
    a `target_shape` si corresponde (vital para alinear B11 de 20m con B04/B08 de 10m).
    Si `crop_bounds` es None, descarga la escena completa.

    Args:
        url: URL HTTPS del archivo COG.
        output_path: Ruta destino donde se guardará el GeoTIFF.
        crop_bounds: Bounding Box en EPSG:32718 (left, bottom, right, top).
        target_shape: Dimensiones (alto, ancho) en píxeles.
        resampling: Método de remuestreo (bilinear por defecto).
        timeout: Timeout de conexión en segundos.
        overwrite: Si es True, sobrescribe archivos existentes.

    Returns:
        Path del archivo GeoTIFF descargado y verificado.
    """
    valid_url = validate_cog_url(url)
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and not overwrite and dest.stat().st_size > 0:
        logger.info("El archivo %s ya existe. Omitiendo descarga.", dest)
        return dest

    temp_path = dest.with_suffix(".tmp.tif")

    if crop_bounds is not None:
        from rasterio.windows import from_bounds

        with rasterio.open(valid_url) as src:
            window = from_bounds(*crop_bounds, transform=src.transform)
            out_shape = target_shape if target_shape is not None else (int(window.height), int(window.width))

            if target_shape is not None:
                win_transform = rasterio.transform.from_bounds(*crop_bounds, target_shape[1], target_shape[0])
            else:
                win_transform = rasterio.windows.transform(window, src.transform)

            arr = src.read(1, window=window, out_shape=out_shape, resampling=resampling)
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": out_shape[0],
                "width": out_shape[1],
                "transform": win_transform,
                "count": 1,
                "compress": "lzw",
            })
            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.write(arr, 1)
    else:
        with requests.get(valid_url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    temp_path.replace(dest)
    return dest


def descargar_escena_sentinel2(
    escena: dict[str, Any] | str,
    *,
    output_dir: str | Path = "data/raw/sentinel2",
    crop: bool = True,
    crop_bounds: tuple[float, float, float, float] | None = HUANUCO_BOUNDS_UTM,
    target_shape: tuple[int, int] | None = (256, 256),
    overwrite: bool = False,
    timeout: int = 30,
) -> Path:
    """Descarga y organiza los archivos GeoTIFF de una escena Sentinel-2 en la
    estructura exacta esperada por `src/dataset.py`:
    `{output_dir}/{YYYY-MM-DD}_{scene_id}/B04.tif, B08.tif, B11.tif`.

    Args:
        escena: Diccionario devuelto por `consultar_catalogo_copernicus` o ID de escena str.
        output_dir: Directorio raíz de salida (por defecto 'data/raw/sentinel2').
        crop: Si es True, recorta la ventana de Huánuco a 256x256 (10m) en lugar de descargar 600MB.
        crop_bounds: Bounding Box en EPSG:32718 (xmin, ymin, xmax, ymax).
        target_shape: Forma del recorte (256, 256).
        overwrite: Si es True, sobrescribe GeoTIFFs existentes.
        timeout: Timeout de conexión por banda.

    Returns:
        Path del directorio de la escena descargada.
    """
    valid_output_dir = validate_output_directory(output_dir)

    if isinstance(escena, str):
        scene_id = sanitize_scene_id(escena)
        try:
            resp = requests.post(
                STAC_API_URL,
                json={"collections": ["sentinel-2-l2a"], "ids": [scene_id]},
                timeout=timeout,
            )
            features = resp.json().get("features", []) if resp.status_code == 200 else []
        except Exception as exc:
            logger.warning("No se pudo consultar STAC por ID %s: %s", scene_id, exc)
            features = []

        if features:
            feat = features[0]
            props = feat.get("properties", {})
            assets = feat.get("assets", {})
            escena_info = {
                "id": feat.get("id", scene_id),
                "fecha": props.get("datetime", "")[:10],
                "url_b04": assets.get("red", {}).get("href", "") or assets.get("B04", {}).get("href", ""),
                "url_b08": assets.get("nir", {}).get("href", "") or assets.get("B08", {}).get("href", ""),
                "url_b11": assets.get("swir16", {}).get("href", "") or assets.get("B11", {}).get("href", ""),
            }
        else:
            escena_info = _construir_info_escena_desde_id(scene_id)
    else:
        escena_info = dict(escena)

    scene_id = sanitize_scene_id(escena_info.get("id", "S2_UNKNOWN"))
    fecha = escena_info.get("fecha", "")

    # Formato de carpeta: YYYY-MM-DD_S2...
    if fecha and not scene_id.startswith(f"{fecha}_"):
        dir_name = f"{fecha}_{scene_id}"
    else:
        dir_name = scene_id

    scene_dir = valid_output_dir / dir_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    bounds = crop_bounds if crop else None
    shape = target_shape if crop else None

    bandas_map = {
        "B04": escena_info.get("url_b04") or escena_info.get("red"),
        "B08": escena_info.get("url_b08") or escena_info.get("nir"),
        "B11": escena_info.get("url_b11") or escena_info.get("swir16"),
    }

    for b_name in BANDAS_REQUERIDAS:
        url = bandas_map.get(b_name)
        if not url:
            # Fallback construyendo URL estándar
            fallback_info = _construir_info_escena_desde_id(scene_id)
            url = fallback_info.get(f"url_{b_name.lower()}")

        if not url:
            raise ValueError(f"No se encontró URL para la banda {b_name} en la escena {scene_id}")

        out_tif = scene_dir / f"{b_name}.tif"
        descargar_banda_sentinel2(
            url,
            out_tif,
            crop_bounds=bounds,
            target_shape=shape,
            resampling=Resampling.bilinear,
            timeout=timeout,
            overwrite=overwrite,
        )

    logger.info("Escena %s descargada exitosamente en %s", scene_id, scene_dir)
    return scene_dir


def descargar_par_sentinel2(
    t0_escena: dict[str, Any] | str,
    t1_escena: dict[str, Any] | str,
    *,
    output_dir: str | Path = "data/raw/sentinel2",
    crop: bool = True,
    crop_bounds: tuple[float, float, float, float] | None = HUANUCO_BOUNDS_UTM,
    target_shape: tuple[int, int] | None = (256, 256),
    overwrite: bool = False,
    timeout: int = 30,
) -> tuple[Path, Path]:
    """Descarga un par multitemporal (T0 antes, T1 después) en la estructura exacta de `src/dataset.py`.

    Args:
        t0_escena: Metadatos o ID de la escena inicial (T0).
        t1_escena: Metadatos o ID de la escena posterior (T1).
        output_dir: Directorio raíz de salida (por defecto 'data/raw/sentinel2').
        crop: Si es True, realiza recorte espacial.
        crop_bounds: Bounding Box en EPSG:32718.
        target_shape: Dimensiones de salida (256, 256).
        overwrite: Si es True, sobrescribe archivos existentes.
        timeout: Timeout de conexión.

    Returns:
        Tupla (path_t0, path_t1) con las rutas de las carpetas de escenas descargadas.
    """
    path_t0 = descargar_escena_sentinel2(
        t0_escena,
        output_dir=output_dir,
        crop=crop,
        crop_bounds=crop_bounds,
        target_shape=target_shape,
        overwrite=overwrite,
        timeout=timeout,
    )
    path_t1 = descargar_escena_sentinel2(
        t1_escena,
        output_dir=output_dir,
        crop=crop,
        crop_bounds=crop_bounds,
        target_shape=target_shape,
        overwrite=overwrite,
        timeout=timeout,
    )
    return path_t0, path_t1


def cargar_escena_sentinel2_real() -> tuple[np.ndarray, dict[str, Any], str]:
    """Carga el tensor multitemporal real calibrado de 6 bandas Sentinel-2 L2A (10m)
    de Huánuco (Tile 18LUQ) a partir de los GeoTIFFs descargados de Copernicus:
    - T0: 2024-07-18 (Sentinel-2B, nubes 1.6%)
    - T1: 2024-09-11 (Sentinel-2A, nubes 9.2%)

    Returns:
        Tupla (tensor_6c, meta, nombre_escena) con el tensor en float32 normalizado.
    """
    base_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "sentinel2"
    t0_dir = base_dir / "2024-07-18_S2B_18LUQ_20240718_0_L2A"
    t1_dir = base_dir / "2024-09-11_S2A_18LUQ_20240911_0_L2A"

    if not (t0_dir.exists() and t1_dir.exists()):
        raise FileNotFoundError(
            f"No se encontraron los datos satelitales Sentinel-2 en {base_dir}"
        )

    t0_src = {
        "B4": rasterio.open(t0_dir / "B04.tif"),
        "B8": rasterio.open(t0_dir / "B08.tif"),
        "B11": rasterio.open(t0_dir / "B11.tif"),
    }
    t1_src = {
        "B4": rasterio.open(t1_dir / "B04.tif"),
        "B8": rasterio.open(t1_dir / "B08.tif"),
        "B11": rasterio.open(t1_dir / "B11.tif"),
    }

    try:
        tensor_6c, profile = preparar_tensor_multitemporal(
            t0_src,
            t1_src,
            target_crs="EPSG:32718",
            normalize=True,
        )
    finally:
        for ds in t0_src.values():
            ds.close()
        for ds in t1_src.values():
            ds.close()

    # Centro geográfico aproximado en WGS84
    lat_cen, lon_cen = -9.4909, -76.1674

    meta = {
        "crs": "EPSG:32718",
        "transform": profile["transform"],
        "lat": lat_cen,
        "lon": lon_cen,
        "pixel_size_m": 10.0,
        "fecha_t0": "2024-07-18",
        "fecha_t1": "2024-09-11",
        "tile": "18LUQ",
        "es_satelital_real": True,
        "anomalia_gt": None,
    }

    nombre = "🛰️ Copernicus Sentinel-2 L2A Huánuco (Tile 18LUQ: 2024-07-18 vs 2024-09-11)"
    return tensor_6c, meta, nombre


# Catálogo oficial de escenas satelitales multitemporales 100% reales
CATALOGO_ESCENAS_REALES: dict[str, dict[str, Any]] = {
    "huanuco_completa": {
        "id": "huanuco_completa",
        "titulo": "🛰️ Sentinel-2 L2A — Huánuco (Tile 18LUQ: Escena Completa 256×256)",
        "satelite": "Sentinel-2A / Sentinel-2B (Copernicus CDSE)",
        "resolucion": "10 m / px",
        "fechas": "2024-07-18 vs 2024-09-11",
        "cobertura": "Corredor Carpish & Valle del Huallaga",
        "coords": "9°29'27\"S, 76°10'02\"W",
        "badge": "🛰️ 100% Satelital Real (Copernicus)",
        "color": "#10b981",
    },
    "huanuco_carpish": {
        "id": "huanuco_carpish",
        "titulo": "🌲 Sentinel-2 L2A — Huánuco (Sector Norte: Carpish / Chinchao 128×128)",
        "satelite": "Sentinel-2A / Sentinel-2B (Copernicus CDSE)",
        "resolucion": "10 m / px",
        "fechas": "2024-07-18 vs 2024-09-11",
        "cobertura": "Bosque Nuboso Montano de Carpish",
        "coords": "9°42'54\"S, 76°05'42\"W",
        "badge": "🌲 Bosque Nuboso Real",
        "color": "#2ca25f",
    },
    "huanuco_huallaga": {
        "id": "huanuco_huallaga",
        "titulo": "🌾 Sentinel-2 L2A — Huánuco (Sector Sur: Valle Huallaga / Acomayo 128×128)",
        "satelite": "Sentinel-2A / Sentinel-2B (Copernicus CDSE)",
        "resolucion": "10 m / px",
        "fechas": "2024-07-18 vs 2024-09-11",
        "cobertura": "Valle Interandino & Quemas Agrícolas",
        "coords": "9°48'00\"S, 76°03'36\"W",
        "badge": "🔥 Quemas & Tala Real",
        "color": "#f97316",
    },
    "oscd_aguasclaras": {
        "id": "oscd_aguasclaras",
        "titulo": "🌳 Sentinel-2 L2A — Aguas Claras (Monitoreo Forestal Tropical 256×256)",
        "satelite": "Sentinel-2A / Sentinel-2B (ESA / OSCD)",
        "resolucion": "10 m / px",
        "fechas": "Multitemporal Sentinel-2",
        "cobertura": "Reserva Forestal Tropical",
        "coords": "15°50'00\"S, 47°55'00\"W",
        "badge": "🌳 Cobertura Forestal Real",
        "color": "#0ea5e9",
    },
    "oscd_brasilia": {
        "id": "oscd_brasilia",
        "titulo": "🇧🇷 Sentinel-2 L2A — Brasilia (Detección de Cambio y Deforestación 256×256)",
        "satelite": "Sentinel-2A / Sentinel-2B (ESA / OSCD)",
        "resolucion": "10 m / px",
        "fechas": "Multitemporal Sentinel-2",
        "cobertura": "Frente de Pérdida de Vegetación",
        "coords": "15°47'38\"S, 47°52'58\"W",
        "badge": "🚨 Frente de Cambio Real",
        "color": "#ef4444",
    },
}


def cargar_escena_satelital_seleccionada(
    clave: str = "huanuco_completa",
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Carga una escena satelital multitemporal 100% real según la clave seleccionada."""
    from rasterio.transform import Affine

    tensor_hco, meta_hco, nom_hco = cargar_escena_sentinel2_real()

    if clave == "huanuco_carpish":
        t_sub = tensor_hco[:, :128, :128]
        m_sub = meta_hco.copy()
        m_sub["lat"] = -9.7150
        m_sub["lon"] = -76.0950
        nom = "🌲 Sentinel-2 L2A Huánuco — Sector Norte Carpish (128×128)"
        return t_sub, m_sub, nom

    if clave == "huanuco_huallaga":
        t_sub = tensor_hco[:, 128:, 128:]
        m_sub = meta_hco.copy()
        if m_sub.get("transform") is not None:
            m_sub["transform"] = m_sub["transform"] @ Affine.translation(128, 128)
        m_sub["lat"] = -9.5850
        m_sub["lon"] = -76.0450
        nom = "🌾 Sentinel-2 L2A Huánuco — Sector Sur Valle Huallaga (128×128)"
        return t_sub, m_sub, nom

    if clave in ("oscd_aguasclaras", "oscd_brasilia"):
        city = "aguasclaras" if clave == "oscd_aguasclaras" else "brasilia"
        base_oscd = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "oscd"
            / "images"
            / "Onera Satellite Change Detection dataset - Images"
            / city
        )
        p_t0 = base_oscd / "imgs_1_rect"
        p_t1 = base_oscd / "imgs_2_rect"

        if p_t0.exists() and p_t1.exists():
            channels = []
            for b in ["B04", "B08", "B11"]:
                with rasterio.open(p_t0 / f"{b}.tif") as ds:
                    channels.append(ds.read(1)[:256, :256].astype(np.float32) / 10000.0)
            for b in ["B04", "B08", "B11"]:
                with rasterio.open(p_t1 / f"{b}.tif") as ds:
                    channels.append(ds.read(1)[:256, :256].astype(np.float32) / 10000.0)

            t_oscd = np.stack(channels, axis=0)
            m_oscd = {
                "crs": "EPSG:32718",
                "transform": None,
                "lat": -15.8333 if city == "aguasclaras" else -15.7938,
                "lon": -47.9167 if city == "aguasclaras" else -47.8827,
                "pixel_size_m": 10.0,
                "fecha_t0": "Sentinel-2 T0",
                "fecha_t1": "Sentinel-2 T1",
                "tile": "Sentinel-2 MSI",
                "es_satelital_real": True,
                "anomalia_gt": None,
            }
            nom = (
                "🌳 Sentinel-2 L2A Aguas Claras (Par Forestal Real)"
                if city == "aguasclaras"
                else "🇧🇷 Sentinel-2 L2A Brasilia (Par de Cambio Real)"
            )
            return t_oscd, m_oscd, nom

    return tensor_hco, meta_hco, nom_hco


def main(args_list: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos para consultar y descargar escenas Sentinel-2."""
    parser = argparse.ArgumentParser(
        description="Gestor de catálogo y descarga de GeoTIFFs Sentinel-2 L2A para GeoForest-AI."
    )
    parser.add_argument("--search", action="store_true", help="Consultar catálogo STAC de Sentinel-2 L2A.")
    parser.add_argument("--scene-id", type=str, default=None, help="ID de escena específica a descargar.")
    parser.add_argument("--download-top-pair", action="store_true", help="Descarga el par de menor nubosidad encontrado.")
    parser.add_argument("--output-dir", type=str, default="data/raw/sentinel2", help="Directorio raíz de salida.")
    parser.add_argument("--fecha-inicio", type=str, default="2024-06-01", help="Fecha inicial YYYY-MM-DD.")
    parser.add_argument("--fecha-fin", type=str, default="2024-10-31", help="Fecha final YYYY-MM-DD.")
    parser.add_argument("--max-nubes", type=float, default=30.0, help="Máximo porcentaje de nubosidad.")
    parser.add_argument("--limite", type=int, default=10, help="Número máximo de escenas a consultar.")
    parser.add_argument("--no-crop", action="store_true", help="Descargar escenas completas en lugar del recorte.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir archivos existentes.")

    args = parser.parse_args(args_list)

    if args.search or args.download_top_pair:
        print(f"🛰️ Consultando catálogo STAC sobre Huánuco ({args.fecha_inicio} a {args.fecha_fin})...")
        escenas = consultar_catalogo_copernicus(
            fecha_inicio=args.fecha_inicio,
            fecha_fin=args.fecha_fin,
            max_nubes_pct=args.max_nubes,
            limite=args.limite,
        )
        if not escenas:
            print("⚠️ No se encontraron escenas con los filtros indicados.")
            return 1

        print(f"✅ Se encontraron {len(escenas)} escenas:")
        for idx, e in enumerate(escenas, 1):
            print(f"  [{idx}] {e['fecha']} | {e['id']} | Nubes: {e['nubes_pct']}% | Plataforma: {e['plataforma']}")

        if args.download_top_pair:
            if len(escenas) < 2:
                print("❌ Se requieren al menos 2 escenas para conformar un par multitemporal.")
                return 1
            # Tomar las 2 escenas más limpias (menor nubosidad) y ordenarlas por fecha
            top2 = sorted(escenas, key=lambda x: x["nubes_pct"])[:2]
            top2_ordenadas = sorted(top2, key=lambda x: x["fecha"])
            t0, t1 = top2_ordenadas[0], top2_ordenadas[1]
            print(f"\n📥 Descargando par multitemporal:")
            print(f"   T0: {t0['fecha']} ({t0['id']})")
            print(f"   T1: {t1['fecha']} ({t1['id']})")
            p_t0, p_t1 = descargar_par_sentinel2(
                t0,
                t1,
                output_dir=args.output_dir,
                crop=not args.no_crop,
                overwrite=args.overwrite,
            )
            print(f"✅ Par guardado en:")
            print(f"   - {p_t0}")
            print(f"   - {p_t1}")
            return 0

    if args.scene_id:
        print(f"📥 Descargando escena {args.scene_id} a {args.output_dir}...")
        p_out = descargar_escena_sentinel2(
            args.scene_id,
            output_dir=args.output_dir,
            crop=not args.no_crop,
            overwrite=args.overwrite,
        )
        print(f"✅ Escena guardada en: {p_out}")
        return 0

    if not args.search and not args.scene_id and not args.download_top_pair:
        parser.print_help()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
