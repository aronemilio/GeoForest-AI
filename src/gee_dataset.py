"""Descarga de dataset multitemporal rico Sentinel-2 L2A para Huánuco vía Google Earth Engine (GEE).

Proyecto GEE configurado: ``geoforest-ai-512500``
Colección GEE: ``COPERNICUS/S2_SR_HARMONIZED`` (Reflectancia de Superficie L2A)

Estructura de salida exacta esperada por ``src/dataset.py``::

    data/raw/sentinel2/
    └── YYYY-MM-DD_<SCENE_ID>/
        ├── B04.tif   (10m, uint16, EPSG:32718, 256x256)
        ├── B08.tif   (10m, uint16, EPSG:32718, 256x256)
        └── B11.tif   (remuestreada 20m -> 10m bilineal, uint16, EPSG:32718, 256x256)

Uso desde terminal::

    ./venv/bin/python src/gee_dataset.py --project geoforest-ai-512500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.windows import Window, transform as window_transform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.copernicus import (  # noqa: E402
    BANDAS_REQUERIDAS,
    DEFAULT_CRS,
    HUANUCO_BOUNDS_UTM,
    STAC_API_URL,
    descargar_escena_sentinel2,
    sanitize_scene_id,
    validate_output_directory,
)

logger = logging.getLogger(__name__)

GEE_PROJECT_ID = "geoforest-ai-512500"
GEE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
MGRS_TILE_HUANUCO = "18LUQ"
DEFAULT_HUANUCO_SHAPE: tuple[int, int] = (256, 256)

# GEE usa nombres de banda sin cero a la izquierda (B4, B8); los GeoTIFF
# locales conservan la nomenclatura B04/B08/B11 que consume src/dataset.py.
GEE_BAND_MAP: dict[str, str] = {"B04": "B4", "B08": "B8", "B11": "B11"}
ZONE_DATE_PAIR: tuple[str, str] = ("2024-07-18", "2024-09-11")


@dataclass(frozen=True)
class HuanucoSceneSpec:
    """Especificación de una escena real Sentinel-2 L2A sobre Huánuco (Tile 18LUQ)."""

    scene_id: str
    fecha: str
    cloud_cover: float
    ciclo_estacional: str
    cog_base_url: str


@dataclass(frozen=True)
class HuanucoZone:
    """Zona operativa de Huánuco para el dataset multiescalar de seis bloques.

    Cada zona se representa con un bloque de ``block_km`` × ``block_km`` centrado
    en ``center_lat``/``center_lon``. El bloque se descarga a 10 m y se trocea en
    ``chips_per_side``² parches de 256×256 para el entrenamiento.
    """

    zone_id: str
    nombre: str
    provincias_distritos: str
    ecosistema: str
    altitud_m: str
    dinamica_cambio: str
    center_lat: float
    center_lon: float
    block_km: float = 10.24
    chips_per_side: int = 4
    seleccion: str = ""


# Catálogo multianual y multiestacional (2023-2026) con baja nubosidad (<17%) sobre Huánuco
HUANUCO_RICHDATASET_SCENES: tuple[HuanucoSceneSpec, ...] = (
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20230624_0_L2A",
        fecha="2023-06-24",
        cloud_cover=5.03,
        ciclo_estacional="2023_inicio_estiaje",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2023/6/S2B_18LUQ_20230624_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20230803_0_L2A",
        fecha="2023-08-03",
        cloud_cover=12.12,
        ciclo_estacional="2023_pico_estiaje_quemas",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2023/8/S2B_18LUQ_20230803_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2A_18LUQ_20240524_0_L2A",
        fecha="2024-05-24",
        cloud_cover=16.37,
        ciclo_estacional="2024_transicion_humedo_seco",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2024/5/S2A_18LUQ_20240524_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2A_18LUQ_20240603_0_L2A",
        fecha="2024-06-03",
        cloud_cover=12.13,
        ciclo_estacional="2024_inicio_estiaje",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2024/6/S2A_18LUQ_20240603_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20240718_0_L2A",
        fecha="2024-07-18",
        cloud_cover=1.68,
        ciclo_estacional="2024_estiaje_despejado",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2024/7/S2B_18LUQ_20240718_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2A_18LUQ_20240911_0_L2A",
        fecha="2024-09-11",
        cloud_cover=9.25,
        ciclo_estacional="2024_pico_quemas_tardio",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2024/9/S2A_18LUQ_20240911_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20250603_0_L2A",
        fecha="2025-06-03",
        cloud_cover=9.29,
        ciclo_estacional="2025_inicio_estiaje",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2025/6/S2B_18LUQ_20250603_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2A_18LUQ_20250710_0_L2A",
        fecha="2025-07-10",
        cloud_cover=4.72,
        ciclo_estacional="2025_estiaje_despejado",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2025/7/S2A_18LUQ_20250710_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2A_18LUQ_20250829_0_L2A",
        fecha="2025-08-29",
        cloud_cover=8.92,
        ciclo_estacional="2025_pico_estiaje_quemas",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2025/8/S2A_18LUQ_20250829_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20260628_0_L2A",
        fecha="2026-06-28",
        cloud_cover=7.57,
        ciclo_estacional="2026_inicio_estiaje",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2026/6/S2B_18LUQ_20260628_0_L2A",
    ),
    HuanucoSceneSpec(
        scene_id="S2B_18LUQ_20260807_0_L2A",
        fecha="2026-08-07",
        cloud_cover=13.40,
        ciclo_estacional="2026_pico_estiaje_quemas",
        cog_base_url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2026/8/S2B_18LUQ_20260807_0_L2A",
    ),
)

# Pares bitemporales (T0 -> T1) recomendados para entrenamiento y validación en Huánuco
HUANUCO_TEMPORAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("2023-06-24_S2B_18LUQ_20230624_0_L2A", "2023-08-03_S2B_18LUQ_20230803_0_L2A"),
    ("2024-05-24_S2A_18LUQ_20240524_0_L2A", "2024-07-18_S2B_18LUQ_20240718_0_L2A"),
    ("2024-06-03_S2A_18LUQ_20240603_0_L2A", "2024-09-11_S2A_18LUQ_20240911_0_L2A"),
    ("2024-07-18_S2B_18LUQ_20240718_0_L2A", "2024-09-11_S2A_18LUQ_20240911_0_L2A"),
    ("2025-06-03_S2B_18LUQ_20250603_0_L2A", "2025-07-10_S2A_18LUQ_20250710_0_L2A"),
    ("2025-07-10_S2A_18LUQ_20250710_0_L2A", "2025-08-29_S2A_18LUQ_20250829_0_L2A"),
    ("2026-06-28_S2B_18LUQ_20260628_0_L2A", "2026-08-07_S2B_18LUQ_20260807_0_L2A"),
)

# Seis zonas operativas del departamento de Huánuco. Los centros se eligieron
# con datos reales: geocodificación inversa (OSM/Nominatim) para verificar el
# distrito/provincia y una cuadrícula de bloques de 10.24 km evaluada con
# Hansen Global Forest Change 2025 (pérdida 2020-2024, cobertura arbórea 2000),
# SRTM (altitud) y ESA WorldCover 2021 (cobertura del suelo). No son polígonos
# inventados: cada centro cae en el distrito/provincia indicado.
HUANUCO_ZONES: tuple[HuanucoZone, ...] = (
    HuanucoZone(
        zone_id="01_carpish_chinchao",
        nombre="Carpish / Chinchao",
        provincias_distritos="Huánuco (Chinchao / Cordillera de Carpish)",
        ecosistema="Bosque montano nuboso / Ceja de selva",
        altitud_m="1800-2800 m",
        dinamica_cambio=(
            "Tala selectiva en laderas empinadas, apertura de trochas y rozas pequeñas."
        ),
        center_lat=-9.52219,
        center_lon=-76.08185,
        seleccion=(
            "Bloque de 10.24 km en Chinchao: 85.9% cobertura arbórea Hansen 2000, "
            "altitud media 1827 m y 276 píxeles de pérdida 2022-2024; prioriza "
            "bosque nuboso de Carpish."
        ),
    ),
    HuanucoZone(
        zone_id="02_leoncio_prado_tingo",
        nombre="Leoncio Prado / Tingo María",
        provincias_distritos="Leoncio Prado (Rupa-Rupa, Castillo Grande, Monzón)",
        ecosistema="Selva alta húmeda / Valle del Alto Huallaga",
        altitud_m="600-1500 m",
        dinamica_cambio=(
            "Deforestación agrícola fragmentada en parches irregulares y laderas."
        ),
        center_lat=-9.10676,
        center_lon=-76.03777,
        seleccion=(
            "Bloque de 10.24 km en Pueblo Nuevo (Leoncio Prado): 85.1% cobertura "
            "arbórea, altitud media 642 m y 531 píxeles de pérdida 2022-2024; "
            "captura deforestación agrícola del Alto Huallaga."
        ),
    ),
    HuanucoZone(
        zone_id="03_puerto_inca_pachitea",
        nombre="Puerto Inca / Pachitea",
        provincias_distritos="Puerto Inca (Yuyapichis, Codo del Pozuzo Este, Constitución)",
        ecosistema="Selva baja amazónica / Cuenca del Pachitea",
        altitud_m="200-500 m",
        dinamica_cambio=(
            "Tala masiva para pastizales ganaderos y apertura de caminos madereros lineales."
        ),
        center_lat=-9.44218,
        center_lon=-75.16081,
        seleccion=(
            "Bloque de 10.24 km en Puerto Inca: 98.3% cobertura arbórea, altitud "
            "media 261 m y 929 píxeles de pérdida 2022-2024; es el bloque con mayor "
            "dinámica de tala y caminos."
        ),
    ),
    HuanucoZone(
        zone_id="04_codo_del_pozuzo",
        nombre="Codo del Pozuzo",
        provincias_distritos="Puerto Inca / Límite con Oxapampa (Codo del Pozuzo)",
        ecosistema="Llanura aluvial y bosque húmedo premontano",
        altitud_m="450-1200 m",
        dinamica_cambio=(
            "Desbosque en bloques geométricos grandes y conversión agropecuaria definida."
        ),
        center_lat=-9.64684,
        center_lon=-75.48312,
        seleccion=(
            "Bloque de 10.24 km en Codo del Pozuzo: 83.9% cobertura arbórea, "
            "altitud media 479 m y 436 píxeles de pérdida 2022-2024; representa "
            "desbosque en bloques de la llanura aluvial."
        ),
    ),
    HuanucoZone(
        zone_id="05_maranon_monzon_oeste",
        nombre="Marañón / Monzón oeste",
        provincias_distritos="Huamalíes / Huacaybamba (Margen oriental del Marañón)",
        ecosistema="Bosque seco interandino y pajonal/matorral",
        altitud_m="1500-3000 m",
        dinamica_cambio=(
            "Cicatrices extensas de incendios forestales estacionales y quema de pastizales."
        ),
        center_lat=-9.19298,
        center_lon=-76.38423,
        seleccion=(
            "Bloque de 10.24 km en Cochabamba (Huacaybamba): 86.9% cobertura "
            "arbórea, altitud media 1574 m y 295 píxeles de pérdida 2022-2024; "
            "cubre el piso interandino seco y las cicatrices de quema."
        ),
    ),
    HuanucoZone(
        zone_id="06_huanuco_amarilis_molino",
        nombre="Huánuco / Amarilis / Molino",
        provincias_distritos="Huánuco / Pachitea (Amarilis, Churubamba, Molino, Umari)",
        ecosistema="Valles secos interandinos y plantaciones",
        altitud_m="1900-2600 m",
        dinamica_cambio=(
            "Quemas agrícolas, tala de eucalipto/pino y contraste con suelo árido "
            "para reducir falsos positivos."
        ),
        center_lat=-9.82614,
        center_lon=-75.99365,
        seleccion=(
            "Bloque de 10.24 km en Umari (Pachitea): solo 6.3% cobertura arbórea, "
            "12% cultivo y 74% pastizal (WorldCover), altitud media 2555 m y pérdida "
            "muy baja (3 píxeles); se usa como contraste seco/agrícola y negativo, "
            "no como zona de tala boscosa."
        ),
    ),
)


def init_gee(project_id: str = GEE_PROJECT_ID) -> bool:
    """Inicializa Google Earth Engine con el proyecto ``geoforest-ai-512500``.

    Returns:
        ``True`` si la sesión GEE quedó autenticada e inicializada, ``False`` si no
        hay credenciales OAuth locales activas en el entorno de ejecución actual.
    """
    try:
        import ee

        ee.Initialize(project=project_id)
        return True
    except Exception as exc:
        logger.info(
            "GEE OAuth no disponible directamente en este entorno (%s); "
            "se usará descarga directa del archivo espejo Sentinel-2 L2A.",
            exc,
        )
        return False


def _mask_s2_clouds_gee(image: Any) -> Any:
    """Enmascara nubes opacas y cirros usando la banda QA60 de Sentinel-2 en GEE."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).copyProperties(image, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"])


def descargar_escena_gee(
    spec: HuanucoSceneSpec,
    sentinel2_root: str | Path = "data/raw/sentinel2",
    project_id: str = GEE_PROJECT_ID,
    utm_bounds: tuple[float, float, float, float] = HUANUCO_BOUNDS_UTM,
    out_shape: tuple[int, int] = DEFAULT_HUANUCO_SHAPE,
    crs: str = DEFAULT_CRS,
) -> Path:
    """Descarga una escena de ``COPERNICUS/S2_SR_HARMONIZED`` desde GEE hacia ``data/raw/sentinel2``.

    Recorta exactamente a ``utm_bounds`` en ``EPSG:32718`` con grilla ``256x256`` (10 m/px),
    remuestreando ``B11`` (20 m) con interpolación bilineal para coincidir con ``B04`` y ``B08``.
    """
    import ee

    ee.Initialize(project=project_id)

    safe_id = sanitize_scene_id(spec.scene_id)
    folder_name = f"{spec.fecha}_{safe_id}" if not safe_id.startswith(spec.fecha) else safe_id
    valid_root = validate_output_directory(sentinel2_root)
    out_dir = valid_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    west, south, east, north = utm_bounds
    region = ee.Geometry.Rectangle([west, south, east, north], proj=crs, geodesic=False)

    start_date = spec.fecha
    end_date = ee.Date(start_date).advance(1, "day")

    collection = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("MGRS_TILE", MGRS_TILE_HUANUCO))
        .map(_mask_s2_clouds_gee)
    )

    image = ee.Image(collection.first())
    height, width = out_shape

    for banda in BANDAS_REQUERIDAS:
        gee_banda = GEE_BAND_MAP[banda]
        band_img = image.select(gee_banda)
        if banda == "B11":
            band_img = band_img.resample("bilinear")
        band_img = band_img.toUint16()

        url = band_img.getDownloadURL(
            {
                "name": banda,
                "bands": [gee_banda],
                "crs": crs,
                "crs_transform": [10.0, 0.0, west, 0.0, -10.0, north],
                "dimensions": f"{width}x{height}",
                "format": "GEO_TIFF",
            }
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        out_file = out_dir / f"{banda}.tif"

        content = resp.content
        if content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                if not tif_names:
                    raise RuntimeError(f"GEE devolvió ZIP sin GeoTIFF para {banda}")
                out_file.write_bytes(zf.read(tif_names[0]))
        else:
            out_file.write_bytes(content)

    return out_dir


def verify_scene_geotiffs(
    scene_dir: Path,
    expected_shape: tuple[int, int] = DEFAULT_HUANUCO_SHAPE,
    expected_crs: str = DEFAULT_CRS,
) -> dict[str, Any]:
    """Verifica que una carpeta de escena contenga B04.tif, B08.tif y B11.tif válidos."""
    stats: dict[str, Any] = {"scene": scene_dir.name, "bands": {}}
    for banda in BANDAS_REQUERIDAS:
        tif_path = scene_dir / f"{banda}.tif"
        if not tif_path.is_file():
            raise FileNotFoundError(f"Falta archivo {tif_path}")
        with rasterio.open(tif_path) as ds:
            arr = ds.read(1)
            if ds.shape != expected_shape:
                raise ValueError(f"{tif_path}: shape {ds.shape} != {expected_shape}")
            if str(ds.crs) != expected_crs:
                raise ValueError(f"{tif_path}: CRS {ds.crs} != {expected_crs}")
            if arr.dtype != np.uint16:
                raise ValueError(f"{tif_path}: dtype {arr.dtype} != uint16")
            stats["bands"][banda] = {
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": round(float(arr.mean()), 1),
            }
    return stats


def zone_utm_bounds(zone: HuanucoZone, crs: str = DEFAULT_CRS) -> tuple[float, float, float, float]:
    """Calcula el rectángulo UTM de ``zone.block_km`` centrado en la zona.

    La conversión es explícita WGS84 -> UTM (EPSG:32718) y se usa tanto para el
    recorte GEE como para la geotransformación de cada chip.
    """
    from pyproj import Transformer

    if zone.block_km <= 0:
        raise ValueError(f"{zone.zone_id}: block_km debe ser positivo.")
    if zone.chips_per_side <= 0:
        raise ValueError(f"{zone.zone_id}: chips_per_side debe ser positivo.")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(zone.center_lon, zone.center_lat)
    half = zone.block_km * 1000.0 / 2.0
    return (x - half, y - half, x + half, y + half)


def _download_gee_band_array(
    image: Any,
    file_band: str,
    utm_bounds: tuple[float, float, float, float],
    size_px: int,
    crs: str = DEFAULT_CRS,
    timeout: float = 180.0,
) -> tuple[np.ndarray, Any]:
    """Descarga una banda GEE como arreglo uint16 y devuelve su geotransformación.

    GEE usa ``B4``/``B8``/``B11``; el nombre de archivo local se mantiene como
    ``B04``/``B08``/``B11`` para que ``src/dataset.py`` no cambie.
    """
    gee_band = GEE_BAND_MAP[file_band]
    band_img = image.select(gee_band)
    if file_band == "B11":
        band_img = band_img.resample("bilinear")
    band_img = band_img.toUint16()

    west, _south, _east, north = utm_bounds
    url = band_img.getDownloadURL(
        {
            "name": file_band,
            "bands": [gee_band],
            "crs": crs,
            "crs_transform": [10.0, 0.0, west, 0.0, -10.0, north],
            "dimensions": f"{size_px}x{size_px}",
            "format": "GEO_TIFF",
        }
    )

    last_error: Exception | None = None
    content: bytes | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content
            break
        except Exception as exc:  # pragma: no cover - depende de la red/GEE
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    if content is None:
        raise RuntimeError(f"GEE no devolvió la banda {file_band}: {last_error}")
    if not content:
        raise RuntimeError(f"GEE devolvió una respuesta vacía para {file_band}")

    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_names:
                raise RuntimeError(f"GEE devolvió ZIP sin GeoTIFF para {file_band}")
            content = zf.read(tif_names[0])

    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as ds:
            if ds.shape != (size_px, size_px):
                raise ValueError(
                    f"{file_band}: shape GEE {ds.shape} != {(size_px, size_px)}"
                )
            if str(ds.crs) != crs:
                raise ValueError(f"{file_band}: CRS GEE {ds.crs} != {crs}")
            if ds.count != 1:
                raise ValueError(f"{file_band}: se esperaba 1 banda, llegaron {ds.count}")
            arr = ds.read(1)
            if arr.dtype != np.uint16:
                raise ValueError(f"{file_band}: dtype GEE {arr.dtype} != uint16")
            return np.ascontiguousarray(arr), ds.transform


def _select_gee_scene(
    region: Any,
    fecha: str,
    max_cloud: float = 40.0,
) -> tuple[Any, dict[str, Any]]:
    """Selecciona la escena menos nubosa que cubra al menos el 98 % de ``region``.

    Si ninguna escena alcanza esa cobertura, devuelve la menos nubosa y deja el
    registro de cobertura en el catálogo para que el problema sea visible.
    """
    import ee

    start = ee.Date(fecha)
    end = start.advance(1, "day")
    collection = (
        ee.ImageCollection(GEE_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    n = int(collection.size().getInfo())
    if n == 0:
        raise RuntimeError(
            f"No hay escenas {GEE_COLLECTION} para {fecha} con nubosidad < {max_cloud}% "
            f"en la región solicitada."
        )

    candidates = collection.limit(10).getInfo()["features"]
    best: tuple[Any, dict[str, Any]] | None = None
    for feat in candidates:
        image = ee.Image(feat["id"])
        try:
            coverage = (
                image.select(GEE_BAND_MAP["B04"])
                .mask()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=100,
                    tileScale=4,
                    maxPixels=1_000_000_000,
                )
                .getInfo()
                .get(GEE_BAND_MAP["B04"])
            )
        except Exception:
            coverage = None
        props = dict(feat.get("properties", {}))
        props["coverage"] = coverage
        if coverage is not None and float(coverage) >= 0.98:
            return image, props
        if best is None:
            best = (image, props)

    if best is None:
        raise RuntimeError(f"No se pudo materializar ninguna escena para {fecha}.")
    logger.warning(
        "Ninguna escena de %s cubre >=98%% de la región; se usa la de menor nubosidad "
        "(cobertura=%s).",
        fecha,
        best[1].get("coverage"),
    )
    return best


def _write_chip(path: Path, array: np.ndarray, transform: Any, crs: str, chip_px: int) -> None:
    """Escribe un chip uint16 de forma atómica (archivo temporal + replace)."""
    profile = {
        "driver": "GTiff",
        "height": chip_px,
        "width": chip_px,
        "count": 1,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 2,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(np.ascontiguousarray(array), 1)
    tmp_path.replace(path)


def _zone_catalog_payload(
    project_id: str,
    crs: str,
    zone_results: list[dict[str, Any]],
    temporal_pairs: list[list[str]],
    fechas: tuple[str, str],
    chip_px: int,
) -> dict[str, Any]:
    total_scenes = sum(
        len(names)
        for zone in zone_results
        for names in zone.get("chips_by_date", {}).values()
    )
    return {
        "gee_project_id": project_id,
        "gee_collection": GEE_COLLECTION,
        "crs": crs,
        "fechas": list(fechas),
        "chip_px": chip_px,
        "selection_method": (
            "Bloques de 10.24 km (1024 px a 10 m) centrados en seis zonas de Huánuco; "
            "centros elegidos con geocodificación inversa OSM/Nominatim y cuadrícula "
            "evaluada con Hansen GFC 2025 (pérdida 2020-2024), SRTM y ESA WorldCover 2021. "
            "Escenas Sentinel-2 L2A seleccionadas por nubosidad y cobertura >=98%."
        ),
        "total_zones": len(zone_results),
        "total_scenes": total_scenes,
        "total_pairs": len(temporal_pairs),
        "temporal_pairs": temporal_pairs,
        "zones": zone_results,
    }


def download_zone_dataset(
    sentinel2_root: str | Path = "data/raw/sentinel2",
    project_id: str = GEE_PROJECT_ID,
    zones: tuple[HuanucoZone, ...] = HUANUCO_ZONES,
    fechas: tuple[str, str] = ZONE_DATE_PAIR,
    overwrite: bool = False,
    crs: str = DEFAULT_CRS,
    chip_px: int = 256,
    max_cloud: float = 40.0,
) -> dict[str, Any]:
    """Descarga las seis zonas de Huánuco y las trocea en chips de 256×256.

    Por cada zona se descarga un bloque de ``block_km`` a 10 m y se escribe una
    carpeta por chip y por fecha. El catálogo resultante
    (``gee_huanuco_zonas_catalog.json``) contiene pares temporales explícitos
    únicamente entre el T0 y el T1 del mismo chip, evitando cruces entre zonas.
    """
    if len(fechas) != 2:
        raise ValueError("fechas debe contener exactamente (T0, T1).")
    if chip_px <= 0:
        raise ValueError("chip_px debe ser positivo.")
    if not init_gee(project_id=project_id):
        raise RuntimeError(
            "GEE no está autenticado. Ejecuta `python -c \"import ee; ee.Authenticate()\"` "
            f"y vuelve a intentar con el proyecto {project_id}."
        )

    import ee

    root = Path(sentinel2_root)
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "gee_huanuco_zonas_catalog.json"
    zone_results: list[dict[str, Any]] = []
    temporal_pairs: list[list[str]] = []
    total_cached = 0
    total_downloaded = 0

    print(f"🛰️  GEE project: {project_id} | colección: {GEE_COLLECTION}", flush=True)
    print(f"📂 Destino: {root.resolve()}", flush=True)
    print(f"🗓️  Fechas: {fechas[0]} -> {fechas[1]} | CRS: {crs} | chip: {chip_px}px", flush=True)
    print("-" * 78, flush=True)

    for zone_idx, zone in enumerate(zones, start=1):
        bounds = zone_utm_bounds(zone, crs=crs)
        west, south, east, north = bounds
        region = ee.Geometry.Rectangle([west, south, east, north], proj=crs, geodesic=False)
        size_px = int(round(zone.block_km * 1000.0 / 10.0))
        expected_px = zone.chips_per_side * chip_px
        if size_px != expected_px:
            raise ValueError(
                f"{zone.zone_id}: block_km={zone.block_km} produce {size_px}px, "
                f"pero chips_per_side*chip_px={expected_px}px."
            )

        print(
            f"[{zone_idx:02d}/{len(zones):02d}] {zone.zone_id} | {zone.nombre} | "
            f"centro=({zone.center_lat:.5f}, {zone.center_lon:.5f}) | "
            f"{zone.block_km:.2f} km | {zone.chips_per_side}x{zone.chips_per_side} chips",
            flush=True,
        )

        chip_names_by_date: dict[str, list[str]] = {}
        scene_meta: dict[str, dict[str, Any]] = {}

        for fecha in fechas:
            image, props = _select_gee_scene(region, fecha, max_cloud=max_cloud)
            scene_id = sanitize_scene_id(
                str(props.get("system:index") or props.get("PRODUCT_ID") or "")
            )
            if not scene_id:
                raise RuntimeError(f"{zone.zone_id}/{fecha}: escena sin identificador.")
            cloud = float(props.get("CLOUDY_PIXEL_PERCENTAGE", -1.0))
            tile = str(props.get("MGRS_TILE", ""))
            coverage = props.get("coverage")

            names = [
                f"{fecha}_{zone.zone_id}_p{r}{c}_{scene_id}"
                for r in range(zone.chips_per_side)
                for c in range(zone.chips_per_side)
            ]
            chip_names_by_date[fecha] = names
            complete = all(
                all((root / name / f"{banda}.tif").is_file() for banda in BANDAS_REQUERIDAS)
                for name in names
            )

            if complete and not overwrite:
                for name in names:
                    verify_scene_geotiffs(root / name, expected_shape=(chip_px, chip_px), expected_crs=crs)
                total_cached += len(names)
                status = "cached"
                print(
                    f"  ✅ {fecha} {scene_id} ({tile}, nubes={cloud:.2f}%, "
                    f"cobertura={coverage}): {len(names)} chips ya existentes",
                    flush=True,
                )
            else:
                arrays: dict[str, np.ndarray] = {}
                transforms: dict[str, Any] = {}
                for banda in BANDAS_REQUERIDAS:
                    arr, transform = _download_gee_band_array(
                        image=image,
                        file_band=banda,
                        utm_bounds=bounds,
                        size_px=size_px,
                        crs=crs,
                    )
                    arrays[banda] = arr
                    transforms[banda] = transform

                for r in range(zone.chips_per_side):
                    for c in range(zone.chips_per_side):
                        idx = r * zone.chips_per_side + c
                        out_dir = root / names[idx]
                        out_dir.mkdir(parents=True, exist_ok=True)
                        for banda in BANDAS_REQUERIDAS:
                            chip_arr = arrays[banda][
                                r * chip_px : (r + 1) * chip_px,
                                c * chip_px : (c + 1) * chip_px,
                            ]
                            if chip_arr.shape != (chip_px, chip_px):
                                raise ValueError(
                                    f"{zone.zone_id}/{fecha}/{banda}: chip {chip_arr.shape} "
                                    f"!= {(chip_px, chip_px)}"
                                )
                            chip_transform = window_transform(
                                Window(c * chip_px, r * chip_px, chip_px, chip_px),
                                transforms[banda],
                            )
                            _write_chip(
                                out_dir / f"{banda}.tif",
                                chip_arr,
                                chip_transform,
                                crs,
                                chip_px,
                            )
                        verify_scene_geotiffs(
                            out_dir,
                            expected_shape=(chip_px, chip_px),
                            expected_crs=crs,
                        )
                total_downloaded += len(names)
                status = "downloaded"
                print(
                    f"  ⬇️  {fecha} {scene_id} ({tile}, nubes={cloud:.2f}%, "
                    f"cobertura={coverage}): {len(names)} chips descargados",
                    flush=True,
                )

            scene_meta[fecha] = {
                "scene_id": scene_id,
                "fecha": fecha,
                "mgrs_tile": tile,
                "cloud_cover": cloud,
                "coverage": coverage,
                "status": status,
                "bounds_utm": [round(v, 3) for v in bounds],
                "chips": names,
            }

        t0, t1 = fechas
        zone_pairs: list[list[str]] = []
        for idx in range(zone.chips_per_side * zone.chips_per_side):
            pair = [chip_names_by_date[t0][idx], chip_names_by_date[t1][idx]]
            zone_pairs.append(pair)
            temporal_pairs.append(pair)

        zone_results.append(
            {
                "zone_id": zone.zone_id,
                "nombre": zone.nombre,
                "provincias_distritos": zone.provincias_distritos,
                "ecosistema": zone.ecosistema,
                "altitud_m": zone.altitud_m,
                "dinamica_cambio": zone.dinamica_cambio,
                "seleccion": zone.seleccion,
                "center_lat": zone.center_lat,
                "center_lon": zone.center_lon,
                "block_km": zone.block_km,
                "chips_per_side": zone.chips_per_side,
                "chip_px": chip_px,
                "bounds_utm": [round(v, 3) for v in bounds],
                "scene_t0": scene_meta[t0],
                "scene_t1": scene_meta[t1],
                "chips_by_date": chip_names_by_date,
                "temporal_pairs": zone_pairs,
            }
        )

        catalog_path.write_text(
            json.dumps(
                _zone_catalog_payload(
                    project_id,
                    crs,
                    zone_results,
                    temporal_pairs,
                    fechas,
                    chip_px,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"  📄 Catálogo parcial: {len(zone_results)} zonas, "
            f"{len(temporal_pairs)} pares",
            flush=True,
        )

    payload = _zone_catalog_payload(
        project_id,
        crs,
        zone_results,
        temporal_pairs,
        fechas,
        chip_px,
    )
    catalog_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 78, flush=True)
    print(
        f"🎉 Zonas Huánuco completadas: {len(zone_results)} zonas, "
        f"{payload['total_scenes']} chips (T0+T1), {payload['total_pairs']} pares. "
        f"Descargados={total_downloaded}, cacheados={total_cached}.",
        flush=True,
    )
    print(f"📄 Catálogo: {catalog_path}", flush=True)
    return payload


def download_rich_huanuco_dataset(
    sentinel2_root: str | Path = "data/raw/sentinel2",
    project_id: str = GEE_PROJECT_ID,
    scenes: tuple[HuanucoSceneSpec, ...] = HUANUCO_RICHDATASET_SCENES,
    overwrite: bool = False,
    utm_bounds: tuple[float, float, float, float] = HUANUCO_BOUNDS_UTM,
    out_shape: tuple[int, int] = DEFAULT_HUANUCO_SHAPE,
    crs: str = DEFAULT_CRS,
) -> list[dict[str, Any]]:
    """Descarga y verifica el catálogo completo de escenas Sentinel-2 de Huánuco.

    Intenta primero la descarga vía API de Google Earth Engine con ``project_id``
    (``geoforest-ai-512500``); si el entorno actual no dispone del token OAuth
    de GEE (p. ej. contenedor/sandbox), descarga exactamente las mismas escenas
    L2A desde el archivo COG público de Sentinel-2.
    """
    root = Path(sentinel2_root)
    root.mkdir(parents=True, exist_ok=True)

    gee_ready = init_gee(project_id=project_id)
    backend_label = f"Google Earth Engine (project={project_id})" if gee_ready else (
        f"Google Earth Engine spec (project={project_id}) -> Mirror COG Sentinel-2 L2A"
    )
    print(f"🛰️  Backend activo: {backend_label}", flush=True)
    print(f"📂 Destino: {root.resolve()}", flush=True)
    print(f"🌎 Región: Huánuco / Carpish - Huallaga | Tile {MGRS_TILE_HUANUCO} | {crs} | Bounds {utm_bounds}", flush=True)
    print("-" * 78, flush=True)

    results: list[dict[str, Any]] = []
    for idx, spec in enumerate(scenes, start=1):
        folder_name = f"{spec.fecha}_{spec.scene_id}"
        target_dir = root / folder_name

        already_complete = (
            target_dir.is_dir()
            and not overwrite
            and all((target_dir / f"{b}.tif").is_file() for b in BANDAS_REQUERIDAS)
        )

        if already_complete:
            info = verify_scene_geotiffs(target_dir, expected_shape=out_shape, expected_crs=crs)
            info["status"] = "cached"
            info["cloud_cover"] = spec.cloud_cover
            info["ciclo"] = spec.ciclo_estacional
            results.append(info)
            b08_mean = info["bands"]["B08"]["mean"]
            print(
                f"[{idx:02d}/{len(scenes):02d}] ✅ Existente: {folder_name} "
                f"(nubes={spec.cloud_cover:.2f}%, ciclo={spec.ciclo_estacional}, B08_mean={b08_mean})",
                flush=True,
            )
            continue

        print(
            f"[{idx:02d}/{len(scenes):02d}] ⬇️  Descargando {folder_name} "
            f"(nubes={spec.cloud_cover:.2f}%, ciclo={spec.ciclo_estacional})...",
            flush=True,
        )

        scene_dir: Path | None = None
        if gee_ready:
            try:
                scene_dir = descargar_escena_gee(
                    spec=spec,
                    sentinel2_root=root,
                    project_id=project_id,
                    utm_bounds=utm_bounds,
                    out_shape=out_shape,
                    crs=crs,
                )
            except Exception as exc:
                logger.warning("Fallo descarga directa GEE para %s (%s); usando COG L2A.", spec.scene_id, exc)
                scene_dir = None

        if scene_dir is None:
            escena_dict = {
                "id": spec.scene_id,
                "fecha": spec.fecha,
                "url_b04": f"{spec.cog_base_url}/B04.tif",
                "url_b08": f"{spec.cog_base_url}/B08.tif",
                "url_b11": f"{spec.cog_base_url}/B11.tif",
            }
            scene_dir = descargar_escena_sentinel2(
                escena=escena_dict,
                output_dir=root,
                crop=True,
                crop_bounds=utm_bounds,
                target_shape=out_shape,
                overwrite=overwrite,
            )

        info = verify_scene_geotiffs(scene_dir, expected_shape=out_shape, expected_crs=crs)
        info["status"] = "downloaded"
        info["cloud_cover"] = spec.cloud_cover
        info["ciclo"] = spec.ciclo_estacional
        results.append(info)
        b04_m = info["bands"]["B04"]["mean"]
        b08_m = info["bands"]["B08"]["mean"]
        b11_m = info["bands"]["B11"]["mean"]
        print(
            f"       ↳ Guardado en {scene_dir.name}/ (B04={b04_m}, B08={b08_m}, B11={b11_m})",
            flush=True,
        )

    manifest_path = root / "gee_huanuco_catalog.json"
    manifest_payload = {
        "gee_project_id": project_id,
        "gee_collection": GEE_COLLECTION,
        "mgrs_tile": MGRS_TILE_HUANUCO,
        "crs": crs,
        "utm_bounds": list(utm_bounds),
        "shape": list(out_shape),
        "total_scenes": len(results),
        "temporal_pairs": [list(p) for p in HUANUCO_TEMPORAL_PAIRS],
        "scenes": results,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 78, flush=True)
    print(
        f"🎉 Dataset Huánuco completado: {len(results)} escenas ({len(results) * 3} archivos GeoTIFF) "
        f"y {len(HUANUCO_TEMPORAL_PAIRS)} pares multitemporales listos.",
        flush=True,
    )
    print(f"📄 Catálogo registrado en: {manifest_path}", flush=True)
    return results


def query_stac_huanuco_scenes(
    datetime_range: str = "2023-01-01T00:00:00Z/2026-12-31T23:59:59Z",
    max_cloud: float = 20.0,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Consulta el catálogo STAC de Sentinel-2 L2A para descubrir escenas adicionales en Huánuco."""
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [-76.25, -9.95, -75.95, -9.45],
        "datetime": datetime_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
    }
    resp = requests.post(STAC_API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    scenes = []
    for feat in features:
        props = feat.get("properties", {})
        assets = feat.get("assets", {})
        red_href = assets.get("red", {}).get("href", "")
        if not red_href:
            continue
        scenes.append(
            {
                "scene_id": feat["id"],
                "fecha": props.get("datetime", "")[:10],
                "cloud_cover": float(props.get("eo:cloud_cover", 0.0)),
                "cog_base_url": red_href.rsplit("/", 1)[0],
            }
        )
    return sorted(scenes, key=lambda x: x["fecha"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Descarga un dataset rico de escenas Sentinel-2 L2A de Huánuco (GEE project: geoforest-ai-512500)."
    )
    parser.add_argument(
        "--project",
        default=GEE_PROJECT_ID,
        help=f"ID del proyecto Google Earth Engine (defecto: {GEE_PROJECT_ID}).",
    )
    parser.add_argument(
        "--sentinel2-root",
        default="data/raw/sentinel2",
        help="Directorio raíz donde se guardan las escenas YYYY-MM-DD_S2.../B04.tif, B08.tif, B11.tif.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-descarga las escenas aunque ya existan en disco.",
    )
    parser.add_argument(
        "--zones",
        action="store_true",
        help=(
            "Descarga las seis zonas operativas de Huánuco (bloques de 10.24 km "
            "trocados en 4x4 chips de 256x256) y genera "
            "gee_huanuco_zonas_catalog.json."
        ),
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=40.0,
        help="Nubosidad máxima (%) para seleccionar escenas de cada zona (defecto: 40).",
    )
    args = parser.parse_args()

    if args.zones:
        download_zone_dataset(
            sentinel2_root=args.sentinel2_root,
            project_id=args.project,
            overwrite=args.overwrite,
            max_cloud=args.max_cloud,
        )
    else:
        download_rich_huanuco_dataset(
            sentinel2_root=args.sentinel2_root,
            project_id=args.project,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
