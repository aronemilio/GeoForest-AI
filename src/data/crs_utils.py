"""Utilidades de validación del sistema de referencia (EPSG:32718) para la Fase A."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from rasterio.crs import CRS

log = logging.getLogger(__name__)

TARGET_EPSG = 32718
TARGET_CRS = CRS.from_epsg(TARGET_EPSG)


def ensure_utm18s_gdf(gdf: gpd.GeoDataFrame, source: str | Path) -> gpd.GeoDataFrame:
    """Devuelve el GeoDataFrame en EPSG:32718.

    - CRS indefinido -> ValueError (no se adivina la proyección).
    - Otro CRS -> reproyección explícita con aviso en el log.
    """
    if gdf.crs is None:
        raise ValueError(f"{source}: CRS indefinido; se exige EPSG:{TARGET_EPSG}")
    if gdf.crs.to_epsg() != TARGET_EPSG:
        log.warning("%s está en %s -> reproyectando a EPSG:%d", source, gdf.crs, TARGET_EPSG)
        gdf = gdf.to_crs(epsg=TARGET_EPSG)
    return gdf


def assert_raster_utm18s(src, path: str | Path) -> None:
    """Lanza ValueError si el ráster abierto no está en EPSG:32718.

    Los rásteres no se reproyectan al vuelo para no alterar la grilla de referencia.
    """
    if src.crs is None or src.crs.to_epsg() != TARGET_EPSG:
        raise ValueError(f"{path}: ráster no está en EPSG:{TARGET_EPSG} (crs={src.crs})")
