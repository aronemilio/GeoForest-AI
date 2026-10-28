"""Rasterización del Ground Truth vectorial sobre la grilla Sentinel-2 L2A (Tile 18LUQ).

Salida: GeoTIFF uint8 (0 = sin cambio, 1 = deforestación/quema) con el transform, CRS
y dimensiones idénticos al ráster plantilla (banda B04 a 10 m). No se aplica remuestreo.

Uso:
    python -m src.data.rasterize --pairs data/raw/pairs.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features

from src.data.crs_utils import assert_raster_utm18s, ensure_utm18s_gdf

log = logging.getLogger(__name__)

TILE_ID = "18LUQ"
VECTOR_SUFFIXES = {".geojson", ".json", ".gpkg"}
DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_MIN_AREA_M2 = 500.0


# ----------------------------------------------------------------------------- utilidades
def load_pairs(pairs_json: str | Path) -> list[dict[str, str]]:
    """Lee pares temporales: [{"t0": "YYYYMMDD", "t1": "YYYYMMDD"}, ...]."""
    with open(pairs_json, encoding="utf-8") as f:
        pairs = json.load(f)
    for p in pairs:
        if "t0" not in p or "t1" not in p:
            raise ValueError(f"Par inválido en {pairs_json}: {p}")
    return pairs


def find_band(date_dir: str | Path, band: str) -> Path:
    """Localiza el GeoTIFF de una banda (p. ej. 'B04') dentro de la carpeta de una fecha."""
    date_dir = Path(date_dir)
    hits = sorted(p for p in date_dir.glob(f"*{band}*") if p.suffix.lower() in {".tif", ".tiff"})
    if not hits:
        raise FileNotFoundError(f"No se encontró la banda {band} en {date_dir}")
    if len(hits) > 1:
        # Preferir explícitamente la versión de 10 m si existe
        ten_m = [p for p in hits if "10m" in p.name]
        if len(ten_m) == 1:
            return ten_m[0]
        raise ValueError(f"Varias coincidencias para {band} en {date_dir}: {[h.name for h in hits]}")
    return hits[0]


def list_vector_files(gt_dir: str | Path) -> list[Path]:
    gt_dir = Path(gt_dir)
    return sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in VECTOR_SUFFIXES)


def _filter_by_window(gdf: gpd.GeoDataFrame, date_t0: str, date_t1: str) -> gpd.GeoDataFrame:
    """Conserva polígonos cuyo `fecha_t1` (fecha post-evento) cae en (t0, t1] del par."""
    if "fecha_t1" not in gdf.columns:
        log.warning("Ground Truth sin columna 'fecha_t1': no se filtra por ventana temporal")
        return gdf
    t0 = pd.to_datetime(date_t0)
    t1 = pd.to_datetime(date_t1)
    ev = pd.to_datetime(gdf["fecha_t1"], errors="coerce")
    return gdf[(ev > t0) & (ev <= t1)]


def load_ground_truth(
    vector_paths: Sequence[str | Path],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Carga, valida (EPSG:32718), repara y filtra las geometrías del Ground Truth."""
    frames = [ensure_utm18s_gdf(gpd.read_file(p), p) for p in vector_paths]
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:32718")
    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:32718")

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.set_geometry(gdf.geometry.make_valid())
    gdf = gdf.explode(index_parts=False).explode(index_parts=False)
    # make_valid puede producir GeometryCollection -> explode las separa; solo se rasterizan polígonos
    gdf = gdf[gdf.geom_type == "Polygon"]
    gdf = gdf[gdf.area >= min_area_m2]
    if "confianza_validacion" in gdf.columns:
        conf = pd.to_numeric(gdf["confianza_validacion"], errors="coerce")
        gdf = gdf[conf >= min_confidence]
    return gdf


# ----------------------------------------------------------------------------- núcleo
def rasterize_geometries(geoms: Iterable, template_raster: str | Path, out_path: str | Path) -> Path:
    """Rasteriza geometrías (ya en EPSG:32718) sobre la grilla exacta del ráster plantilla."""
    template_raster, out_path = Path(template_raster), Path(out_path)
    with rasterio.open(template_raster) as tpl:
        assert_raster_utm18s(tpl, template_raster)
        profile = tpl.profile.copy()
        transform, shape = tpl.transform, (tpl.height, tpl.width)

    shapes = [(g, 1) for g in geoms]
    if shapes:
        mask = features.rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False,
        )
    else:
        import numpy as np

        mask = np.zeros(shape, dtype="uint8")

    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=None, compress="deflate")
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    log.info("Máscara escrita: %s (%d píxeles positivos)", out_path, int(mask.sum()))
    return out_path


def rasterize_ground_truth(
    vector_paths: Sequence[str | Path],
    template_raster: str | Path,
    out_path: str | Path,
    date_t0: str | None = None,
    date_t1: str | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> Path:
    gdf = load_ground_truth(vector_paths, min_confidence=min_confidence, min_area_m2=min_area_m2)
    if date_t0 and date_t1:
        gdf = _filter_by_window(gdf, date_t0, date_t1)
    return rasterize_geometries(gdf.geometry, template_raster, out_path)


def mask_name(t0: str, t1: str) -> str:
    return f"mask_{TILE_ID}_{t0}_{t1}.tif"


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/raw/pairs.json")
    ap.add_argument("--s2-dir", default="data/raw/sentinel2")
    ap.add_argument("--gt-dir", default="data/raw/ground_truth")
    ap.add_argument("--out-dir", default="data/processed/masks")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    ap.add_argument("--min-area-m2", type=float, default=DEFAULT_MIN_AREA_M2)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    vectors = list_vector_files(args.gt_dir)
    if not vectors:
        raise SystemExit(f"No hay vectores en {args.gt_dir}")
    for pair in load_pairs(args.pairs):
        # La plantilla es la B04 (10 m) de T1; T0 y T1 se validan como misma grilla en patch_extractor
        template = find_band(Path(args.s2_dir) / pair["t1"], "B04")
        rasterize_ground_truth(
            vectors,
            template,
            Path(args.out_dir) / mask_name(pair["t0"], pair["t1"]),
            date_t0=pair["t0"],
            date_t1=pair["t1"],
            min_confidence=args.min_confidence,
            min_area_m2=args.min_area_m2,
        )


if __name__ == "__main__":
    main()
