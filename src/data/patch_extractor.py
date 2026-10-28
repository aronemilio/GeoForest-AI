"""Teselado T0/T1/máscara en chips 256x256 con partición espacial disjunta (train/val).

Reglas:
- Un chip se asigna a una zona solo si su huella está contenida por completo en ella y
  dista al menos `buffer_m` de cualquier otra zona (franja de exclusión anti-autocorrelación).
  Chips en la frontera o fuera de las zonas se descartan. No hay particionado aleatorio.
- Filtrado de fondo (solo train): se descarta el 75 % de chips con 0 positivos, se retiene el
  100 % con >= 10 positivos; los de 1–9 positivos se retienen (configurable).
- Val no se filtra (se conserva la distribución real).
- Chips con más de `max_nodata_frac` de píxeles NoData (0 / NaN) se descartan.
- B04 y B08 deben compartir exactamente la grilla de referencia; B11 (20 m) se remuestrea a
  10 m con bilinear (reflectancia continua). Las máscaras nunca se remuestrean.

Uso:
    python -m src.data.patch_extractor --pairs data/raw/pairs.json \
        --zones data/raw/vectores/zonas_split.geojson --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import box

from src.data.crs_utils import assert_raster_utm18s, ensure_utm18s_gdf
from src.data.rasterize import TILE_ID, find_band, load_pairs, mask_name

log = logging.getLogger(__name__)

BANDS = ("B04", "B08", "B11")
RESAMPLABLE_BANDS = {"B11"}
ZONE_TO_SPLIT = {"carpish": "train", "huallaga_valle": "val"}
PATCH_SIZE = 256
DEFAULT_BUFFER_M = 2560.0  # 256 px * 10 m
DEFAULT_EMPTY_KEEP_FRAC = 0.25  # se descarta el 75 % de chips vacíos en train
DEFAULT_MIN_POS_KEEP = 10
DEFAULT_MAX_NODATA_FRAC = 0.05


# ----------------------------------------------------------------------------- grilla
def same_grid(a, b) -> bool:
    return a.crs == b.crs and a.transform == b.transform and a.width == b.width and a.height == b.height


def open_aligned(path: Path, band: str, ref, stack: ExitStack):
    """Abre una banda garantizando la grilla de `ref`.

    Devuelve un dataset (o WarpedVRT) legible por ventanas en la grilla de referencia.
    """
    src = stack.enter_context(rasterio.open(path))
    assert_raster_utm18s(src, path)
    if same_grid(src, ref):
        return src
    if band not in RESAMPLABLE_BANDS:
        raise ValueError(
            f"{path}: la banda {band} no comparte la grilla de referencia "
            f"(transform={src.transform}, shape={src.height}x{src.width}) y no se permite remuestrearla"
        )
    log.info("Remuestreando %s a la grilla de 10 m (bilinear)", path.name)
    return stack.enter_context(
        WarpedVRT(
            src,
            crs=ref.crs,
            transform=ref.transform,
            width=ref.width,
            height=ref.height,
            resampling=Resampling.bilinear,
        )
    )


def iter_windows(height: int, width: int, size: int = PATCH_SIZE):
    """Ventanas sin solape; se descartan chips incompletos del borde."""
    for row in range(0, height - size + 1, size):
        for col in range(0, width - size + 1, size):
            yield Window(col, row, size, size)


# ----------------------------------------------------------------------------- zonas
class ZoneIndex:
    """Asignación espacial de chips a splits según `zonas_split.geojson`."""

    def __init__(self, zones: gpd.GeoDataFrame, buffer_m: float = DEFAULT_BUFFER_M, zone_field: str = "zona"):
        if zone_field not in zones.columns:
            raise ValueError(f"El vector de zonas no tiene el atributo '{zone_field}'")
        zones = ensure_utm18s_gdf(zones, "zonas_split")
        unknown = set(zones[zone_field]) - set(ZONE_TO_SPLIT)
        if unknown:
            raise ValueError(f"Zonas desconocidas en zonas_split: {unknown}")
        dissolved = zones.dissolve(by=zone_field)
        self.geoms = {ZONE_TO_SPLIT[z]: geom for z, geom in dissolved.geometry.items()}
        self.buffer_m = buffer_m

    @classmethod
    def from_file(cls, path: str | Path, buffer_m: float = DEFAULT_BUFFER_M) -> "ZoneIndex":
        return cls(gpd.read_file(path), buffer_m=buffer_m)

    def assign(self, chip_geom) -> str | None:
        for split, geom in self.geoms.items():
            if not geom.contains(chip_geom):
                continue
            others = [g for s, g in self.geoms.items() if s != split]
            if self.buffer_m > 0 and any(chip_geom.distance(o) < self.buffer_m for o in others):
                return None  # dentro de la franja de exclusión
            return split
        return None


# ----------------------------------------------------------------------------- filtrado
def keep_patch(
    split: str,
    positives: int,
    rng: np.random.Generator,
    empty_keep_frac: float = DEFAULT_EMPTY_KEEP_FRAC,
    min_pos_keep: int = DEFAULT_MIN_POS_KEEP,
    keep_low_positive: bool = True,
) -> bool:
    if split != "train":
        return True
    if positives >= min_pos_keep:
        return True
    if positives == 0:
        return bool(rng.random() < empty_keep_frac)
    return keep_low_positive


def nodata_fraction(*stacks: np.ndarray) -> float:
    """Fracción de píxeles donde cualquier banda de cualquier fecha es 0 o NaN."""
    invalid = np.zeros(stacks[0].shape[-2:], dtype=bool)
    for s in stacks:
        invalid |= np.any((s == 0) | np.isnan(s.astype("float32")), axis=0)
    return float(invalid.mean())


# ----------------------------------------------------------------------------- escritura
def write_chip(arr: np.ndarray, window: Window, ref_profile: dict, ref_transform, path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "width": int(window.width),
        "height": int(window.height),
        "count": arr.shape[0],
        "dtype": arr.dtype.name,
        "crs": ref_profile["crs"],
        "transform": window_transform(window, ref_transform),
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def patch_name(window: Window, t0: str, t1: str) -> str:
    return f"patch_{TILE_ID}_x{int(window.col_off)}_y{int(window.row_off)}_{t0}_{t1}.tif"


# ----------------------------------------------------------------------------- núcleo
def extract_patches(
    t0_dir: str | Path,
    t1_dir: str | Path,
    mask_path: str | Path,
    zones: ZoneIndex,
    out_root: str | Path,
    t0: str,
    t1: str,
    rng: np.random.Generator,
    patch_size: int = PATCH_SIZE,
    empty_keep_frac: float = DEFAULT_EMPTY_KEEP_FRAC,
    min_pos_keep: int = DEFAULT_MIN_POS_KEEP,
    keep_low_positive: bool = True,
    max_nodata_frac: float = DEFAULT_MAX_NODATA_FRAC,
) -> dict:
    """Extrae chips de un par T0/T1 y devuelve un reporte de conteos."""
    out_root, mask_path = Path(out_root), Path(mask_path)
    stats: Counter = Counter()

    with ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(find_band(t1_dir, "B04")))
        assert_raster_utm18s(ref, ref.name)
        ref_profile, ref_transform = ref.profile, ref.transform

        mask_ds = stack.enter_context(rasterio.open(mask_path))
        if not same_grid(mask_ds, ref):
            raise ValueError(f"{mask_path}: la máscara no comparte la grilla de la plantilla Sentinel-2")

        dsets = {
            date: [open_aligned(find_band(d, b), b, ref, stack) for b in BANDS]
            for date, d in (("t0", Path(t0_dir)), ("t1", Path(t1_dir)))
        }

        for window in iter_windows(ref.height, ref.width, patch_size):
            stats["total"] += 1
            chip_geom = box(*window_bounds(window, ref_transform))
            split = zones.assign(chip_geom)
            if split is None:
                stats["descartado_frontera_o_fuera"] += 1
                continue

            mask = mask_ds.read(1, window=window)
            positives = int((mask > 0).sum())
            if not keep_patch(split, positives, rng, empty_keep_frac, min_pos_keep, keep_low_positive):
                stats[f"{split}_descartado_fondo"] += 1
                continue

            arr_t0 = np.stack([ds.read(1, window=window) for ds in dsets["t0"]])
            arr_t1 = np.stack([ds.read(1, window=window) for ds in dsets["t1"]])
            if nodata_fraction(arr_t0, arr_t1) > max_nodata_frac:
                stats[f"{split}_descartado_nodata"] += 1
                continue

            name = patch_name(window, t0, t1)
            base = out_root / split
            write_chip(arr_t0, window, ref_profile, ref_transform, base / "t0" / name)
            write_chip(arr_t1, window, ref_profile, ref_transform, base / "t1" / name)
            write_chip(mask[np.newaxis].astype("uint8"), window, ref_profile, ref_transform, base / "mask" / name)
            stats[f"{split}_guardado"] += 1
            stats[f"{split}_con_cambio" if positives > 0 else f"{split}_sin_cambio"] += 1

    report = {"t0": t0, "t1": t1, **dict(stats)}
    log.info("Par %s-%s: %s", t0, t1, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/raw/pairs.json")
    ap.add_argument("--s2-dir", default="data/raw/sentinel2")
    ap.add_argument("--masks-dir", default="data/processed/masks")
    ap.add_argument("--zones", default="data/raw/vectores/zonas_split.geojson")
    ap.add_argument("--out-dir", default="data/processed/patches")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    ap.add_argument("--max-nodata-frac", type=float, default=DEFAULT_MAX_NODATA_FRAC)
    ap.add_argument("--drop-low-positive", action="store_true", help="Descartar chips de train con 1–9 positivos")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rng = np.random.default_rng(args.seed)
    zones = ZoneIndex.from_file(args.zones, buffer_m=args.buffer_m)
    s2 = Path(args.s2_dir)
    reports = [
        extract_patches(
            s2 / p["t0"],
            s2 / p["t1"],
            Path(args.masks_dir) / mask_name(p["t0"], p["t1"]),
            zones,
            args.out_dir,
            p["t0"],
            p["t1"],
            rng,
            keep_low_positive=not args.drop_low_positive,
            max_nodata_frac=args.max_nodata_frac,
        )
        for p in load_pairs(args.pairs)
    ]
    out = Path(args.out_dir) / "extraction_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "buffer_m": args.buffer_m, "pairs": reports}, indent=2))
    log.info("Reporte: %s", out)


if __name__ == "__main__":
    main()
