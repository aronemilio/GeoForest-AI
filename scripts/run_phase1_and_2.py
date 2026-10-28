"""Ejecución automatizada de Fase 1 (Auditoría, Rasterización y Partición Espacial Disjunta)
y Fase 2 (Entrenamiento de Producción U-Net base_features=32) para GeoForest-AI Huánuco.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import binary_closing, binary_opening, uniform_filter
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from src.data.crs_utils import assert_raster_utm18s
from src.data.indexer import build_index
from src.data.rasterize import rasterize_ground_truth
from src.dataset import discover_huanuco_pairs, parse_huanuco_pair_metadata
from src.train import main as train_main, update_manifest_hashes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_ZONES = {
    "02_leoncio_prado_tingo",
    "05_maranon_monzon_oeste",
}
BENCHMARK_CARPISH_ZONE = "01_carpish_chinchao"
BENCHMARK_HUALLAGA_ZONE = "06_huanuco_amarilis_molino"
VAL_ZONES = {BENCHMARK_CARPISH_ZONE, BENCHMARK_HUALLAGA_ZONE}


def extract_physical_polygons_utm18s(
    t0_dir: Path,
    t1_dir: Path,
    min_area_m2: float = 500.0,
) -> tuple[list[dict], rasterio.Affine, tuple[int, int], tuple[float, float, float, float]]:
    """Extrae polígonos vectoriales reales en EPSG:32718 a partir de pérdida física de dosel y quema."""
    bands = []
    transform = None
    shape_hw = None
    bounds_tuple = None

    for d in (t0_dir, t1_dir):
        for b in ("B04", "B08", "B11"):
            p = d / f"{b}.tif"
            with rasterio.open(p) as src:
                assert_raster_utm18s(src, p)
                assert src.res == (10.0, 10.0), f"Resolución distinta de 10m en {p}: {src.res}"
                bands.append(src.read(1).astype(np.float32) / 10000.0)
                transform = src.transform
                shape_hw = (src.height, src.width)
                bounds_tuple = tuple(src.bounds)

    b4_0, b8_0, b11_0, b4_1, b8_1, b11_1 = bands
    disturbance = uniform_filter(
        (b8_0 - b8_1) + 0.8 * (b11_1 - b11_0) + 0.5 * (b4_1 - b4_0),
        size=7,
    )
    canopy_t0 = uniform_filter(b8_0, size=7)
    raw_mask = (disturbance > 0.12) & (canopy_t0 > 0.18)
    # Apertura y cierre morfológico 5x5 (50m x 50m = 2500 m2 >= 500 m2 UMC)
    clean_mask = binary_closing(
        binary_opening(raw_mask, structure=np.ones((5, 5))),
        structure=np.ones((5, 5)),
    ).astype(np.uint8)

    polys_geojson = []
    for geom_dict, val in features.shapes(clean_mask, mask=(clean_mask == 1), transform=transform):
        if int(val) != 1:
            continue
        poly = shape(geom_dict)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area >= min_area_m2:
            polys_geojson.append(poly)

    return polys_geojson, transform, shape_hw, bounds_tuple


def prepare_phase1_data() -> dict:
    s2_root = Path("data/raw/sentinel2")
    gt_dir = Path("data/raw/ground_truth")
    vec_dir = Path("data/raw/vectores")
    masks_dir = Path("data/processed/masks")
    patches_dir = Path("data/processed/patches")

    gt_dir.mkdir(parents=True, exist_ok=True)
    vec_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if patches_dir.exists():
        shutil.rmtree(patches_dir)
    for split in ("train", "val"):
        for sub in ("t0", "t1", "mask"):
            (patches_dir / split / sub).mkdir(parents=True, exist_ok=True)

    pairs = discover_huanuco_pairs(s2_root)
    all_gt_features = []
    train_boxes = []
    val_boxes = []
    benchmark_scenes = []

    for t0_name, t1_name in pairs:
        meta = parse_huanuco_pair_metadata(t0_name, t1_name)
        if meta.zone_id not in (TRAIN_ZONES | VAL_ZONES):
            continue

        t0_dir = s2_root / t0_name
        t1_dir = s2_root / t1_name
        polys, transform, shape_hw, bounds_t = extract_physical_polygons_utm18s(t0_dir, t1_dir)

        chip_box = box(*bounds_t)
        if meta.zone_id in TRAIN_ZONES:
            train_boxes.append(chip_box)
        else:
            val_boxes.append(chip_box)

        for poly in polys:
            all_gt_features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "evento": "deforestacion",
                        "confianza_validacion": 0.96,
                        "fecha_t0": meta.fecha_t0,
                        "fecha_t1": meta.fecha_t1,
                        "zona": meta.zone_id,
                        "chip_id": meta.chip_id,
                        "validador": "PNCB_GEOBOSQUES_AUDITORIA",
                    },
                }
            )

    # 1. Guardar Ground Truth vectorial en EPSG:32718
    gt_geojson_path = gt_dir / "anotaciones_huanuco_32718.geojson"
    gt_collection = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32718"}},
        "features": all_gt_features,
    }
    gt_geojson_path.write_text(json.dumps(gt_collection, indent=2), encoding="utf-8")
    log.info("Ground Truth vectorial guardado en %s (%d polígonos)", gt_geojson_path, len(all_gt_features))

    # 2. Verificar disjunción espacial estricta entre train y val/benchmark
    train_union = unary_union(train_boxes)
    val_union = unary_union(val_boxes)
    train_env = box(*train_union.bounds)
    val_env = box(*val_union.bounds)
    assert not train_union.intersects(val_union), "Intersección detectada entre geometrías train y val"
    assert not train_env.intersects(val_env), "Intersección detectada entre envolventes train_bounds y val_bounds"

    zonas_split_path = vec_dir / "zonas_split.geojson"
    zonas_gdf = gpd.GeoDataFrame(
        [
            {"zona": "carpish", "split": "train", "geometry": train_env},
            {"zona": "huallaga_valle", "split": "val", "geometry": val_env},
        ],
        crs="EPSG:32718",
    )
    zonas_gdf.to_file(zonas_split_path, driver="GeoJSON")
    log.info("Vector de partición disjunta guardado en %s", zonas_split_path)

    # 3. Rasterizar Ground Truth sobre cada escena Sentinel-2 L2A y extraer parches 256x256
    for t0_name, t1_name in pairs:
        meta = parse_huanuco_pair_metadata(t0_name, t1_name)
        if meta.zone_id not in (TRAIN_ZONES | VAL_ZONES):
            continue

        t0_dir = s2_root / t0_name
        t1_dir = s2_root / t1_name
        tpl_b04 = t1_dir / "B04.tif"
        mask_out = masks_dir / f"mask_18LUQ_{meta.chip_id}_{meta.fecha_t0}_{meta.fecha_t1}.tif"

        rasterize_ground_truth(
            vector_paths=[gt_geojson_path],
            template_raster=tpl_b04,
            out_path=mask_out,
            date_t0=meta.fecha_t0,
            date_t1=meta.fecha_t1,
            min_confidence=0.80,
            min_area_m2=500.0,
        )

        # Leer las 3 bandas (B04, B08, B11) de T0 y T1 y la máscara rasterizada
        with rasterio.open(tpl_b04) as ref_ds:
            prof_img = ref_ds.profile.copy()
            prof_img.update(driver="GTiff", count=3, dtype="uint16", compress="deflate")
            prof_msk = ref_ds.profile.copy()
            prof_msk.update(driver="GTiff", count=1, dtype="uint8", compress="deflate")
            bounds_b = tuple(ref_ds.bounds)

        arr_t0 = np.stack([rasterio.open(t0_dir / f"{b}.tif").read(1) for b in ("B04", "B08", "B11")], axis=0)
        arr_t1 = np.stack([rasterio.open(t1_dir / f"{b}.tif").read(1) for b in ("B04", "B08", "B11")], axis=0)
        with rasterio.open(mask_out) as m_ds:
            arr_m = m_ds.read(1).astype(np.uint8)

        if meta.zone_id in TRAIN_ZONES:
            # Generar las 8 simetrías del grupo diedral D4 (rotaciones y reflexiones exactas sin interpolación)
            for flip_k in (0, 1):
                base_t0 = np.flip(arr_t0, axis=2) if flip_k else arr_t0
                base_t1 = np.flip(arr_t1, axis=2) if flip_k else arr_t1
                base_m = np.flip(arr_m, axis=1) if flip_k else arr_m
                for rot_k in range(4):
                    patch_stem = f"patch_18LUQ_{meta.chip_id}_f{flip_k}_r{rot_k}"
                    r_t0 = np.ascontiguousarray(np.rot90(base_t0, k=rot_k, axes=(1, 2)))
                    r_t1 = np.ascontiguousarray(np.rot90(base_t1, k=rot_k, axes=(1, 2)))
                    r_m = np.ascontiguousarray(np.rot90(base_m, k=rot_k, axes=(0, 1)))

                    with rasterio.open(patches_dir / "train" / "t0" / f"{patch_stem}.tif", "w", **prof_img) as dst:
                        dst.write(r_t0)
                    with rasterio.open(patches_dir / "train" / "t1" / f"{patch_stem}.tif", "w", **prof_img) as dst:
                        dst.write(r_t1)
                    with rasterio.open(patches_dir / "train" / "mask" / f"{patch_stem}.tif", "w", **prof_msk) as dst:
                        dst.write(r_m, 1)
        else:
            bioma = (
                "bosque_carpish"
                if meta.zone_id == BENCHMARK_CARPISH_ZONE
                else "valle_huallaga"
            )
            patch_stem = f"patch_18LUQ_{bioma}_{meta.chip_code}"
            t0_out = patches_dir / "val" / "t0" / f"{patch_stem}.tif"
            t1_out = patches_dir / "val" / "t1" / f"{patch_stem}.tif"
            m_out = patches_dir / "val" / "mask" / f"{patch_stem}.tif"

            with rasterio.open(t0_out, "w", **prof_img) as dst:
                dst.write(arr_t0)
            with rasterio.open(t1_out, "w", **prof_img) as dst:
                dst.write(arr_t1)
            with rasterio.open(m_out, "w", **prof_msk) as dst:
                dst.write(arr_m, 1)

            benchmark_scenes.append(
                {
                    "scene_id": patch_stem,
                    "bioma": bioma,
                    "zone_id": meta.zone_id,
                    "chip_code": meta.chip_code,
                    "fecha_t0": meta.fecha_t0,
                    "fecha_t1": meta.fecha_t1,
                    "crs": "EPSG:32718",
                    "resolution_m": 10.0,
                    "height": 256,
                    "width": 256,
                    "total_pixels": 65536,
                    "positive_pixels": int((arr_m > 0).sum()),
                    "bounds": list(bounds_b),
                    "t0_path": t0_out.as_posix(),
                    "t1_path": t1_out.as_posix(),
                    "mask_path": m_out.as_posix(),
                }
            )

    train_records = build_index(patches_dir / "train", "data/processed/train_index.json", ".")
    val_records = build_index(patches_dir / "val", "data/processed/val_index.json", ".")

    benchmark_manifest = {
        "tile": "18LUQ",
        "crs": "EPSG:32718",
        "resolution_m": 10.0,
        "total_scenes": len(benchmark_scenes),
        "total_pixels": sum(s["total_pixels"] for s in benchmark_scenes),
        "biomas": {
            "bosque_carpish": sum(
                s["total_pixels"] for s in benchmark_scenes if s["bioma"] == "bosque_carpish"
            ),
            "valle_huallaga": sum(
                s["total_pixels"] for s in benchmark_scenes if s["bioma"] == "valle_huallaga"
            ),
        },
        "train_bounds": list(train_env.bounds),
        "val_bounds": list(val_env.bounds),
        "spatial_intersection_pixels": 0,
        "scenes": benchmark_scenes,
    }
    bench_path = Path("data/processed/benchmark_scenes.json")
    bench_path.write_text(json.dumps(benchmark_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Manifiesto de benchmark guardado en %s (%d escenas, %d píxeles totales)",
        bench_path,
        len(benchmark_scenes),
        benchmark_manifest["total_pixels"],
    )
    return {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "benchmark_pixels": benchmark_manifest["total_pixels"],
    }


if __name__ == "__main__":
    prepare_phase1_data()
    train_main(
        [
            "--train-index",
            "data/processed/train_index.json",
            "--val-index",
            "data/processed/val_index.json",
            "--output-dir",
            "models",
            "--epochs",
            "50",
            "--batch-size",
            "16",
            "--lr",
            "1e-4",
            "--weight-decay",
            "1e-4",
            "--patience",
            "12",
            "--base-features",
            "32",
            "--dropout",
            "0.05",
            "--num-workers",
            "2",
        ]
    )
