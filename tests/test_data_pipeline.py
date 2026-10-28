"""Pruebas de la Fase A: rasterización, partición espacial y indexación (datos sintéticos).

Escenario sintético (EPSG:32718, 10 m):
- Tile de 1024x1024 px con origen (300000, 8950000) -> 10 240 m de lado.
- B04/B08 a 10 m; B11 a 20 m (512x512) para probar el remuestreo bilinear.
- Zonas: mitad oeste `carpish` (train), mitad este `huallaga_valle` (val); frontera x = 305120.
- Ground Truth: un polígono válido de 1 km² (100x100 px) dentro del chip (0,0), uno sub-UMC
  (200 m²) y uno de baja confianza (0.5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.crs_utils import ensure_utm18s_gdf  # noqa: E402
from src.data.indexer import INDEX_KEYS, build_index  # noqa: E402
from src.data.patch_extractor import (  # noqa: E402
    ZoneIndex,
    extract_patches,
    keep_patch,
    nodata_fraction,
)
from src.data.rasterize import rasterize_geometries, rasterize_ground_truth  # noqa: E402

X0, Y0, RES, SIZE = 300_000.0, 8_950_000.0, 10.0, 1024
BORDER_X = X0 + SIZE * RES / 2  # 305120
T0, T1 = "20230715", "20240120"
CRS_UTM = "EPSG:32718"


# ----------------------------------------------------------------------------- helpers
def _write_band(path: Path, data: np.ndarray, res: float, crs: str = CRS_UTM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
        dtype=data.dtype.name, crs=crs, transform=from_origin(X0, Y0, res, res),
    ) as dst:
        dst.write(data, 1)


def _make_date(s2_dir: Path, date: str, rng: np.random.Generator) -> Path:
    d = s2_dir / date
    for band in ("B04", "B08"):
        _write_band(d / f"T18LUQ_{date}_{band}_10m.tif",
                    rng.integers(500, 5000, (SIZE, SIZE), dtype=np.uint16), RES)
    _write_band(d / f"T18LUQ_{date}_B11_20m.tif",
                rng.integers(500, 5000, (SIZE // 2, SIZE // 2), dtype=np.uint16), 2 * RES)
    return d


def _gt_frame() -> gpd.GeoDataFrame:
    valid = box(X0 + 500, Y0 - 1500, X0 + 1500, Y0 - 500)            # 1 km² -> 10 000 px
    sub_umc = box(X0 + 3000, Y0 - 3020, X0 + 3010, Y0 - 3000)         # 200 m²
    low_conf = box(X0 + 7000, Y0 - 8000, X0 + 8000, Y0 - 7000)        # 1 km², confianza 0.5
    return gpd.GeoDataFrame(
        {
            "evento": ["deforestacion"] * 3,
            "confianza_validacion": [0.95, 0.95, 0.5],
            "fecha_t0": ["2023-07-15"] * 3,
            "fecha_t1": ["2024-01-10"] * 3,
        },
        geometry=[valid, sub_umc, low_conf],
        crs=CRS_UTM,
    )


def _zones_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"zona": ["carpish", "huallaga_valle"]},
        geometry=[
            box(X0, Y0 - SIZE * RES, BORDER_X, Y0),
            box(BORDER_X, Y0 - SIZE * RES, X0 + SIZE * RES, Y0),
        ],
        crs=CRS_UTM,
    )


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("geoforest")
    rng = np.random.default_rng(0)
    s2 = root / "data/raw/sentinel2"
    t0_dir, t1_dir = _make_date(s2, T0, rng), _make_date(s2, T1, rng)

    gt_path = root / "data/raw/ground_truth/gt.geojson"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    _gt_frame().to_file(gt_path, driver="GeoJSON")

    zones_path = root / "data/raw/vectores/zonas_split.geojson"
    zones_path.parent.mkdir(parents=True, exist_ok=True)
    _zones_frame().to_file(zones_path, driver="GeoJSON")

    template = t1_dir / f"T18LUQ_{T1}_B04_10m.tif"
    mask_path = root / f"data/processed/masks/mask_18LUQ_{T0}_{T1}.tif"
    rasterize_ground_truth([gt_path], template, mask_path, date_t0=T0, date_t1=T1)

    return {
        "root": root, "t0_dir": t0_dir, "t1_dir": t1_dir, "gt": gt_path,
        "zones": zones_path, "template": template, "mask": mask_path,
    }


@pytest.fixture(scope="module")
def patches(scene):
    """Extrae chips conservando todos los vacíos (empty_keep_frac=1) para inspeccionar la partición."""
    out = scene["root"] / "data/processed/patches"
    report = extract_patches(
        scene["t0_dir"], scene["t1_dir"], scene["mask"],
        ZoneIndex.from_file(scene["zones"], buffer_m=2560.0), out, T0, T1,
        rng=np.random.default_rng(42), empty_keep_frac=1.0,
    )
    return out, report


# ----------------------------------------------------------------------------- rasterize
def test_mask_grid_alignment(scene):
    with rasterio.open(scene["mask"]) as mask_raster, rasterio.open(scene["template"]) as sentinel_raster:
        assert mask_raster.transform == sentinel_raster.transform
        assert mask_raster.crs == sentinel_raster.crs
        assert mask_raster.shape == sentinel_raster.shape
        assert mask_raster.res == (10.0, 10.0)
        assert mask_raster.dtypes[0] == "uint8"
        assert set(np.unique(mask_raster.read(1))) <= {0, 1}


def test_rasterize_filters_umc_and_confidence(scene):
    with rasterio.open(scene["mask"]) as m:
        mask = m.read(1)
    assert int(mask.sum()) == 10_000  # solo el polígono válido de 100x100 px
    assert mask[50:150, 50:150].all()


def test_rasterize_reprojects_4326_vector(scene, tmp_path):
    gt_4326 = tmp_path / "gt_4326.geojson"
    _gt_frame().to_crs("EPSG:4326").to_file(gt_4326, driver="GeoJSON")
    out = tmp_path / "mask_4326.tif"
    rasterize_ground_truth([gt_4326], scene["template"], out, date_t0=T0, date_t1=T1)
    with rasterio.open(out) as a, rasterio.open(scene["mask"]) as b:
        assert np.array_equal(a.read(1), b.read(1))


def test_rasterize_rejects_undefined_crs():
    src = _gt_frame()
    gdf = gpd.GeoDataFrame(src.drop(columns="geometry"), geometry=list(src.geometry))
    assert gdf.crs is None
    with pytest.raises(ValueError, match="CRS indefinido"):
        ensure_utm18s_gdf(gdf, "gt_sin_crs")


def test_raster_not_utm18s_rejected(tmp_path):
    bad = tmp_path / "B04_utm17s.tif"
    _write_band(bad, np.ones((64, 64), dtype=np.uint16), RES, crs="EPSG:32717")
    with pytest.raises(ValueError, match="EPSG:32718"):
        rasterize_geometries([], bad, tmp_path / "out.tif")


def test_rasterize_temporal_window(scene, tmp_path):
    out = tmp_path / "mask_otro_par.tif"
    rasterize_ground_truth([scene["gt"]], scene["template"], out, date_t0="20220101", date_t1="20220601")
    with rasterio.open(out) as m:
        assert int(m.read(1).sum()) == 0  # el evento (2024-01-10) cae fuera de la ventana


# ----------------------------------------------------------------------------- patches
def _chip_names(out: Path, split: str) -> list[str]:
    return sorted(p.name for p in (out / split / "mask").glob("patch_*.tif"))


def test_spatial_split_no_leakage(patches):
    out, report = patches
    train, val = _chip_names(out, "train"), _chip_names(out, "val")
    assert train and val
    assert not set(train) & set(val)
    # columnas x256 y x512 tocan la frontera -> franja de exclusión
    assert all("_x0_" in n for n in train)
    assert all("_x768_" in n for n in val)
    assert report["descartado_frontera_o_fuera"] == 8

    zones = _zones_frame().set_index("zona").geometry
    for split, other in (("train", "huallaga_valle"), ("val", "carpish")):
        for p in (out / split / "mask").glob("patch_*.tif"):
            with rasterio.open(p) as ds:
                chip = box(*ds.bounds)
            assert chip.distance(zones[other]) >= 2560.0


def test_background_filter_rate():
    rng = np.random.default_rng(42)
    n = 20_000
    kept_empty = sum(keep_patch("train", 0, rng) for _ in range(n)) / n
    assert 0.23 < kept_empty < 0.27
    assert all(keep_patch("train", 10, rng) for _ in range(100))
    assert all(keep_patch("train", 3, rng) for _ in range(100))
    assert not keep_patch("train", 3, rng, keep_low_positive=False)
    assert all(keep_patch("val", 0, rng) for _ in range(100))


def test_background_filter_reproducible():
    def run(seed):
        rng = np.random.default_rng(seed)
        return [keep_patch("train", 0, rng) for _ in range(200)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_chip_georeference(scene, patches):
    out, _ = patches
    name = f"patch_18LUQ_x0_y0_{T0}_{T1}.tif"
    with rasterio.open(scene["template"]) as ref:
        expected = window_transform(Window(0, 0, 256, 256), ref.transform)
        ref_crs = ref.crs
    for sub in ("t0", "t1", "mask"):
        with rasterio.open(out / "train" / sub / name) as ds:
            assert ds.transform == expected
            assert ds.crs == ref_crs
            assert ds.shape == (256, 256)
    with rasterio.open(out / "train" / "mask" / name) as m:
        assert m.dtypes[0] == "uint8"
        assert int(m.read(1).sum()) == 10_000


def test_b11_resampled_to_10m(patches):
    out, _ = patches
    name = f"patch_18LUQ_x0_y0_{T0}_{T1}.tif"
    with rasterio.open(out / "train" / "t0" / name) as ds:
        arr = ds.read()
        assert arr.shape == (3, 256, 256)
        assert ds.res == (10.0, 10.0)
        assert (arr[2] > 0).mean() > 0.99  # B11 remuestreada; tolera píxeles de borde del warper


def test_nodata_fraction():
    a = np.ones((3, 4, 4), dtype=np.uint16)
    b = np.ones((3, 4, 4), dtype=np.uint16)
    b[1, 0, :] = 0  # 4 de 16 píxeles inválidos
    assert nodata_fraction(a, b) == pytest.approx(0.25)


# ----------------------------------------------------------------------------- indexer
def test_indexer_schema(scene, patches):
    out, _ = patches
    root = scene["root"]
    index_path = root / "data/processed/train_index.json"
    records = build_index(out / "train", index_path, project_root=root)

    on_disk = json.loads(index_path.read_text(encoding="utf-8"))
    assert on_disk == records and len(records) == 4
    for r in records:
        assert tuple(r.keys()) == INDEX_KEYS
        assert r["t0_path"].startswith("data/processed/patches/train/t0/patch_")
        assert r["has_change"] == int(r["positive_pixels"] > 0)
        assert (root / r["mask_path"]).exists()
    by_id = {r["patch_id"]: r for r in records}
    first = by_id[f"patch_18LUQ_x0_y0_{T0}_{T1}"]
    assert first["has_change"] == 1 and first["positive_pixels"] == 10_000
