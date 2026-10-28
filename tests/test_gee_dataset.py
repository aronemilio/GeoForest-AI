"""Pruebas unitarias y de integración para ``src/gee_dataset.py`` y el dataset multitemporal de Huánuco."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.dataset import (
    discover_huanuco_pairs,
    discover_sentinel2_scenes,
    load_all_huanuco_sentinel2_pairs,
)
from src.gee_dataset import (
    GEE_COLLECTION,
    GEE_PROJECT_ID,
    HUANUCO_RICHDATASET_SCENES,
    HUANUCO_TEMPORAL_PAIRS,
    HUANUCO_ZONES,
    MGRS_TILE_HUANUCO,
    ZONE_DATE_PAIR,
    verify_scene_geotiffs,
    zone_utm_bounds,
)

ZONE_SCENE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)_p(\d{2})_")


def test_gee_project_configuration() -> None:
    """Verifica el ID del proyecto GEE y colección oficial."""
    assert GEE_PROJECT_ID == "geoforest-ai-512500"
    assert GEE_COLLECTION == "COPERNICUS/S2_SR_HARMONIZED"
    assert MGRS_TILE_HUANUCO == "18LUQ"
    assert len(HUANUCO_RICHDATASET_SCENES) >= 11
    assert len(HUANUCO_TEMPORAL_PAIRS) >= 7


def test_huanuco_six_zones_configuration() -> None:
    """Verifica las seis zonas operativas y que cada bloque mida 10.24 km (4x4 chips)."""
    assert len(HUANUCO_ZONES) == 6
    assert [z.zone_id for z in HUANUCO_ZONES] == [
        "01_carpish_chinchao",
        "02_leoncio_prado_tingo",
        "03_puerto_inca_pachitea",
        "04_codo_del_pozuzo",
        "05_maranon_monzon_oeste",
        "06_huanuco_amarilis_molino",
    ]
    for zone in HUANUCO_ZONES:
        assert -10.0 < zone.center_lat < -8.0
        assert -77.5 < zone.center_lon < -74.0
        assert zone.block_km == pytest.approx(10.24)
        assert zone.chips_per_side == 4
        assert zone.ecosistema
        assert zone.altitud_m
        assert zone.dinamica_cambio
        west, south, east, north = zone_utm_bounds(zone)
        assert east - west == pytest.approx(10240.0, abs=0.5)
        assert north - south == pytest.approx(10240.0, abs=0.5)

    assert ZONE_DATE_PAIR == ("2024-07-18", "2024-09-11")


def test_downloaded_huanuco_zones_on_disk() -> None:
    """Verifica las 6 zonas, 192 chips, 96 pares y que no haya cruces entre parches."""
    s2_root = Path("data/raw/sentinel2")
    catalog_path = s2_root / "gee_huanuco_zonas_catalog.json"
    assert catalog_path.is_file(), "Falta ejecutar `python src/gee_dataset.py --zones`"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["gee_project_id"] == "geoforest-ai-512500"
    assert catalog["total_zones"] == 6
    assert catalog["total_scenes"] == 192
    assert catalog["total_pairs"] == 96
    assert len(catalog["zones"]) == 6
    assert len(catalog["temporal_pairs"]) == 96

    scene_names = set(discover_sentinel2_scenes(s2_root))
    zone_ids = {zone["zone_id"] for zone in catalog["zones"]}
    assert zone_ids == {zone.zone_id for zone in HUANUCO_ZONES}

    for zone in catalog["zones"]:
        assert len(zone["chips_by_date"][ZONE_DATE_PAIR[0]]) == 16
        assert len(zone["chips_by_date"][ZONE_DATE_PAIR[1]]) == 16
        assert len(zone["temporal_pairs"]) == 16
        for fecha in ZONE_DATE_PAIR:
            for name in zone["chips_by_date"][fecha]:
                assert name in scene_names
                stats = verify_scene_geotiffs(
                    s2_root / name, expected_shape=(256, 256)
                )
                for banda in ("B04", "B08", "B11"):
                    assert stats["bands"][banda]["max"] > 0

        for t0, t1 in zone["temporal_pairs"]:
            assert t0 in scene_names and t1 in scene_names
            m0 = ZONE_SCENE_RE.match(t0)
            m1 = ZONE_SCENE_RE.match(t1)
            assert m0 is not None and m1 is not None
            assert m0.group(2) == zone["zone_id"]
            assert m1.group(2) == zone["zone_id"]
            assert m0.group(3) == m1.group(3)


def test_downloaded_huanuco_rich_dataset_on_disk() -> None:
    """Verifica que las 11 escenas reales y el catálogo GEE estén descargados e íntegros en disco."""
    s2_root = Path("data/raw/sentinel2")
    scenes = discover_sentinel2_scenes(s2_root)
    assert len(scenes) >= 11

    for spec in HUANUCO_RICHDATASET_SCENES:
        folder = f"{spec.fecha}_{spec.scene_id}"
        assert folder in scenes, f"Falta escena descargada: {folder}"
        stats = verify_scene_geotiffs(s2_root / folder)
        assert stats["scene"] == folder
        for b in ("B04", "B08", "B11"):
            assert stats["bands"][b]["max"] > 0

    catalog_path = s2_root / "gee_huanuco_catalog.json"
    assert catalog_path.is_file()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert catalog["gee_project_id"] == "geoforest-ai-512500"
    assert catalog["total_scenes"] >= 11


def test_discover_and_load_all_huanuco_pairs() -> None:
    """Verifica que src/dataset.py descubra y cargue todos los pares multitemporales de Huánuco."""
    pairs = discover_huanuco_pairs("data/raw/sentinel2")
    assert len(pairs) >= 103
    for t0, t1 in HUANUCO_TEMPORAL_PAIRS:
        assert (t0, t1) in pairs

    # Los pares de las seis zonas nunca deben cruzar zona ni parche espacial.
    for t0, t1 in pairs:
        m0 = ZONE_SCENE_RE.match(t0)
        m1 = ZONE_SCENE_RE.match(t1)
        if m0 is not None and m1 is not None:
            assert m0.group(2) == m1.group(2), f"Cruce de zona: {t0} -> {t1}"
            assert m0.group(3) == m1.group(3), f"Cruce de parche: {t0} -> {t1}"

    loaded = load_all_huanuco_sentinel2_pairs("data/raw/sentinel2")
    assert len(loaded) == len(pairs)
    for t0_name, t1_name, img, msk in loaded:
        assert img.shape == (6, 256, 256)
        assert img.dtype == np.float32
        assert 0.0 <= float(img.min()) <= float(img.max()) <= 1.0
        assert msk.shape == (256, 256)
        assert msk.dtype == np.uint8
        assert int(msk.sum()) > 0, f"Máscara vacía para el par {t0_name} -> {t1_name}"


def test_init_gee_with_mock() -> None:
    """Verifica que init_gee invoque ee.Initialize con project='geoforest-ai-512500'."""
    from src.gee_dataset import init_gee

    mock_ee = MagicMock()
    with patch.dict("sys.modules", {"ee": mock_ee}):
        ok = init_gee("geoforest-ai-512500")
        assert ok is True
        mock_ee.Initialize.assert_called_once_with(project="geoforest-ai-512500")


def test_resolve_training_device_prioritizes_gpu() -> None:
    """Verifica que resolve_training_device priorice GPU CUDA, active TF32/cuDNN y limite hilos de CPU."""
    import torch
    from src.train import resolve_training_device

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "empty_cache"
    ):
        dev = resolve_training_device("auto")
        assert dev.type == "cuda"
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.get_num_threads() == 2

    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="--require-gpu"):
            resolve_training_device("auto", require_gpu=True)

