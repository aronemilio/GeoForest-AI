"""Pruebas del dataset OSCD real (sin red ni datos externos)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.dataset import (
    DatasetError,
    OSCDChangeDataset,
    augment_pair,
    build_geometric_augmentation,
    discover_oscd_cities,
    parse_huanuco_pair_metadata,
    radiometric_jitter,
    split_cities,
    split_huanuco_pairs,
)



def _write_tif(path: Path, array: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=None,
        transform=from_origin(0, height, 1, 1),
    ) as ds:
        ds.write(array.astype(dtype), 1)


def _make_city(
    root: Path,
    city: str,
    *,
    label_encoding: str = "1_2",
    size: int = 64,
) -> None:
    images = root / "images" / "Onera Satellite Change Detection dataset - Images" / city
    labels = (
        root
        / "train_labels"
        / "Onera Satellite Change Detection dataset - Train Labels"
        / city
        / "cm"
    )

    base = np.arange(size * size, dtype=np.uint16).reshape(size, size) % 6000 + 1000
    for time_name in ("imgs_1_rect", "imgs_2_rect"):
        for offset, band in enumerate(("B04", "B08", "B11")):
            _write_tif(
                images / time_name / f"{band}.tif",
                base + offset * 100,
                "uint16",
            )

    if label_encoding == "1_2":
        label = np.ones((size, size), dtype=np.uint8)
        label[8:40, 8:40] = 2
    elif label_encoding == "0_1":
        label = np.zeros((size, size), dtype=np.uint8)
        label[8:40, 8:40] = 1
    else:  # pragma: no cover - ayuda a detectar errores del propio test
        raise ValueError(label_encoding)

    _write_tif(labels / f"{city}-cm.tif", label, "uint8")


def test_discover_oscd_cities(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a")
    _make_city(tmp_path, "city_b")

    assert discover_oscd_cities(tmp_path) == ["city_a", "city_b"]


def test_discover_oscd_cities_rejects_incomplete(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a")
    (tmp_path / "train_labels" / "Onera Satellite Change Detection dataset - Train Labels" / "city_b").mkdir(
        parents=True
    )

    assert discover_oscd_cities(tmp_path) == ["city_a"]


def test_split_cities_is_deterministic_and_disjoint() -> None:
    cities = ["d", "c", "b", "a"]
    train_1, val_1 = split_cities(cities, val_fraction=0.25, seed=7)
    train_2, val_2 = split_cities(cities, val_fraction=0.25, seed=7)

    assert train_1 == train_2
    assert val_1 == val_2
    assert set(train_1).isdisjoint(val_1)
    assert sorted(train_1 + val_1) == sorted(cities)
    assert len(val_1) == 1


def test_dataset_reads_six_channels_and_binary_mask(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a", label_encoding="1_2")
    dataset = OSCDChangeDataset(
        tmp_path,
        cities=["city_a"],
        patch_size=32,
        train=True,
        samples_per_epoch=8,
        positive_fraction=1.0,
        seed=3,
        augment=False,
    )

    image, mask = dataset[0]

    assert tuple(image.shape) == (6, 32, 32)
    assert tuple(mask.shape) == (1, 32, 32)
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert set(np.unique(mask.numpy()).tolist()) <= {0.0, 1.0}
    assert float(mask.sum()) > 0.0


def test_dataset_supports_zero_one_labels(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a", label_encoding="0_1")
    dataset = OSCDChangeDataset(
        tmp_path,
        cities=["city_a"],
        patch_size=32,
        train=True,
        samples_per_epoch=4,
        positive_fraction=1.0,
        seed=5,
        augment=False,
    )

    _, mask = dataset[0]
    assert float(mask.sum()) > 0.0


def test_validation_dataset_is_deterministic(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a")
    dataset = OSCDChangeDataset(
        tmp_path,
        cities=["city_a"],
        patch_size=32,
        train=False,
        seed=11,
    )

    assert len(dataset) > 0
    image_1, mask_1 = dataset[0]
    image_2, mask_2 = dataset[0]
    assert image_1.equal(image_2)
    assert mask_1.equal(mask_2)


def test_change_fraction_is_between_zero_and_one(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a")
    dataset = OSCDChangeDataset(
        tmp_path,
        cities=["city_a"],
        patch_size=32,
        train=True,
        samples_per_epoch=2,
    )

    fraction = dataset.change_fraction()
    assert 0.0 < fraction < 1.0


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="imágenes OSCD"):
        OSCDChangeDataset(
            tmp_path / "no_existe",
            patch_size=32,
            train=True,
            samples_per_epoch=1,
        )


def test_radiometric_jitter_alters_bands_within_bounds() -> None:
    rng = np.random.default_rng(42)
    img = np.full((6, 32, 32), 0.5, dtype=np.float32)
    jittered = radiometric_jitter(img, rng, p=1.0)
    assert jittered.shape == (6, 32, 32)
    assert jittered.dtype == np.float32
    assert float(jittered.min()) >= 0.0
    assert float(jittered.max()) <= 1.0
    # Comprobar que no es idéntico a la entrada constante
    assert not np.allclose(jittered, img)


def test_augment_pair_preserves_shape_and_binary_mask() -> None:
    rng = np.random.default_rng(123)
    geom = build_geometric_augmentation(patch_size=32, seed=123)
    img = np.random.rand(6, 32, 32).astype(np.float32)
    mask = (np.random.rand(32, 32) > 0.8).astype(np.uint8)

    aug_img, aug_mask = augment_pair(img, mask, geom, rng)
    assert aug_img.shape == (6, 32, 32)
    assert aug_mask.shape == (32, 32)
    assert set(np.unique(aug_mask).tolist()) <= {0, 1}
    assert float(aug_img.min()) >= 0.0
    assert float(aug_img.max()) <= 1.0


def test_dataset_with_augmentation_smoke(tmp_path: Path) -> None:
    _make_city(tmp_path, "city_a", label_encoding="1_2")
    dataset = OSCDChangeDataset(
        tmp_path,
        cities=["city_a"],
        patch_size=32,
        train=True,
        samples_per_epoch=4,
        positive_fraction=0.5,
        seed=42,
        augment=True,
    )

    image, mask = dataset[0]
    assert tuple(image.shape) == (6, 32, 32)
    assert tuple(mask.shape) == (1, 32, 32)
    assert set(np.unique(mask.numpy()).tolist()) <= {0.0, 1.0}


def test_parse_huanuco_pair_metadata_extracts_zone_and_chip() -> None:
    meta_zone = parse_huanuco_pair_metadata(
        "2023-08-03_03_puerto_inca_pachitea_p12",
        "2024-07-13_03_puerto_inca_pachitea_p12",
    )
    assert meta_zone.zone_id == "03_puerto_inca_pachitea"
    assert meta_zone.chip_code == "p12"
    assert meta_zone.chip_id == "03_puerto_inca_pachitea_p12"
    assert meta_zone.fecha_t0 == "2023-08-03"
    assert meta_zone.fecha_t1 == "2024-07-13"

    meta_ref = parse_huanuco_pair_metadata(
        "2024-07-18_S2A_MSIL2A_20240718T151701_N0510_R125_T18LTM_20240718T221950.SAFE",
        "2024-09-11_S2B_MSIL2A_20240911T151719_N0511_R125_T18LTM_20240911T203556.SAFE",
    )
    assert meta_ref.zone_id == "00_carpish_referencia"
    assert meta_ref.chip_code == "p00"
    assert meta_ref.chip_id == "00_carpish_referencia_p00"


def test_parse_huanuco_pair_metadata_rejects_mismatched_chips() -> None:
    with pytest.raises(DatasetError, match="distinto chip"):
        parse_huanuco_pair_metadata(
            "2023-08-03_03_puerto_inca_pachitea_p12",
            "2024-07-13_03_puerto_inca_pachitea_p13",
        )


def test_split_huanuco_pairs_stratified_by_zone_and_disjoint_by_chip() -> None:
    zones = [
        "01_carpish_chinchao",
        "02_leoncio_prado_tingo",
        "03_puerto_inca_pachitea",
        "04_codo_del_pozuzo",
        "05_maranon_monzon_oeste",
        "06_huanuco_amarilis_molino",
    ]
    pairs: list[tuple[str, str]] = [
        (f"2020-07-{10 + i:02d}_S2_REF", f"2021-08-{10 + i:02d}_S2_REF")
        for i in range(7)
    ]
    for z in zones:
        for r in range(4):
            for c in range(4):
                pairs.append((f"2023-08-01_{z}_p{r}{c}", f"2024-08-01_{z}_p{r}{c}"))

    assert len(pairs) == 103

    train_pairs, val_pairs = split_huanuco_pairs(pairs, val_fraction=0.25, seed=42)
    assert len(val_pairs) == 24  # 4 chips independientes por cada una de las 6 zonas
    assert len(train_pairs) == 79  # 72 chips de las 6 zonas + 7 pares temporales de referencia

    train_chips = {parse_huanuco_pair_metadata(t0, t1).chip_id for t0, t1 in train_pairs}
    val_chips = {parse_huanuco_pair_metadata(t0, t1).chip_id for t0, t1 in val_pairs}
    assert train_chips.isdisjoint(val_chips)

    # Verificar que las 6 zonas tienen exactamente 4 chips en validación
    val_zone_counts: dict[str, int] = {}
    for t0, t1 in val_pairs:
        zid = parse_huanuco_pair_metadata(t0, t1).zone_id
        val_zone_counts[zid] = val_zone_counts.get(zid, 0) + 1
    assert set(val_zone_counts.keys()) == set(zones)
    assert all(count == 4 for count in val_zone_counts.values())


def test_split_huanuco_pairs_supports_full_zone_holdout() -> None:
    pairs = [
        ("2023-08-01_01_carpish_chinchao_p00", "2024-08-01_01_carpish_chinchao_p00"),
        ("2023-08-01_01_carpish_chinchao_p01", "2024-08-01_01_carpish_chinchao_p01"),
        ("2023-08-01_03_puerto_inca_pachitea_p00", "2024-08-01_03_puerto_inca_pachitea_p00"),
        ("2023-08-01_03_puerto_inca_pachitea_p01", "2024-08-01_03_puerto_inca_pachitea_p01"),
    ]
    train_pairs, val_pairs = split_huanuco_pairs(
        pairs,
        val_zones=["03_puerto_inca_pachitea"],
    )
    assert len(train_pairs) == 2
    assert len(val_pairs) == 2
    assert all("03_puerto_inca_pachitea" in p[0] for p in val_pairs)
    assert all("01_carpish_chinchao" in p[0] for p in train_pairs)


def test_evaluate_reports_per_zone_and_macro_metrics(tmp_path: Path) -> None:
    import torch
    from torch.utils.data import DataLoader
    from src.train import FocalTverskyLoss, evaluate
    from src.unet import UNet

    _make_city(tmp_path, "city_val", label_encoding="0_1", size=64)
    val_ds = OSCDChangeDataset(
        tmp_path,
        cities=["city_val"],
        patch_size=32,
        train=False,
        seed=7,
    )
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = UNet(in_channels=6, base_features=8)
    criterion = FocalTverskyLoss()

    rng = np.random.default_rng(123)
    img_z1 = rng.random((6, 256, 256), dtype=np.float32)
    mask_z1 = np.zeros((256, 256), dtype=np.uint8)
    mask_z1[32:96, 32:96] = 1

    img_z2 = rng.random((6, 256, 256), dtype=np.float32)
    mask_z2 = np.zeros((256, 256), dtype=np.uint8)
    mask_z2[128:192, 128:192] = 1

    val_hco_pairs = [
        (
            "2023-08-01_01_carpish_chinchao_p00",
            "2024-08-01_01_carpish_chinchao_p00",
            img_z1,
            mask_z1,
        ),
        (
            "2023-08-01_03_puerto_inca_pachitea_p01",
            "2024-08-01_03_puerto_inca_pachitea_p01",
            img_z2,
            mask_z2,
        ),
    ]

    metrics = evaluate(
        model,
        val_loader,
        criterion,
        torch.device("cpu"),
        threshold=0.5,
        val_hco_pairs=val_hco_pairs,
    )

    assert metrics["huanuco_val_chips_count"] == 2
    assert set(metrics["huanuco_zones"].keys()) == {
        "01_carpish_chinchao",
        "03_puerto_inca_pachitea",
    }
    for zone_metrics in metrics["huanuco_zones"].values():
        assert zone_metrics["n_chips"] == 1
        assert 0.0 <= zone_metrics["iou"] <= 1.0
        assert 0.0 <= zone_metrics["f1"] <= 1.0
    assert 0.0 <= metrics["huanuco_macro_zone_iou"] <= 1.0
    assert 0.0 <= metrics["huanuco_macro_zone_f1"] <= 1.0


def test_train_main_end_to_end_independent_zone_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import sys
    from src.train import main

    oscd_root = tmp_path / "oscd"
    s2_root = tmp_path / "sentinel2"
    out_dir = tmp_path / "models"

    _make_city(oscd_root, "city_train", label_encoding="0_1", size=128)
    _make_city(oscd_root, "city_val", label_encoding="0_1", size=128)

    zones = ["01_carpish_chinchao", "03_puerto_inca_pachitea"]
    temporal_pairs: list[list[str]] = []
    for z in zones:
        for idx in range(4):
            t0 = f"2023-08-01_{z}_p0{idx}_S2A"
            t1 = f"2024-08-01_{z}_p0{idx}_S2A"
            temporal_pairs.append([t0, t1])
            _write_tif(s2_root / t0 / "B04.tif", np.full((256, 256), 400, dtype=np.uint16), "uint16")
            _write_tif(s2_root / t0 / "B08.tif", np.full((256, 256), 3600, dtype=np.uint16), "uint16")
            _write_tif(s2_root / t0 / "B11.tif", np.full((256, 256), 1100, dtype=np.uint16), "uint16")
            b04_t1 = np.full((256, 256), 400, dtype=np.uint16)
            b08_t1 = np.full((256, 256), 3600, dtype=np.uint16)
            b11_t1 = np.full((256, 256), 1100, dtype=np.uint16)
            b04_t1[64:128, 64:128] = 1800
            b08_t1[64:128, 64:128] = 900
            b11_t1[64:128, 64:128] = 2800
            _write_tif(s2_root / t1 / "B04.tif", b04_t1, "uint16")
            _write_tif(s2_root / t1 / "B08.tif", b08_t1, "uint16")
            _write_tif(s2_root / t1 / "B11.tif", b11_t1, "uint16")

    (s2_root / "gee_huanuco_zonas_catalog.json").write_text(
        json.dumps({"temporal_pairs": temporal_pairs}, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src/train.py",
            "--data-root",
            str(oscd_root),
            "--sentinel2-root",
            str(s2_root),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--samples-per-epoch",
            "4",
            "--patch-size",
            "64",
            "--base-features",
            "8",
            "--huanuco-val-fraction",
            "0.25",
            "--device",
            "cpu",
        ],
    )
    rc = main()
    assert rc == 0

    metrics_data = json.loads((out_dir / "training_metrics.json").read_text(encoding="utf-8"))
    assert metrics_data["data"]["huanuco_train_pairs_count"] == 6
    assert metrics_data["data"]["huanuco_val_pairs_count"] == 2

    train_chips = {
        parse_huanuco_pair_metadata(t0, t1).chip_id
        for t0, t1 in metrics_data["data"]["huanuco_train_pairs"]
    }
    val_chips = {
        parse_huanuco_pair_metadata(t0, t1).chip_id
        for t0, t1 in metrics_data["data"]["huanuco_val_pairs"]
    }
    assert train_chips.isdisjoint(val_chips)

    best_m = metrics_data["training"]["metrics"]
    assert best_m["huanuco_val_chips_count"] == 2
    assert set(best_m["huanuco_zones"].keys()) == set(zones)
    assert "huanuco_macro_zone_iou" in best_m
    assert "huanuco_macro_zone_f1" in best_m

    # Probar también holdout de zona geográfica completa (--val-huanuco-zones)
    out_dir_zone = tmp_path / "models_zone_holdout"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "src/train.py",
            "--data-root",
            str(oscd_root),
            "--sentinel2-root",
            str(s2_root),
            "--output-dir",
            str(out_dir_zone),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--samples-per-epoch",
            "4",
            "--patch-size",
            "64",
            "--base-features",
            "8",
            "--val-huanuco-zones",
            "03_puerto_inca_pachitea",
            "--device",
            "cpu",
        ],
    )
    assert main() == 0
    zone_metrics_data = json.loads((out_dir_zone / "training_metrics.json").read_text(encoding="utf-8"))
    assert zone_metrics_data["data"]["huanuco_train_pairs_count"] == 4
    assert zone_metrics_data["data"]["huanuco_val_pairs_count"] == 4
    assert list(zone_metrics_data["training"]["metrics"]["huanuco_zones"].keys()) == [
        "03_puerto_inca_pachitea"
    ]
    assert (
        zone_metrics_data["training"]["metrics"]["huanuco_zones"]["03_puerto_inca_pachitea"][
            "n_chips"
        ]
        == 4
    )





