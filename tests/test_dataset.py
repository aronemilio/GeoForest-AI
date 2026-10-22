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
    radiometric_jitter,
    split_cities,
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

