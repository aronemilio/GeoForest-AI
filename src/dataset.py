"""Dataset real de cambio de cobertura basado en OSCD.

OSCD (Onera Satellite Change Detection) contiene pares de imágenes Sentinel-2
registradas y etiquetas binarias de cambio a nivel de píxel. Este módulo **no
descarga datos**: exige que el dataset ya esté en disco.

Contrato de entrada (6 canales, orden idéntico al de la app):

- T0_B04, T0_B08, T0_B11
- T1_B04, T1_B08, T1_B11

Las etiquetas OSCD vienen en dos codificaciones según la versión del dataset:

- ``0/1``: 1 = cambio.
- ``1/2``: 2 = cambio (es la codificación de los TIFF de este repositorio).

La semántica es **cambio de cobertura/urbano de OSCD**, no deforestación en
Huánuco. No se usan puntos PNCB como máscaras: convertirlos en polígonos o
áreas sería inventar etiquetas.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

try:  # Aumento geométrico agresivo (opcional; hay fallback con NumPy)
    import albumentations as A

    _HAS_ALBUMENTATIONS = True
except ImportError:  # pragma: no cover - depende del entorno
    A = None
    _HAS_ALBUMENTATIONS = False

BANDAS: tuple[str, ...] = ("B04", "B08", "B11")
CANALES: tuple[str, ...] = (
    "T0_B04",
    "T0_B08",
    "T0_B11",
    "T1_B04",
    "T1_B08",
    "T1_B11",
)
IMAGE_ROOT_NAME = "Onera Satellite Change Detection dataset - Images"
LABEL_ROOT_NAME = "Onera Satellite Change Detection dataset - Train Labels"
NORMALIZATION_DIVISOR = 10_000.0


class DatasetError(RuntimeError):
    """Error de estructura, lectura o etiquetas del dataset OSCD."""


@dataclass(frozen=True)
class OSCDPatch:
    """Coordenada superior izquierda de un parche dentro de una ciudad."""

    city: str
    row: int
    col: int


def _resolve_roots(data_root: str | Path) -> tuple[Path, Path]:
    """Localiza las carpetas de imágenes y etiquetas dentro de ``data_root``."""
    root = Path(data_root)
    images = root / "images" / IMAGE_ROOT_NAME
    if not images.is_dir():
        images = root / "images"

    labels = root / "train_labels" / LABEL_ROOT_NAME
    if not labels.is_dir():
        labels = root / "train_labels"

    if not images.is_dir():
        raise DatasetError(
            f"No se encontró la carpeta de imágenes OSCD en {root}. "
            "Se esperaba 'images/Onera Satellite Change Detection dataset - Images' "
            "o 'images'."
        )
    if not labels.is_dir():
        raise DatasetError(
            f"No se encontró la carpeta de etiquetas OSCD en {root}. "
            "Se esperaba 'train_labels/Onera Satellite Change Detection dataset - Train Labels' "
            "o 'train_labels'."
        )
    return images, labels


def _label_path(labels_root: Path, city: str) -> Path:
    return labels_root / city / "cm" / f"{city}-cm.tif"


def _city_paths(images_root: Path, labels_root: Path, city: str) -> dict[str, Path]:
    city_images = images_root / city
    paths: dict[str, Path] = {}
    for time_name in ("imgs_1_rect", "imgs_2_rect"):
        for band in BANDAS:
            paths[f"{time_name}/{band}"] = city_images / time_name / f"{band}.tif"
    paths["label"] = _label_path(labels_root, city)
    return paths


def discover_oscd_cities(data_root: str | Path) -> list[str]:
    """Devuelve las ciudades con las 6 bandas y la etiqueta completas."""
    images_root, labels_root = _resolve_roots(data_root)
    cities: list[str] = []

    for label_dir in sorted(labels_root.iterdir()):
        if not label_dir.is_dir():
            continue
        city = label_dir.name
        paths = _city_paths(images_root, labels_root, city)
        if all(path.is_file() for path in paths.values()):
            cities.append(city)

    if not cities:
        raise DatasetError(
            f"No hay ciudades OSCD completas en {Path(data_root)}. "
            "Cada ciudad necesita imgs_1_rect, imgs_2_rect (B04/B08/B11) "
            "y train_labels/<ciudad>/cm/<ciudad>-cm.tif."
        )
    return cities


def split_cities(
    cities: Sequence[str],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Divide ciudades (no parches) para evitar fuga de información."""
    unique = sorted(set(cities))
    if len(unique) < 2:
        raise DatasetError("Se necesitan al menos 2 ciudades para separar train/val.")
    if not 0.0 < val_fraction < 1.0:
        raise DatasetError("val_fraction debe estar entre 0 y 1.")

    n_val = max(1, int(round(len(unique) * val_fraction)))
    n_val = min(n_val, len(unique) - 1)

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


@lru_cache(maxsize=32)
def _load_label(label_path: str) -> np.ndarray:
    """Lee una etiqueta OSCD y la convierte a máscara binaria de cambio."""
    path = Path(label_path)
    with rasterio.open(path) as ds:
        arr = ds.read(1)

    values = {int(v) for v in np.unique(arr)}
    if values <= {0, 1}:
        change = arr == 1
    elif values <= {1, 2}:
        change = arr == 2
    elif values <= {0, 1, 2}:
        # Algunas versiones usan 0 como fondo y 1/2 como no cambio/cambio.
        change = arr == 2
    else:
        raise DatasetError(
            f"Etiqueta OSCD con valores no soportados en {path}: {sorted(values)}. "
            "Se esperaba 0/1 o 1/2."
        )

    return change.astype(np.uint8)


@lru_cache(maxsize=6)
def _load_city_arrays(data_root: str, city: str) -> tuple[np.ndarray, np.ndarray]:
    """Carga y normaliza las 6 bandas de una ciudad y su máscara de cambio."""
    images_root, labels_root = _resolve_roots(data_root)
    paths = _city_paths(images_root, labels_root, city)

    bands: list[np.ndarray] = []
    for time_name in ("imgs_1_rect", "imgs_2_rect"):
        for band in BANDAS:
            path = paths[f"{time_name}/{band}"]
            with rasterio.open(path) as ds:
                arr = ds.read(1)
            bands.append(arr)

    shapes = {arr.shape for arr in bands}
    if len(shapes) != 1:
        raise DatasetError(
            f"Las bandas de {city} no comparten la misma cuadrícula: {sorted(shapes)}."
        )

    image = np.stack(bands, axis=0).astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=NORMALIZATION_DIVISOR, neginf=0.0)
    image = np.clip(image / NORMALIZATION_DIVISOR, 0.0, 1.0)

    label = _load_label(str(paths["label"]))
    if label.shape != image.shape[1:]:
        raise DatasetError(
            f"La etiqueta de {city} tiene forma {label.shape}, pero las bandas "
            f"tienen forma {image.shape[1:]}. Deben estar alineadas."
        )

    return image, label


def _positive_top_lefts(label: np.ndarray, patch_size: int, stride: int = 16) -> list[tuple[int, int]]:
    """Posiciones con al menos un píxel de cambio, usando imagen integral."""
    height, width = label.shape
    if height < patch_size or width < patch_size:
        return []

    integral = np.pad(
        label.astype(np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    positions: list[tuple[int, int]] = []
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            total = (
                integral[row + patch_size, col + patch_size]
                - integral[row, col + patch_size]
                - integral[row + patch_size, col]
                + integral[row, col]
            )
            if total > 0:
                positions.append((row, col))
    return positions


def _grid_top_lefts(label: np.ndarray, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Posiciones deterministas que cubren la imagen, incluyendo bordes."""
    height, width = label.shape
    if height < patch_size or width < patch_size:
        return []

    rows = list(range(0, height - patch_size + 1, stride))
    cols = list(range(0, width - patch_size + 1, stride))
    if rows[-1] != height - patch_size:
        rows.append(height - patch_size)
    if cols[-1] != width - patch_size:
        cols.append(width - patch_size)

    return [(row, col) for row in sorted(set(rows)) for col in sorted(set(cols))]


def build_geometric_augmentation(patch_size: int, seed: int | None = None):
    """Pipeline geométrico agresivo idéntico para T0, T1 (canales apilados) y máscara.

    - ``RandomRotate90``: 0/90/180/270° (siempre aplicado, ángulo aleatorio).
    - Volteos H/V.
    - ``RandomResizedCrop``: recorte aleatorio 50–100 % del parche reescalado al tamaño original.
    - ``ElasticTransform`` o ``GridDistortion``: deforma bordes de parcelas y trochas
      para que la red aprenda geometría robusta y no posiciones/texturas fijas.

    La imagen se interpola bilinealmente y la máscara con vecino más cercano
    (por defecto en albumentations 2.x), así la etiqueta sigue siendo binaria.
    Devuelve ``None`` si albumentations no está instalado (se usa el fallback NumPy).
    """
    if not _HAS_ALBUMENTATIONS:
        return None
    return A.Compose(
        [
            A.RandomRotate90(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(
                size=(patch_size, patch_size),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.333),
                p=0.7,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=40.0, sigma=6.0, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                ],
                p=0.5,
            ),
        ],
        seed=seed,
    )


def radiometric_jitter(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    gain_range: tuple[float, float] = (0.8, 1.2),
    inter_date_gain: float = 0.05,
    offset_std: float = 0.02,
    noise_std: float = 0.01,
    p: float = 0.8,
) -> np.ndarray:
    """Perturba la firma espectral cruda (C, H, W) para romper atajos de umbral.

    Ganancia distinta por banda (cambia el NDVI/NBR absoluto), ligera ganancia
    distinta entre fechas (cambia ΔNDVI/ΔNBR), offset y ruido gaussiano.
    """
    if rng.random() > p:
        return image
    channels = image.shape[0]
    n_bands = channels // 2
    gain = rng.uniform(*gain_range, size=(n_bands, 1, 1))
    gain_t1 = gain * rng.uniform(1.0 - inter_date_gain, 1.0 + inter_date_gain, size=(n_bands, 1, 1))
    gains = np.concatenate([gain, gain_t1], axis=0)
    offset = rng.normal(0.0, offset_std, size=(channels, 1, 1))
    noise = rng.normal(0.0, noise_std, size=image.shape)
    out = image * gains + offset + noise
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def augment_pair(
    image_chw: np.ndarray,
    mask_hw: np.ndarray,
    geometric,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica el aumento geométrico (mismo para imagen y máscara) y luego el radiométrico."""
    if geometric is not None:
        out = geometric(
            image=np.ascontiguousarray(image_chw.transpose(1, 2, 0), dtype=np.float32),
            mask=np.ascontiguousarray(mask_hw, dtype=np.uint8),
        )
        image_chw = out["image"].transpose(2, 0, 1)
        mask_hw = out["mask"]
    else:  # Fallback sin albumentations: rotaciones 90° y volteos
        k = int(rng.integers(0, 4))
        image_chw = np.rot90(image_chw, k=k, axes=(1, 2))
        mask_hw = np.rot90(mask_hw, k=k, axes=(0, 1))
        if rng.random() < 0.5:
            image_chw, mask_hw = image_chw[:, :, ::-1], mask_hw[:, ::-1]
        if rng.random() < 0.5:
            image_chw, mask_hw = image_chw[:, ::-1, :], mask_hw[::-1, :]

    image_chw = radiometric_jitter(np.ascontiguousarray(image_chw, dtype=np.float32), rng)
    mask_hw = (np.ascontiguousarray(mask_hw) > 0).astype(np.uint8)
    return np.ascontiguousarray(image_chw), mask_hw


class OSCDChangeDataset(Dataset):
    """Dataset de pares multitemporales Sentinel-2 con etiquetas de cambio OSCD.

    Args:
        data_root: Ruta que contiene ``images/`` y ``train_labels/``.
        cities: Ciudades a incluir. Si es ``None``, usa todas las completas.
        patch_size: Tamaño cuadrado del parche.
        train: Si es ``True``, muestrea parches aleatorios con oversampling de
            cambios. Si es ``False``, usa una rejilla determinista.
        samples_per_epoch: Número de muestras por época en entrenamiento.
        positive_fraction: Probabilidad de elegir un parche con cambio.
        seed: Semilla para muestreo y aumentos.
        augment: Activa volteos y rotaciones de 90° solo en entrenamiento.
    """

    def __init__(
        self,
        data_root: str | Path,
        *,
        cities: Sequence[str] | None = None,
        patch_size: int = 256,
        train: bool = True,
        samples_per_epoch: int = 512,
        positive_fraction: float = 0.5,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        if patch_size < 16:
            raise DatasetError("patch_size debe ser >= 16.")
        if train and samples_per_epoch < 1:
            raise DatasetError("samples_per_epoch debe ser >= 1.")
        if not 0.0 <= positive_fraction <= 1.0:
            raise DatasetError("positive_fraction debe estar entre 0 y 1.")

        self.data_root = Path(data_root)
        self.cities = sorted(cities) if cities is not None else discover_oscd_cities(self.data_root)
        if not self.cities:
            raise DatasetError("La lista de ciudades está vacía.")

        self.patch_size = int(patch_size)
        self.train = bool(train)
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.augment = bool(augment)
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._geometric = (
            build_geometric_augmentation(int(patch_size), seed=seed)
            if self.augment and self.train
            else None
        )

        self._positive_positions: dict[str, list[tuple[int, int]]] = {}
        self._val_patches: list[OSCDPatch] = []

        for city in self.cities:
            image, label = _load_city_arrays(str(self.data_root), city)
            if image.shape[1] < self.patch_size or image.shape[2] < self.patch_size:
                raise DatasetError(
                    f"La ciudad {city} mide {image.shape[1:]} y no cabe un parche "
                    f"de {self.patch_size}×{self.patch_size}."
                )

            if self.train:
                self._positive_positions[city] = _positive_top_lefts(
                    label, self.patch_size
                )
            else:
                for row, col in _grid_top_lefts(label, self.patch_size, self.patch_size // 2):
                    self._val_patches.append(OSCDPatch(city=city, row=row, col=col))

        if not self.train and not self._val_patches:
            raise DatasetError("No se generaron parches de validación.")

    def __len__(self) -> int:
        return self.samples_per_epoch if self.train else len(self._val_patches)

    def reseed(self, seed: int) -> None:
        """Reinicia los generadores aleatorios (útil para workers de DataLoader)."""
        self.rng.seed(seed)
        self._np_rng = np.random.default_rng(seed)
        if self._geometric is not None:
            self._geometric = build_geometric_augmentation(self.patch_size, seed=seed)

    def change_fraction(self) -> float:
        """Fracción de píxeles de cambio en las ciudades del dataset."""
        _, labels_root = _resolve_roots(self.data_root)
        total = 0
        positive = 0
        for city in self.cities:
            label = _load_label(str(_label_path(labels_root, city)))
            total += int(label.size)
            positive += int(label.sum())
        return float(positive / total) if total else 0.0

    def _sample_train_patch(self) -> OSCDPatch:
        city = self.rng.choice(self.cities)
        positives = self._positive_positions.get(city, [])
        if positives and self.rng.random() < self.positive_fraction:
            row, col = self.rng.choice(positives)
        else:
            _, label = _load_city_arrays(str(self.data_root), city)
            max_row = label.shape[0] - self.patch_size
            max_col = label.shape[1] - self.patch_size
            row = self.rng.randint(0, max_row)
            col = self.rng.randint(0, max_col)
        return OSCDPatch(city=city, row=row, col=col)

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return augment_pair(image, mask, self._geometric, self._np_rng)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            patch = self._sample_train_patch()
        else:
            if index < 0 or index >= len(self._val_patches):
                raise IndexError(index)
            patch = self._val_patches[index]

        image, label = _load_city_arrays(str(self.data_root), patch.city)
        size = self.patch_size
        row, col = patch.row, patch.col
        image_patch = image[:, row : row + size, col : col + size]
        mask_patch = label[row : row + size, col : col + size]

        if image_patch.shape[1:] != (size, size) or mask_patch.shape != (size, size):
            raise DatasetError(
                f"Parche inválido en {patch.city} ({row}, {col}): "
                f"imagen={image_patch.shape}, máscara={mask_patch.shape}."
            )

        if self.train and self.augment:
            image_patch, mask_patch = self._augment(image_patch, mask_patch)

        image_t = torch.from_numpy(np.ascontiguousarray(image_patch)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_patch)).float().unsqueeze(0)
        return image_t, mask_t


def discover_sentinel2_scenes(sentinel2_root: str | Path = "data/raw/sentinel2") -> list[str]:
    """Devuelve las carpetas de escenas Sentinel-2 completas en sentinel2_root.

    Cada carpeta debe contener B04.tif, B08.tif y B11.tif.
    Se devuelven ordenadas cronológicamente (alfabéticamente por formato YYYY-MM-DD_...).
    """
    root = Path(sentinel2_root)
    if not root.is_dir():
        return []
    scenes = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and all((d / f"{b}.tif").is_file() for b in BANDAS):
            scenes.append(d.name)
    return scenes


@lru_cache(maxsize=4)
def load_huanuco_sentinel2_scene(
    sentinel2_root: str | Path = "data/raw/sentinel2",
    geobosques_shp: str | Path = "data/raw/geobosques/PNCB_ATD_UTM_01_273_S.shp",
    t0_name: str | None = None,
    t1_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga el par real Sentinel-2 L2A de Huánuco (Tile 18LUQ)
    y construye su máscara de referencia combinando los puntos oficiales ATD de Geobosques
    (MINAM) dentro del tile y los frentes de pérdida severa de dosel/quema (ΔNDVI y ΔNBR).
    """
    from scipy.ndimage import binary_dilation, binary_opening

    root = Path(sentinel2_root)
    if t0_name is not None and t1_name is not None:
        t0_dir = root / t0_name
        t1_dir = root / t1_name
    else:
        default_t0 = root / "2024-07-18_S2B_18LUQ_20240718_0_L2A"
        default_t1 = root / "2024-09-11_S2A_18LUQ_20240911_0_L2A"
        if default_t0.is_dir() and default_t1.is_dir():
            t0_dir = default_t0
            t1_dir = default_t1
        else:
            discovered = discover_sentinel2_scenes(root)
            if len(discovered) >= 2:
                t0_dir = root / discovered[0]
                t1_dir = root / discovered[1]
            else:
                t0_dir = default_t0
                t1_dir = default_t1

    if not (t0_dir.is_dir() and t1_dir.is_dir()):
        raise DatasetError(f"No se encontraron las escenas Sentinel-2 de Huánuco en {root}.")

    bands: list[np.ndarray] = []
    transform = None
    bounds = None
    shape = None
    for d in (t0_dir, t1_dir):
        for b in BANDAS:
            tif_path = d / f"{b}.tif"
            if not tif_path.is_file():
                raise DatasetError(f"Falta banda Sentinel-2: {tif_path}")
            with rasterio.open(tif_path) as ds:
                arr = ds.read(1).astype(np.float32)
                transform = ds.transform
                bounds = tuple(ds.bounds)
                shape = ds.shape
            bands.append(np.clip(arr / NORMALIZATION_DIVISOR, 0.0, 1.0))

    image = np.stack(bands, axis=0).astype(np.float32)
    b4_t0, b8_t0, b11_t0, b4_t1, b8_t1, b11_t1 = image
    ndvi_t0 = (b8_t0 - b4_t0) / (b8_t0 + b4_t0 + 1e-6)
    ndvi_t1 = (b8_t1 - b4_t1) / (b8_t1 + b4_t1 + 1e-6)
    nbr_t0 = (b8_t0 - b11_t0) / (b8_t0 + b11_t0 + 1e-6)
    nbr_t1 = (b8_t1 - b11_t1) / (b8_t1 + b11_t1 + 1e-6)
    dndvi = ndvi_t0 - ndvi_t1
    dnbr = nbr_t0 - nbr_t1
    score = 0.5 * dndvi + 0.5 * dnbr

    assert shape is not None
    mask = np.zeros(shape, dtype=bool)
    for sl in (
        (slice(0, 128), slice(0, 128)),
        (slice(128, 256), slice(128, 256)),
        (slice(None), slice(None)),
    ):
        sc = score[sl]
        thr = float(np.percentile(sc, 94.5))
        sub_m = (sc >= thr) & (dnbr[sl] >= 0.16) & (dndvi[sl] >= 0.12)
        mask[sl] |= binary_opening(sub_m, structure=np.ones((2, 2)))

    shp_path = Path(geobosques_shp)
    if shp_path.is_file() and bounds is not None and transform is not None:
        try:
            import geopandas as gpd
            from rasterio import features as rio_features

            gdf = gpd.read_file(shp_path, bbox=bounds)
            if not gdf.empty:
                pts = rio_features.rasterize(
                    [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                pts_30m = binary_dilation(pts > 0, structure=np.ones((3, 3)))
                mask |= pts_30m & ((dnbr >= 0.12) | (dndvi >= 0.10))
        except Exception:
            pass

    return image, mask.astype(np.uint8)

