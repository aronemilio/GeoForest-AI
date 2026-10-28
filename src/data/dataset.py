"""Dataset de pares multitemporales Sentinel-2 L2A para GeoForest-AI Huánuco.

Carga parches de T0, T1 y máscaras binarias de Ground Truth con normalización
espectral fija (división por 10,000.0) y cálculo de pesos balanceados para el Sampler.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

NORMALIZATION_DIVISOR = 10000.0


class GeoForestDataset(Dataset):
    """Dataset para pares multitemporales Sentinel-2 L2A y máscaras de cambio.

    - Carga pares T0 y T1 concatenando sus bandas espectrales a lo largo del eje de canales: [C, H, W].
    - Normalización espectral fija: divide la reflectancia estrictamente por 10,000.0.
    - Carga máscaras de cambio Ground Truth como tensores float binarios [1, H, W].
    - Provee `get_sampler_weights()` para balancear clases en el DataLoader.

    Args:
        records: Ruta al archivo JSON de índice (o lista de diccionarios de metadatos).
        root_dir: Directorio raíz para resolver rutas relativas en el índice.
        transform: Función u objeto de transformación/aumento opcional.
        normalization_divisor: Factor de escala fijo (por defecto 10000.0).
    """

    def __init__(
        self,
        records: str | Path | Sequence[dict[str, Any]],
        root_dir: str | Path = ".",
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
        normalization_divisor: float = NORMALIZATION_DIVISOR,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.normalization_divisor = float(normalization_divisor)

        if isinstance(records, (str, Path)):
            idx_path = Path(records)
            if not idx_path.is_absolute():
                idx_path = self.root_dir / idx_path
            with open(idx_path, encoding="utf-8") as f:
                self.records: list[dict[str, Any]] = json.load(f)
        else:
            self.records = list(records)

        # Precomputar vector de etiquetas binarias por muestra
        self.has_change = np.array(
            [int(r.get("has_change", 0)) for r in self.records],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.root_dir / p

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # Modo datos en memoria (para pruebas sintéticas ultrarrápidas)
        if "t0_data" in rec and "t1_data" in rec and "mask_data" in rec:
            t0 = np.asarray(rec["t0_data"], dtype=np.float32) / self.normalization_divisor
            t1 = np.asarray(rec["t1_data"], dtype=np.float32) / self.normalization_divisor
            image_np = np.concatenate([t0, t1], axis=0)

            mask_np = np.asarray(rec["mask_data"], dtype=np.float32)
            if mask_np.ndim == 2:
                mask_np = mask_np[np.newaxis, ...]
            mask_np = (mask_np > 0).astype(np.float32)
        else:
            t0_path = self._resolve_path(rec["t0_path"])
            t1_path = self._resolve_path(rec["t1_path"])
            mask_path = self._resolve_path(rec["mask_path"])

            with rasterio.open(t0_path) as s0:
                t0 = s0.read().astype(np.float32) / self.normalization_divisor
            with rasterio.open(t1_path) as s1:
                t1 = s1.read().astype(np.float32) / self.normalization_divisor

            image_np = np.concatenate([t0, t1], axis=0)

            with rasterio.open(mask_path) as sm:
                raw_mask = sm.read()
                # Asegurar forma (1, H, W) binaria
                if raw_mask.ndim == 2:
                    raw_mask = raw_mask[np.newaxis, ...]
                mask_np = (raw_mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).float()
        mask_tensor = torch.from_numpy(mask_np).float()

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def get_sampler_weights(self, target_pos_ratio: float = 0.5) -> torch.DoubleTensor:
        """Calcula los pesos inversos por muestra para balancear clases en WeightedRandomSampler.

        Garantiza que la probabilidad marginal de extraer una muestra positiva (has_change=1)
        sea exactamente `target_pos_ratio` (por defecto 0.5 = 50% de muestras con cambio).

        Args:
            target_pos_ratio: Proporción objetivo de muestras con cambio en (0.0, 1.0).

        Returns:
            torch.DoubleTensor con los pesos normalizados por muestra.
        """
        if not 0.0 < float(target_pos_ratio) < 1.0:
            raise ValueError(f"target_pos_ratio debe estar en (0.0, 1.0), recibido: {target_pos_ratio}")

        n_total = len(self.has_change)
        if n_total == 0:
            return torch.DoubleTensor([])

        n_pos = int((self.has_change == 1).sum())
        n_neg = n_total - n_pos

        if n_pos == 0 or n_neg == 0:
            # Si no hay mezcla de clases, pesos uniformes
            weights = np.ones(n_total, dtype=np.float64) / n_total
        else:
            w_pos = float(target_pos_ratio) / n_pos
            w_neg = float(1.0 - target_pos_ratio) / n_neg
            weights = np.where(self.has_change == 1, w_pos, w_neg).astype(np.float64)

        return torch.DoubleTensor(weights)
