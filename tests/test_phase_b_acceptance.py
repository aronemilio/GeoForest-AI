"""Test de aceptación de la Fase B.

Aprueba, con datos sintéticos y rutas temporales, la configuración de:
1. Dataset: T0/T1 concatenados como ``[C, H, W]``, normalización estricta
   ``/10000`` y máscara binaria ``[1, H, W]``.
2. Sampler: ``WeightedRandomSampler`` solo en train, con masa marginal 50 %
   para ``has_change == 1``; validación secuencial y determinista.
3. Pérdida híbrida: ``FocalDiceLoss`` sobre logits, estable, con gradientes
   finitos y casos extremos (máscaras vacías/llenas).
4. Pipeline: AdamW, CosineAnnealingLR, early stopping por ``val_iou``,
   checkpoint con ``val_iou`` y JSON de métricas.

No escribe en ``models/`` ni toca datos reales: todos los artefactos van a
``tmp_path`` y los registros se construyen en memoria.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.train as train_module
from src.data.dataset import NORMALIZATION_DIVISOR, GeoForestDataset
from src.models.loss import FocalDiceLoss


# ---------------------------------------------------------------------------
# Datos sintéticos deterministas
# ---------------------------------------------------------------------------
def _synthetic_record(
    idx: int,
    has_change: int,
    *,
    h: int = 32,
    w: int = 32,
    bands: int = 3,
) -> dict[str, Any]:
    """Registro en memoria; no depende de GeoTIFF ni del dataset real."""
    rng = np.random.default_rng(10_000 + idx)
    t0 = rng.integers(0, 10_001, size=(bands, h, w), dtype=np.int32)
    t1 = rng.integers(0, 10_001, size=(bands, h, w), dtype=np.int32)

    # Máscara 2D con valores no binarios a propósito: el Dataset debe binarizarla.
    mask = np.zeros((h, w), dtype=np.uint8)
    if has_change:
        mask[h // 4 : h // 2, w // 4 : w // 2] = 2

    return {
        "patch_id": f"synthetic_{idx}",
        "has_change": int(has_change),
        "positive_pixels": int((mask > 0).sum()),
        "t0_data": t0,
        "t1_data": t1,
        "mask_data": mask,
    }


def _make_records(n_neg: int, n_pos: int, **kwargs: Any) -> list[dict[str, Any]]:
    records = [
        _synthetic_record(i, 0, **kwargs) for i in range(n_neg)
    ]
    records.extend(
        _synthetic_record(n_neg + i, 1, **kwargs) for i in range(n_pos)
    )
    return records


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
def test_dataset_contract_channels_normalization_and_binary_mask() -> None:
    rec = _synthetic_record(0, has_change=1, h=32, w=32, bands=3)
    dataset = GeoForestDataset([rec])

    assert NORMALIZATION_DIVISOR == 10_000.0
    assert dataset.normalization_divisor == 10_000.0

    image, mask = dataset[0]

    # 3 bandas T0 + 3 bandas T1 -> [6, H, W]
    assert image.shape == (6, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32

    # Normalización estricta por 10,000.0, canal por canal.
    t0_expected = torch.from_numpy(rec["t0_data"].astype(np.float32) / 10_000.0)
    t1_expected = torch.from_numpy(rec["t1_data"].astype(np.float32) / 10_000.0)
    assert torch.allclose(image[:3], t0_expected, atol=1e-7)
    assert torch.allclose(image[3:], t1_expected, atol=1e-7)
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0

    # La máscara de entrada tiene valores {0, 2}; la salida debe ser binaria.
    assert set(np.unique(mask.numpy()).tolist()) <= {0.0, 1.0}
    assert int(mask.sum().item()) == int(rec["positive_pixels"])
    assert dataset.has_change.tolist() == [1]


# ---------------------------------------------------------------------------
# 2. Sampler
# ---------------------------------------------------------------------------
def test_sampler_marginal_balance_and_weighted_sampler() -> None:
    records = _make_records(n_neg=95, n_pos=5)
    dataset = GeoForestDataset(records)

    weights = dataset.get_sampler_weights(target_pos_ratio=0.5)
    assert weights.dtype == torch.float64
    assert len(weights) == 100
    assert float(weights.sum().item()) == pytest.approx(1.0)

    # La masa total de cada clase debe ser exactamente 0.5.
    pos_mass = float(weights[dataset.has_change == 1].sum().item())
    neg_mass = float(weights[dataset.has_change == 0].sum().item())
    assert pos_mass == pytest.approx(0.5)
    assert neg_mass == pytest.approx(0.5)

    # Comportamiento empírico del sampler real, no solo de los pesos.
    sampler = WeightedRandomSampler(weights=weights, num_samples=10_000, replacement=True)
    assert isinstance(sampler, WeightedRandomSampler)
    torch.manual_seed(1234)
    sampled = torch.tensor(list(sampler), dtype=torch.long)
    empirical_pos_ratio = float(dataset.has_change[sampled.numpy()].mean())
    assert 0.47 <= empirical_pos_ratio <= 0.53, (
        f"El sampler produjo {empirical_pos_ratio:.3f} de positivos; se esperaba ~0.50"
    )

    with pytest.raises(ValueError):
        dataset.get_sampler_weights(target_pos_ratio=0.0)
    with pytest.raises(ValueError):
        dataset.get_sampler_weights(target_pos_ratio=1.0)


# ---------------------------------------------------------------------------
# 3. Pérdida híbrida
# ---------------------------------------------------------------------------
def test_focal_dice_loss_is_hybrid_stable_and_differentiable() -> None:
    criterion = FocalDiceLoss()
    assert criterion.alpha == 0.25
    assert criterion.gamma == 2.0
    assert criterion.lambda_focal == 1.0
    assert criterion.lambda_dice == 1.0

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 4:10, 4:10] = 1.0

    loss = criterion(logits, targets)
    assert torch.isfinite(loss).item()
    assert float(loss.item()) >= 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all().item()
    assert float(logits.grad.abs().sum().item()) > 0.0

    # La pérdida combinada debe ser exactamente la suma de sus dos componentes.
    with torch.no_grad():
        focal_only = FocalDiceLoss(lambda_dice=0.0)(logits.detach(), targets)
        dice_only = FocalDiceLoss(lambda_focal=0.0)(logits.detach(), targets)
        combined = criterion(logits.detach(), targets)
    assert float(combined.item()) == pytest.approx(
        float(focal_only.item()) + float(dice_only.item()), abs=1e-6
    )

    # Casos extremos: máscara vacía, máscara llena y logits muy confiados.
    for target in (
        torch.zeros(2, 1, 8, 8),
        torch.ones(2, 1, 8, 8),
    ):
        edge_logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        edge_loss = criterion(edge_logits, target)
        assert torch.isfinite(edge_loss).item()
        edge_loss.backward()
        assert edge_logits.grad is not None
        assert torch.isfinite(edge_logits.grad).all().item()

    extreme_logits = torch.tensor(
        [[[[-80.0, 80.0], [80.0, -80.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    extreme_targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)
    extreme_loss = criterion(extreme_logits, extreme_targets)
    assert torch.isfinite(extreme_loss).item()
    extreme_loss.backward()
    assert extreme_logits.grad is not None
    assert torch.isfinite(extreme_logits.grad).all().item()


# ---------------------------------------------------------------------------
# 4. Pipeline de entrenamiento
# ---------------------------------------------------------------------------
def test_train_pipeline_uses_expected_configuration_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_ds = GeoForestDataset(_make_records(n_neg=12, n_pos=4))
    val_ds = GeoForestDataset(_make_records(n_neg=6, n_pos=2))

    # Capturar los DataLoader reales construidos por train_pipeline.
    real_dataloader = DataLoader
    loader_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    created_loaders: list[DataLoader] = []

    def recording_dataloader(*args: Any, **kwargs: Any) -> DataLoader:
        loader_calls.append((args, kwargs))
        loader = real_dataloader(*args, **kwargs)
        created_loaders.append(loader)
        return loader

    monkeypatch.setattr(train_module, "DataLoader", recording_dataloader)

    # Capturar AdamW y CosineAnnealingLR sin sustituir su comportamiento.
    real_adamw = torch.optim.AdamW
    adamw_calls: list[dict[str, Any]] = []

    def recording_adamw(params: Any, **kwargs: Any) -> torch.optim.AdamW:
        adamw_calls.append(dict(kwargs))
        return real_adamw(params, **kwargs)

    monkeypatch.setattr(train_module.optim, "AdamW", recording_adamw)

    real_cosine = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_calls: list[dict[str, Any]] = []

    def recording_cosine(optimizer: Any, **kwargs: Any) -> Any:
        scheduler_calls.append(dict(kwargs))
        return real_cosine(optimizer, **kwargs)

    monkeypatch.setattr(
        train_module.optim.lr_scheduler,
        "CosineAnnealingLR",
        recording_cosine,
    )

    checkpoint_file = tmp_path / "models" / "unet_best.pth"
    metrics_file = tmp_path / "models" / "training_metrics.json"

    result = train_module.train_pipeline(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        weight_decay=2e-4,
        patience=2,
        device="cpu",
        checkpoint_path=checkpoint_file,
        metrics_path=metrics_file,
        num_workers=0,
        pin_memory=False,
        base_features=8,
    )

    # --- Sampler solo en train; val secuencial y determinista ---
    assert len(loader_calls) == 2
    assert len(created_loaders) == 2
    train_loader, val_loader = created_loaders

    train_kwargs = loader_calls[0][1]
    val_kwargs = loader_calls[1][1]

    assert isinstance(train_loader.sampler, WeightedRandomSampler)
    assert train_loader.sampler.num_samples == len(train_ds)
    assert "sampler" in train_kwargs

    assert isinstance(val_loader.sampler, SequentialSampler)
    assert "sampler" not in val_kwargs
    assert val_kwargs["shuffle"] is False

    val_pass_1 = [mask.clone() for _, mask in val_loader]
    val_pass_2 = [mask.clone() for _, mask in val_loader]
    assert len(val_pass_1) == len(val_pass_2)
    for mask_1, mask_2 in zip(val_pass_1, val_pass_2):
        assert torch.equal(mask_1, mask_2)

    # --- AdamW y CosineAnnealingLR ---
    assert len(adamw_calls) == 1
    assert adamw_calls[0]["lr"] == pytest.approx(1e-3)
    assert adamw_calls[0]["weight_decay"] == pytest.approx(2e-4)

    assert len(scheduler_calls) == 1
    assert scheduler_calls[0]["T_max"] == 2
    assert scheduler_calls[0]["eta_min"] == pytest.approx(1e-6)

    # --- Checkpoint y JSON de métricas ---
    assert checkpoint_file.exists()
    assert metrics_file.exists()

    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    assert isinstance(ckpt["val_iou"], float)
    assert 0.0 <= ckpt["val_iou"] <= 1.0
    assert ckpt["in_channels"] == 6
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt

    assert metrics["best_epoch"] == ckpt["epoch"]
    assert metrics["best_val_iou"] == pytest.approx(float(ckpt["val_iou"]))
    assert metrics["total_epochs"] == 2
    assert len(metrics["history"]) == 2
    assert all("val_iou" in row and "lr" in row for row in metrics["history"])
    assert max(row["val_iou"] for row in metrics["history"]) == pytest.approx(
        float(result["best_val_iou"]), abs=2e-5
    )


def test_early_stopping_is_driven_by_val_iou(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Si val_iou no mejora, debe parar aunque val_loss baje y val_f1 suba."""
    train_ds = GeoForestDataset(_make_records(n_neg=6, n_pos=2))
    val_ds = GeoForestDataset(_make_records(n_neg=3, n_pos=1))

    # (val_loss, val_iou, val_f1): loss mejora y f1 sube, pero IoU baja.
    responses = [
        (0.60, 0.30, 0.10),
        (0.50, 0.20, 0.90),
        (0.40, 0.10, 0.95),
        (0.30, 0.05, 0.96),
        (0.20, 0.04, 0.97),
        (0.10, 0.03, 0.98),
    ]
    calls: list[int] = []

    def fake_evaluate(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        calls.append(1)
        return responses[len(calls) - 1]

    monkeypatch.setattr(train_module, "evaluate_epoch", fake_evaluate)

    checkpoint_file = tmp_path / "models" / "unet_best.pth"
    metrics_file = tmp_path / "models" / "training_metrics.json"

    result = train_module.train_pipeline(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=6,
        batch_size=4,
        patience=2,
        device="cpu",
        checkpoint_path=checkpoint_file,
        metrics_path=metrics_file,
        num_workers=0,
        pin_memory=False,
        base_features=8,
    )

    # Época 1: mejor IoU. Épocas 2 y 3 sin mejora -> early stop en la 3.
    assert len(calls) == 3
    assert result["total_epochs"] == 3
    assert len(result["history"]) == 3
    assert result["best_epoch"] == 1
    assert float(result["best_val_iou"]) == pytest.approx(0.30)
    assert float(result["best_val_loss"]) == pytest.approx(0.60)
    assert float(result["best_val_f1"]) == pytest.approx(0.10)
    assert [row["val_iou"] for row in result["history"]] == pytest.approx(
        [0.30, 0.20, 0.10]
    )

    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert ckpt["epoch"] == 1
    assert float(ckpt["val_iou"]) == pytest.approx(0.30)
    assert metrics["best_epoch"] == 1
    assert float(metrics["best_val_iou"]) == pytest.approx(0.30)
    assert metrics["total_epochs"] == 3


def test_phase_b_default_configuration() -> None:
    """Fija los valores por defecto que definen la configuración de Fase B."""
    assert NORMALIZATION_DIVISOR == 10_000.0

    loss = FocalDiceLoss()
    assert loss.alpha == 0.25
    assert loss.gamma == 2.0
    assert loss.lambda_focal == 1.0
    assert loss.lambda_dice == 1.0

    params = inspect.signature(train_module.train_pipeline).parameters
    assert params["target_pos_ratio"].default == 0.5
    assert params["lr"].default == 1e-4
    assert params["weight_decay"].default == 1e-4
    assert params["patience"].default == 12
