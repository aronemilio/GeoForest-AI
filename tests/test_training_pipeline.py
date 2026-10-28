"""Pruebas unitarias de la Fase B: Dataset, Sampler, FocalDiceLoss y Pipeline de Entrenamiento.

Valida con datos sintéticos:
1. FocalDiceLoss: estabilidad numérica (sin NaN ni Inf) y gradientes válidos.
2. train_loader: muestreo balanceado con WeightedRandomSampler con proporción ~50% has_change=1.
3. val_loader: modo determinista (shuffle=False) sin sampler sobre distribución real.
4. Checkpoint y metadatos: generación de models/unet_best.pth con val_iou y métricas JSON.
5. GeoForestDataset: concatenación multiespectral [C, H, W] y normalización estricta por 10,000.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

# Configurar path del proyecto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import NORMALIZATION_DIVISOR, GeoForestDataset
from src.models.loss import FocalDiceLoss
from src.train import evaluate_epoch, train_pipeline


# ---------------------------------------------------------------------------
# Fixtures y Helpers Sintéticos
# ---------------------------------------------------------------------------
def _create_synthetic_record(idx: int, has_change: int, h: int = 64, w: int = 64, bands: int = 3) -> dict:
    """Genera registros sintéticos en memoria sin necesidad de archivos temporales en disco."""
    rng = np.random.default_rng(idx)
    # Valores de reflectancia Sentinel-2 L2A (0 a 10000)
    t0_data = rng.integers(500, 4000, size=(bands, h, w), dtype=np.int32)
    t1_data = rng.integers(500, 4000, size=(bands, h, w), dtype=np.int32)

    mask_data = np.zeros((1, h, w), dtype=np.uint8)
    if has_change:
        # 15x15 píxeles positivos de deforestación
        mask_data[0, 10:25, 10:25] = 1

    return {
        "patch_id": f"synthetic_patch_{idx}",
        "has_change": int(has_change),
        "positive_pixels": int(mask_data.sum()),
        "t0_data": t0_data,
        "t1_data": t1_data,
        "mask_data": mask_data,
    }


def _make_imbalanced_records(n_neg: int = 95, n_pos: int = 5) -> list[dict]:
    """Genera lista de registros con desbalance severo (por defecto 95% negativos, 5% positivos)."""
    records = []
    idx = 0
    for _ in range(n_neg):
        records.append(_create_synthetic_record(idx, has_change=0))
        idx += 1
    for _ in range(n_pos):
        records.append(_create_synthetic_record(idx, has_change=1))
        idx += 1
    return records


# ---------------------------------------------------------------------------
# Test 1: FocalDiceLoss - Estabilidad y Gradientes
# ---------------------------------------------------------------------------
def test_focal_dice_loss_numerics_and_gradients():
    """Valida que FocalDiceLoss opere sobre logits directos sin NaN/Inf y produzca gradientes."""
    criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, lambda_focal=1.0, lambda_dice=1.0, eps=1e-6)

    b, c, h, w = 4, 1, 32, 32
    logits = torch.randn(b, c, h, w, requires_grad=True, dtype=torch.float32)
    targets = torch.randint(0, 2, (b, c, h, w), dtype=torch.float32)

    loss = criterion(logits, targets)

    # 1. Sin valores NaN ni Inf
    assert torch.is_tensor(loss)
    assert not torch.isnan(loss).any(), "FocalDiceLoss devolvió NaN"
    assert not torch.isinf(loss).any(), "FocalDiceLoss devolvió Inf"
    assert loss.item() >= 0.0

    # 2. Gradientes válidos hacia los logits
    loss.backward()
    assert logits.grad is not None, "No se calcularon gradientes para los logits"
    assert not torch.isnan(logits.grad).any(), "Gradientes contienen NaN"
    assert not torch.isinf(logits.grad).any(), "Gradientes contienen Inf"


def test_focal_dice_loss_extreme_logits_stability():
    """Valida estabilidad numérica con logits extremos (+/- 80) y máscaras vacías o llenas."""
    criterion = FocalDiceLoss()

    # Logits extremos (muy confiados positivos y negativos) como tensor hoja (leaf tensor)
    extreme_logits = torch.tensor([[[[-80.0, 80.0], [80.0, -80.0]]]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)

    loss = criterion(extreme_logits, targets)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()
    assert extreme_logits.grad is not None
    assert not torch.isnan(extreme_logits.grad).any()

    # Caso borde: todo ceros (sin cambio en todo el lote)
    zeros_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    all_zeros_target = torch.zeros(2, 1, 16, 16)
    loss_zeros = criterion(zeros_logits, all_zeros_target)
    assert not torch.isnan(loss_zeros) and not torch.isinf(loss_zeros)
    loss_zeros.backward()
    assert not torch.isnan(zeros_logits.grad).any()


# ---------------------------------------------------------------------------
# Test 2: train_loader y WeightedRandomSampler (~50% positivos)
# ---------------------------------------------------------------------------
def test_train_loader_balanced_sampling():
    """Comprueba que el sampler balancee los lotes para alcanzar ~50% de muestras positivas."""
    torch.manual_seed(42)
    # Dataset con 95 negativos y 5 positivos (5% positivos originalmente)
    records = _make_imbalanced_records(n_neg=95, n_pos=5)
    dataset = GeoForestDataset(records)

    weights = dataset.get_sampler_weights(target_pos_ratio=0.5)
    assert len(weights) == 100
    # Los pesos de las muestras positivas deben ser sustancialmente mayores
    pos_indices = np.where(dataset.has_change == 1)[0]
    neg_indices = np.where(dataset.has_change == 0)[0]
    assert weights[pos_indices[0]] > weights[neg_indices[0]]

    # Extraer 2400 muestras en lotes de 16
    num_samples = 2400
    sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
    train_loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0)

    total_samples = 0
    pos_samples = 0
    for _, masks in train_loader:
        # has_change == 1 si el tensor de máscara contiene algún píxel positivo
        has_pos = (masks.sum(dim=(1, 2, 3)) > 0).long()
        pos_samples += int(has_pos.sum().item())
        total_samples += masks.size(0)

    empirical_ratio = pos_samples / total_samples
    # Con target 0.50 y 2400 muestras, la proporción debe estar cercana a 0.50 (+/- 0.08)
    assert 0.42 <= empirical_ratio <= 0.58, (
        f"Proporción de positivos {empirical_ratio:.3f} alejada del 50% esperado"
    )


# ---------------------------------------------------------------------------
# Test 3: val_loader Determinista sin Sampler (shuffle=False)
# ---------------------------------------------------------------------------
def test_val_loader_deterministic_no_sampler():
    """Valida que val_loader itere en modo determinista sobre la distribución real."""
    records = _make_imbalanced_records(n_neg=20, n_pos=5)
    dataset = GeoForestDataset(records)

    # val_loader no debe tener WeightedRandomSampler
    val_loader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=0)

    assert isinstance(val_loader.sampler, SequentialSampler), (
        "val_loader debe usar SequentialSampler determinista y no un Sampler ponderado"
    )

    # Iterar dos veces y verificar identidad estricta de orden y contenido
    pass1 = [(img.clone(), msk.clone()) for img, msk in val_loader]
    pass2 = [(img.clone(), msk.clone()) for img, msk in val_loader]

    assert len(pass1) == len(pass2) == 5
    for (img1, m1), (img2, m2) in zip(pass1, pass2):
        assert torch.equal(img1, img2), "val_loader no es determinista entre épocas"
        assert torch.equal(m1, m2), "Máscaras difieren en iteración determinista"

    # Verificar que conserva la proporción real original (5 / 25 = 20%)
    total_val_pos = sum((m.sum(dim=(1, 2, 3)) > 0).sum().item() for _, m in pass1)
    assert total_val_pos == 5


# ---------------------------------------------------------------------------
# Test 4: Pipeline de Entrenamiento y Checkpoint con val_iou
# ---------------------------------------------------------------------------
def test_training_pipeline_and_checkpoint_verification(tmp_path):
    """Comprueba el ciclo de entrenamiento y la metadata de val_iou en models/unet_best.pth."""
    train_records = _make_imbalanced_records(n_neg=30, n_pos=10)
    val_records = _make_imbalanced_records(n_neg=15, n_pos=5)

    train_ds = GeoForestDataset(train_records)
    val_ds = GeoForestDataset(val_records)

    checkpoint_file = tmp_path / "models/unet_best.pth"
    metrics_file = tmp_path / "models/training_metrics.json"

    result = train_pipeline(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=2,
        batch_size=4,
        lr=1e-4,
        weight_decay=1e-4,
        patience=12,
        device="cpu",
        checkpoint_path=checkpoint_file,
        metrics_path=metrics_file,
        num_workers=0,
        pin_memory=False,
        base_features=8,  # Red ligera para prueba rápida
    )

    # 1. Existencia de archivos de entrega
    assert checkpoint_file.exists(), f"No se generó el checkpoint en {checkpoint_file}"
    assert metrics_file.exists(), f"No se generó el archivo de métricas en {metrics_file}"

    # 2. Verificación estricta de metadata en el checkpoint .pth
    ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    assert "val_iou" in ckpt, "El checkpoint no contiene la metadata 'val_iou'"
    assert "model_state_dict" in ckpt, "El checkpoint no contiene 'model_state_dict'"
    assert "epoch" in ckpt, "El checkpoint no contiene 'epoch'"
    assert isinstance(ckpt["val_iou"], float)
    assert 0.0 <= ckpt["val_iou"] <= 1.0

    # 3. Verificación del JSON de métricas
    with open(metrics_file, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "best_val_iou" in metrics
    assert metrics["best_val_iou"] == ckpt["val_iou"]
    assert len(metrics["history"]) == 2


# ---------------------------------------------------------------------------
# Test 5: GeoForestDataset - Normalización Fija y Formato de Entrada
# ---------------------------------------------------------------------------
def test_geoforest_dataset_normalization_and_channels():
    """Valida concatenación [C, H, W] y división estricta por 10,000.0."""
    rec = _create_synthetic_record(idx=0, has_change=1, h=32, w=32, bands=3)
    dataset = GeoForestDataset([rec])

    image, mask = dataset[0]

    # Con 3 bandas en T0 y 3 bandas en T1 -> 6 canales
    assert image.shape == (6, 32, 32)
    assert mask.shape == (1, 32, 32)

    # Comprobar normalización estricta por 10,000.0
    expected_t0_channel0 = rec["t0_data"][0].astype(np.float32) / NORMALIZATION_DIVISOR
    assert np.allclose(image[0].numpy(), expected_t0_channel0, atol=1e-6)

    # Máscara estrictamente binaria {0.0, 1.0}
    unique_vals = set(np.unique(mask.numpy()))
    assert unique_vals <= {0.0, 1.0}
