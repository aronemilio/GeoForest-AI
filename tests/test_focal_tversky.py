"""Pruebas unitarias para FocalTverskyLoss y pipeline de entrenamiento."""

import numpy as np
import pytest
import torch

from src.train import FocalTverskyLoss


def test_focal_tversky_loss_initialization() -> None:
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.333, smooth=1.0)
    assert loss_fn.alpha == 0.3
    assert loss_fn.beta == 0.7
    assert loss_fn.gamma == 1.333
    assert loss_fn.smooth == 1.0

    with pytest.raises(ValueError, match="alpha \\+ beta debe ser > 0"):
        FocalTverskyLoss(alpha=0.0, beta=0.0)


def test_focal_tversky_perfect_prediction() -> None:
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)
    # Logits muy altos para target=1 y muy bajos para target=0
    logits = torch.tensor([[[[20.0, -20.0], [-20.0, 20.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    val = loss_fn(logits, targets)
    assert val.item() < 1e-4


def test_focal_tversky_penalizes_false_negatives_more_than_false_positives() -> None:
    # Con beta=0.7 > alpha=0.3, cometer Falsos Negativos (omitir deforestación)
    # reduce el índice Tversky y penaliza más la pérdida que cometer Falsos Positivos.
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0, smooth=1.0)

    # Base: 10 píxeles verdaderos positivos (TP=10)
    # Caso A (10 FN): target tiene 20 unos, el modelo solo predice 10
    targets_a = torch.zeros(1, 1, 1, 20)
    targets_a[..., :20] = 1.0  # 20 positivos reales
    logits_a = torch.full((1, 1, 1, 20), -20.0)
    logits_a[..., :10] = 20.0  # detecta 10 (TP=10, FN=10, FP=0)
    loss_fn_penalty = loss_fn(logits_a, targets_a).item()

    # Caso B (10 FP): target tiene 10 unos, el modelo predice 20
    targets_b = torch.zeros(1, 1, 1, 20)
    targets_b[..., :10] = 1.0  # 10 positivos reales
    logits_b = torch.full((1, 1, 1, 20), -20.0)
    logits_b[..., :20] = 20.0  # predice 20 (TP=10, FN=0, FP=10)
    loss_fp_penalty = loss_fn(logits_b, targets_b).item()

    # FN debe tener una pérdida mayor que FP:
    # TI(A) = (10+1)/(10 + 0.7*10 + 1) = 11/18 = 0.611 -> Loss = 0.389
    # TI(B) = (10+1)/(10 + 0.3*10 + 1) = 11/14 = 0.786 -> Loss = 0.214
    assert loss_fn_penalty > loss_fp_penalty
    assert np.isclose(loss_fn_penalty, 1.0 - 11.0 / 18.0, atol=1e-3)
    assert np.isclose(loss_fp_penalty, 1.0 - 11.0 / 14.0, atol=1e-3)


def test_focal_tversky_backward_gradients() -> None:
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.333)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 16, 16)).float()

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
