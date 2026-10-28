"""Calibración probabilística post-hoc (Temperature Scaling) para GeoForest-AI Huánuco.

Corrige la sobreconfianza de la red neuronal mediante un parámetro escalar óptimo T
ajustado sobre la partición de validación independiente sin alterar el orden ni el
ranking (AUROC/AP) ni las métricas de segmentación dura en el umbral p = 0.5.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Calibrador de temperatura post-hoc (Guo et al., 2017) optimizado para producción.

    Ajusta una única temperatura escalar T > 0 sobre los logits del modelo:
        p = sigmoid(logits / T)

    Propiedades:
    - Conserva estrictamente el ordenamiento y el ranking de predicciones (AUROC/AP invariantes).
    - Preserva de forma exacta las predicciones binarias en el umbral p = 0.5 (logits = 0).
    - Minimiza conjuntamente las reglas de puntuación estrictamente propias (NLL + Brier Score)
      mediante L-BFGS con búsqueda lineal Strong Wolfe sobre validación independiente.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError(f"initial_temperature debe ser > 0; recibido {initial_temperature}")
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica el escalado de temperatura a los logits directos."""
        return logits / self.temperature

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Devuelve las probabilidades calibradas en [0, 1]."""
        return torch.sigmoid(self.forward(logits))

    def fit_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.5,
        brier_weight: float = 1.0,
    ) -> float:
        """Optimiza T minimizando BCEWithLogitsLoss (+ Brier Score) usando L-BFGS con Strong Wolfe.

        Args:
            logits: Tensor de logits acumulados (N, 1, H, W) o (N,).
            targets: Tensor de etiquetas binarias acumuladas con forma coincidente.
            max_iter: Número máximo de iteraciones para L-BFGS.
            lr: Tasa de aprendizaje del optimizador.
            brier_weight: Peso del término Brier (regla de puntuación estrictamente propia)
                para garantizar convergencia óptima de Brier y NLL en producción.

        Returns:
            Valor final de temperatura T como float.
        """
        device = logits.device
        self.to(device)

        targets_f = targets.float().view_as(logits)
        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.LBFGS(
            [self.temperature],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            # Asegurar T estrictamente positivo durante la búsqueda lineal
            t = torch.clamp(self.temperature, min=1e-3, max=100.0)
            scaled_logits = logits / t
            nll_loss = criterion(scaled_logits, targets_f)
            if brier_weight > 0.0:
                probs = torch.sigmoid(scaled_logits)
                brier_loss = torch.mean((probs - targets_f) ** 2)
                loss = nll_loss + float(brier_weight) * brier_loss
            else:
                loss = nll_loss
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-3, max=100.0)

        fitted_temp = float(self.temperature.item())
        log.info("Temperatura óptima ajustada con L-BFGS (strong_wolfe): T = %.4f", fitted_temp)
        return fitted_temp

    def fit(
        self,
        val_loader: DataLoader,
        model: nn.Module,
        device: str | torch.device = "cpu",
        max_iter: int = 100,
        lr: float = 0.5,
        brier_weight: float = 1.0,
    ) -> float:
        """Optimiza T sobre el conjunto de validación manteniendo los pesos de la U-Net congelados.

        Args:
            val_loader: DataLoader de validación espacialmente disjunto (shuffle=False).
            model: U-Net u otro modelo en evaluación.
            device: Dispositivo de cómputo.
            max_iter: Iteraciones máximas de L-BFGS.
            lr: Tasa de aprendizaje.
            brier_weight: Peso del término Brier en el objetivo compuesto.

        Returns:
            Valor de temperatura óptimo.
        """
        dev = torch.device(device)
        model.eval()

        # 1. Congelar estrictamente todos los parámetros de la U-Net
        for param in model.parameters():
            param.requires_grad = False

        logits_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []

        # 2. Pase hacia adelante único para recolectar logits de validación
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(dev)
                out_logits = model(images)
                logits_list.append(out_logits.detach().cpu())
                targets_list.append(masks.detach().cpu())

        if not logits_list:
            raise ValueError("El val_loader proporcionado para calibración está vacío.")

        all_logits = torch.cat(logits_list, dim=0).to(dev)
        all_targets = torch.cat(targets_list, dim=0).to(dev)

        return self.fit_logits(
            all_logits,
            all_targets,
            max_iter=max_iter,
            lr=lr,
            brier_weight=brier_weight,
        )


class CalibratedUNetWrapper(nn.Module):
    """Wrapper de inferencia para exportar la U-Net a ONNX con calibración integrada.

    Sustituye cualquier heurística manual en la interfaz integrando directamente:
        return torch.sigmoid(logits / T)
    """

    def __init__(self, base_model: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        self.base_model = base_model
        if temperature <= 0:
            raise ValueError(f"temperature debe ser > 0; recibido {temperature}")
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return torch.sigmoid(logits / self.temperature)


# ---------------------------------------------------------------------------
# Métricas de Calibración y Segmentación: ECE-15, MCE-15, Brier, BSS, IoU, F1
# ---------------------------------------------------------------------------
def compute_ece(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> float:
    """Calcula el Expected Calibration Error (ECE) binario con n_bins bins uniformes.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    n_total = len(p)
    if n_total == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p > bin_lower) & (p <= bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            accuracy_in_bin = float(np.mean(y[in_bin]))
            confidence_in_bin = float(np.mean(p[in_bin]))
            ece += (bin_size / n_total) * abs(accuracy_in_bin - confidence_in_bin)

    return float(ece)


def compute_mce(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    n_bins: int = 15,
) -> float:
    """Calcula el Maximum Calibration Error (MCE) binario sobre bins con al menos 30 píxeles."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    if len(p) == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p > bin_lower) & (p <= bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size >= 30:
            accuracy_in_bin = float(np.mean(y[in_bin]))
            confidence_in_bin = float(np.mean(p[in_bin]))
            mce = max(mce, abs(accuracy_in_bin - confidence_in_bin))

    return float(mce)


def compute_brier_score(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> float:
    """Calcula el Brier Score (error cuadrático medio entre probabilidad y verdad fundamental).

    Brier = mean((p - y)^2)
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    if len(p) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def compute_brier_skill_score(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> tuple[float, float]:
    """Calcula el Brier Score climatológico de referencia (BS_ref) y el Brier Skill Score (BSS).

    BS_ref = y_bar * (1 - y_bar)
    BSS = 1 - (BS / BS_ref)
    Un BSS > 0 certifica que las probabilidades calibradas superan a la predicción constante
    de la tasa base marginal.
    """
    if isinstance(targets, torch.Tensor):
        y = targets.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        y = np.asarray(targets, dtype=np.float64).ravel()

    if len(y) == 0:
        return 0.0, 0.0

    y_bar = float(np.mean(y))
    bs_ref = y_bar * (1.0 - y_bar)
    bs = compute_brier_score(probs, y)
    if bs_ref <= 1e-12:
        return 0.0, 0.0
    bss = 1.0 - (bs / bs_ref)
    return float(bs_ref), float(bss)


def compute_segmentation_metrics(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Calcula IoU, F1, Precisión y Recall sobre la clase positiva (deforestación = 1)."""
    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().numpy().astype(np.float64).ravel()
    else:
        p = np.asarray(probs, dtype=np.float64).ravel()

    if isinstance(targets, torch.Tensor):
        y = (targets.detach().cpu().numpy().ravel() > 0.5).astype(np.float64)
    else:
        y = (np.asarray(targets).ravel() > 0.5).astype(np.float64)

    preds = (p >= float(threshold)).astype(np.float64)
    tp = float(np.sum(preds * y))
    fp = float(np.sum(preds * (1.0 - y)))
    fn = float(np.sum((1.0 - preds) * y))

    iou = (tp + eps) / (tp + fp + fn + eps) if (tp + fp + fn) > 0 else 1.0
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps) if (2.0 * tp + fp + fn) > 0 else 1.0
    precision = (tp + eps) / (tp + fp + eps) if (tp + fp) > 0 else 1.0
    recall = (tp + eps) / (tp + fn + eps) if (tp + fn) > 0 else 1.0

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_calibration(
    logits: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    temperature: float = 1.0,
    n_bins: int = 15,
) -> dict[str, float]:
    """Evalúa métricas completas de calibración y segmentación antes y después de aplicar T."""
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
    else:
        logits_np = np.asarray(logits, dtype=np.float64)

    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy().astype(np.float64)
    else:
        targets_np = np.asarray(targets, dtype=np.float64)

    # Probabilidades sin calibrar (T = 1.0 implícito)
    probs_uncal = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -50.0, 50.0)))
    # Probabilidades calibradas con T óptimo
    probs_cal = 1.0 / (1.0 + np.exp(-np.clip(logits_np / float(temperature), -50.0, 50.0)))

    ece_before = compute_ece(probs_uncal, targets_np, n_bins=n_bins)
    ece_after = compute_ece(probs_cal, targets_np, n_bins=n_bins)
    mce_before = compute_mce(probs_uncal, targets_np, n_bins=n_bins)
    mce_after = compute_mce(probs_cal, targets_np, n_bins=n_bins)
    brier_before = compute_brier_score(probs_uncal, targets_np)
    brier_after = compute_brier_score(probs_cal, targets_np)
    bs_ref, bss_before = compute_brier_skill_score(probs_uncal, targets_np)
    _, bss_after = compute_brier_skill_score(probs_cal, targets_np)
    seg = compute_segmentation_metrics(probs_cal, targets_np, threshold=0.5)

    return {
        "temperature": float(temperature),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "mce_before": float(mce_before),
        "mce_after": float(mce_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "brier_reference": float(bs_ref),
        "brier_skill_score_before": float(bss_before),
        "brier_skill_score_after": float(bss_after),
        "val_iou": float(seg["iou"]),
        "val_f1": float(seg["f1"]),
        "val_precision": float(seg["precision"]),
        "val_recall": float(seg["recall"]),
    }
