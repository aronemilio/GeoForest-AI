"""Función de pérdida híbrida FocalDiceLoss para GeoForest-AI Huánuco.

Optimización conjunta de Focal Loss y Dice Loss operando directamente sobre logits
para resolver el desbalance extremo de clases en segmentación de deforestación.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """Pérdida híbrida Focal + Dice para segmentación con desbalance extremo de clases.

    Opera directamente sobre logits no normalizados para máxima estabilidad numérica.

    Fórmula:
        L_total = lambda_focal * L_focal + lambda_dice * L_dice

    Parámetros:
        alpha (float): Factor de ponderación para la clase positiva (deforestación). Default: 0.25.
        gamma (float): Exponente focal para suprimir pérdidas en ejemplos fáciles. Default: 2.0.
        lambda_focal (float): Ponderación del componente Focal Loss. Default: 1.0.
        lambda_dice (float): Ponderación del componente Dice Loss. Default: 1.0.
        eps (float): Término de suavizado numérico contra divisiones por cero. Default: 1e-6.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_focal = float(lambda_focal)
        self.lambda_dice = float(lambda_dice)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida conjunta Focal + Dice sobre logits.

        Args:
            logits: Tensor (B, 1, H, W) o (B, H, W) con logits directos del modelo.
            targets: Tensor con misma forma o broadcastable con valores binarios en {0, 1}.

        Returns:
            Scalar tensor con la pérdida combinada.
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)

        targets = targets.float()

        # --- 1. Focal Loss numéricamente estable sobre logits ---
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)

        # Probabilidad de la clase verdadera p_t
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)

        # Ponderación alpha_t
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        # Factor modulador (1 - p_t)^gamma
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=0.0, max=1.0), self.gamma)
        focal_loss = (focal_weight * bce_loss).mean()

        # --- 2. Dice Loss sobre probabilidades con eps ---
        b = logits.size(0)
        p_flat = probs.view(b, -1)
        t_flat = targets.view(b, -1)

        intersection = (p_flat * t_flat).sum(dim=1)
        cardinality = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = (1.0 - dice_score).mean()

        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss
