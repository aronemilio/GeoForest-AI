"""
Módulo de Funciones de Pérdida Avanzadas para GeoForest-AI:
- AsymmetricFocalTverskyLoss: Penalización asimétrica de Falsos Positivos (FP).
- ForestMaskedLoss: Restricción del gradiente estrictamente a cobertura boscosa continua.
- AgroforestryBoundaryOHEMLoss: Hard Negative Mining focalizado en bordes y ecotonos agroforestales.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss con penalización asimétrica de Falsos Positivos.
    
    alpha > beta para priorizar la reducción drástica de falsos positivos
    en mosaicos de vegetación y bordes de parcelas.
    gamma > 1 para concentrar el aprendizaje en píxeles difíciles (Hard Negatives).
    """
    def __init__(self, alpha: float = 0.75, beta: float = 0.25, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha      # Ponderación de Falsos Positivos
        self.beta = beta        # Ponderación de Falsos Negativos
        self.gamma = gamma      # Exponente focal para hard examples
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        if mask is not None:
            probs = probs * mask
            targets = targets * mask

        # Aplanar espacialmente por lote
        b = probs.size(0)
        p_flat = probs.view(b, -1)
        t_flat = targets.view(b, -1)

        tp = (p_flat * t_flat).sum(dim=1)
        fp = (p_flat * (1.0 - t_flat)).sum(dim=1)
        fn = ((1.0 - p_flat) * t_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)
        return focal_tversky.mean()


class ForestMaskedLoss(nn.Module):
    """
    Wrapper que enmascara cualquier función de pérdida para restringir
    el cálculo del gradiente exclusivamente a píxeles dentro de la cobertura
    boscosa continua válida (M_bosque == 1).
    """
    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, forest_mask: torch.Tensor) -> torch.Tensor:
        # forest_mask: (B, 1, H, W) binario (1 = bosque continuo, 0 = no-bosque/agua/urbano)
        mask = forest_mask.float()
        
        # Evitar división por cero si un chip tiene mínima cobertura
        valid_pixels = mask.sum()
        if valid_pixels < 1.0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if hasattr(self.base_criterion, "forward_masked"):
            return self.base_criterion.forward_masked(logits, targets, mask)
        
        # Para pérdidas a nivel de píxel (BCE píxel a píxel)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        masked_bce = (bce * mask).sum() / (valid_pixels + 1e-6)
        
        # Si la base tiene componente Tversky/Dice:
        if isinstance(self.base_criterion, AsymmetricFocalTverskyLoss):
            tversky = self.base_criterion(logits, targets, mask=mask)
            return 0.5 * masked_bce + 0.5 * tversky
            
        return masked_bce


class AgroforestryBoundaryOHEMLoss(nn.Module):
    """
    Pérdida con Online Hard Negative Mining (OHEM) y ponderación de bordes agroforestales.
    
    1. Detecta la frontera de transición agroforestal (gradiente morfológico de la máscara de bosque).
    2. Penaliza con peso extra (border_weight) los píxeles de frontera.
    3. Selecciona los píxeles negativos con mayor probabilidad de falso positivo (top-k hardest negatives).
    """
    def __init__(
        self,
        border_weight: float = 2.5,
        ohem_ratio: float = 0.25,
        alpha_tversky: float = 0.75,
        beta_tversky: float = 0.25,
        focal_gamma: float = 1.33,
    ):
        super().__init__()
        self.border_weight = border_weight
        self.ohem_ratio = ohem_ratio
        self.tversky_loss = AsymmetricFocalTverskyLoss(
            alpha=alpha_tversky, beta=beta_tversky, gamma=focal_gamma
        )

    def extract_boundary(self, forest_mask: torch.Tensor) -> torch.Tensor:
        """Extrae los bordes del bosque continuo mediante gradiente morfológico (Maxpool - Minpool)."""
        kernel_size = 3
        padding = kernel_size // 2
        dilated = F.max_pool2d(forest_mask, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-forest_mask, kernel_size=kernel_size, stride=1, padding=padding)
        boundary = (dilated - eroded).clamp(0.0, 1.0)
        return boundary

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        forest_mask: torch.Tensor,
        custom_boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits: (B, 1, H, W)
        targets: (B, 1, H, W) - Máscaras validadas por especialistas
        forest_mask: (B, 1, H, W) - Cobertura boscosa continua
        custom_boundary: (B, 1, H, W) - Opcional, máscara vectorial de borde de parcelas
        """
        probs = torch.sigmoid(logits)
        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # 1. Mapa de borde agroforestal
        if custom_boundary is not None:
            boundary_map = custom_boundary.float()
        else:
            boundary_map = self.extract_boundary(forest_mask)

        # Matriz de pesos: píxeles válidos dentro del bosque, con sobrepeso en la frontera agroforestal
        weight_map = forest_mask.clone().float()
        weight_map = weight_map + (boundary_map * (self.border_weight - 1.0))

        # 2. Hard Negative Mining (OHEM) en píxeles negativos (targets == 0)
        neg_mask = (targets == 0.0) & (forest_mask > 0.5)
        pos_mask = (targets == 1.0) & (forest_mask > 0.5)

        loss_pos = (bce_map * weight_map * pos_mask.float()).sum() / (pos_mask.float().sum() + 1e-6)

        # Seleccionar los negativos más difíciles (top-k hardest negatives)
        neg_losses = bce_map[neg_mask]
        if neg_losses.numel() > 0:
            k = max(1, int(neg_losses.numel() * self.ohem_ratio))
            hard_neg_losses, _ = torch.topk(neg_losses, k)
            loss_neg = hard_neg_losses.mean()
        else:
            loss_neg = torch.tensor(0.0, device=logits.device)

        bce_ohem = 0.5 * loss_pos + 0.5 * loss_neg

        # 3. Componente Tversky focalizado en la región forestal válida
        tversky = self.tversky_loss(logits, targets, mask=forest_mask)

        return 0.6 * bce_ohem + 0.4 * tversky
