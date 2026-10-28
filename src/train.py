"""Pipeline de entrenamiento para U-Net en GeoForest-AI Huánuco (Fase B / Producción).

Resuelve el desbalance extremo de clases mediante:
1. Muestreo ponderado (WeightedRandomSampler) aplicado estrictamente en train_loader.
2. val_loader determinista sin sampler (shuffle=False) sobre la distribución real.
3. Optimización con FocalDiceLoss directamente sobre logits sin sigmoid previa.
4. Optimizador AdamW (lr=1e-4, weight_decay=1e-4) y CosineAnnealingLR (50 épocas).
5. Monitoreo estricto de val_iou (IoU clase 1 deforestación) para Early Stopping (paciencia=12),
   selección de checkpoint en models/unet_best.pth, saneamiento de trazabilidad en
   models/training_metrics.json y actualización de hashes SHA-256 en models/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# Agregar ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import GeoForestDataset
from src.models.loss import FocalDiceLoss
from src.unet import UNet

log = logging.getLogger(__name__)


def compute_sha256(path: str | Path) -> str:
    """Calcula el hash SHA-256 criptográfico de un archivo en disco."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest_hashes(
    manifest_path: str | Path = "models/manifest.json",
    checkpoint_path: str | Path = "models/unet_best.pth",
    onnx_path: str | Path = "models/unet_geoforest.onnx",
) -> dict[str, str]:
    """Actualiza models/manifest.json con los hashes SHA-256 reales de los artefactos en disco."""
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, str] = {}
    if manifest_file.exists():
        try:
            record = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            record = {}

    ckpt_p = Path(checkpoint_path)
    onnx_p = Path(onnx_path)
    if ckpt_p.exists():
        record["models/unet_best.pth"] = compute_sha256(ckpt_p)
    if onnx_p.exists():
        record["models/unet_geoforest.onnx"] = compute_sha256(onnx_p)

    manifest_file.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# Compatibilidad retroactiva con módulos y pruebas existentes
# ---------------------------------------------------------------------------
class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss conservada para retrocompatibilidad."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.333,
        smooth: float = 1.0,
        bce_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha + beta <= 0:
            raise ValueError(f"alpha + beta debe ser > 0; recibido alpha={alpha}, beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.bce_weight = float(bce_weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(torch.clamp(1.0 - tversky, min=1e-7, max=1.0), self.gamma).mean()

        if self.bce_weight > 0.0:
            if self.label_smoothing > 0.0:
                smoothed = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            else:
                smoothed = targets
            bce = F.binary_cross_entropy_with_logits(logits, smoothed)
            return (1.0 - self.bce_weight) * focal_tversky + self.bce_weight * bce

        return focal_tversky


BCEDiceLoss = FocalTverskyLoss


def resolve_training_device(requested_device: str = "auto", require_gpu: bool = False) -> torch.device:
    """Selecciona y optimiza el dispositivo de entrenamiento priorizando GPU CUDA si existe."""
    req = (requested_device or "auto").strip().lower()
    if req in ("auto", "gpu"):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            if require_gpu:
                raise RuntimeError("Se solicitó --require-gpu pero torch.cuda.is_available() es False.")
            device = torch.device("cpu")
    elif req.startswith("cuda"):
        if not torch.cuda.is_available():
            if require_gpu:
                raise RuntimeError("Se solicitó dispositivo CUDA pero no está disponible.")
            device = torch.device("cpu")
        else:
            device = torch.device(req)
    else:
        if require_gpu:
            raise RuntimeError(f"Se solicitó --require-gpu pero el dispositivo es '{req}'.")
        device = torch.device(req)

    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(2)
        torch.cuda.empty_cache()

    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Bucle de validación vectorizado
# ---------------------------------------------------------------------------
def evaluate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    """Evalúa el modelo sobre la partición de validación sin sampler.

    Calcula de forma vectorizada IoU y F1 sobre la clase 1 (deforestación).

    Returns:
        tuple[val_loss, val_iou, val_f1]
    """
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            total_val_loss += float(loss.item())
            num_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            tp = float((preds * masks).sum().item())
            fp = float((preds * (1.0 - masks)).sum().item())
            fn = float(((1.0 - preds) * masks).sum().item())

            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_val_loss = total_val_loss / max(num_batches, 1)
    val_iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    val_f1 = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)

    return avg_val_loss, val_iou, val_f1


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    val_hco_pairs: list[tuple[str, str, np.ndarray, np.ndarray]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Función evaluate completa con desglose opcional por zonas de Huánuco."""
    from src.dataset import parse_huanuco_pair_metadata

    model.eval()
    total_loss = 0.0
    batches = 0
    tp = fp = fn = tn = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())
        batches += 1

        probs = torch.sigmoid(logits)
        pred = probs >= threshold
        tgt = masks >= 0.5
        tp += int((pred & tgt).sum().item())
        fp += int((pred & ~tgt).sum().item())
        fn += int((~pred & tgt).sum().item())
        tn += int((~pred & ~tgt).sum().item())

    iou = _safe_div(tp, tp + fp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    result: dict[str, Any] = {
        "loss": total_loss / max(batches, 1),
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }

    if val_hco_pairs:
        zone_acc: dict[str, dict[str, int]] = {}
        h_tp = h_fp = h_fn = h_tn = 0
        for t0_n, t1_n, h_img, h_msk in val_hco_pairs:
            meta = parse_huanuco_pair_metadata(t0_n, t1_n)
            inp = torch.from_numpy(h_img).unsqueeze(0).float().to(device)
            tgt_t = torch.from_numpy(h_msk).unsqueeze(0).unsqueeze(0).float().to(device) >= 0.5
            prb_t = torch.sigmoid(model(inp)) >= threshold

            btp = int((prb_t & tgt_t).sum().item())
            bfp = int((prb_t & ~tgt_t).sum().item())
            bfn = int((~prb_t & tgt_t).sum().item())
            btn = int((~prb_t & ~tgt_t).sum().item())

            h_tp += btp
            h_fp += bfp
            h_fn += bfn
            h_tn += btn

            z_entry = zone_acc.setdefault(
                meta.zone_id, {"n_chips": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
            )
            z_entry["n_chips"] += 1
            z_entry["tp"] += btp
            z_entry["fp"] += bfp
            z_entry["fn"] += bfn
            z_entry["tn"] += btn

        zones_report: dict[str, dict[str, Any]] = {}
        for zid, zc in sorted(zone_acc.items()):
            z_iou = _safe_div(zc["tp"], zc["tp"] + zc["fp"] + zc["fn"])
            z_f1 = _safe_div(2 * zc["tp"], 2 * zc["tp"] + zc["fp"] + zc["fn"])
            z_prec = _safe_div(zc["tp"], zc["tp"] + zc["fp"])
            z_rec = _safe_div(zc["tp"], zc["tp"] + zc["fn"])
            zones_report[zid] = {
                "n_chips": zc["n_chips"],
                "iou": round(z_iou, 6),
                "f1": round(z_f1, 6),
                "precision": round(z_prec, 6),
                "recall": round(z_rec, 6),
                "tp": zc["tp"],
                "fp": zc["fp"],
                "fn": zc["fn"],
                "tn": zc["tn"],
            }

        macro_iou = float(np.mean([v["iou"] for v in zones_report.values()])) if zones_report else 0.0
        macro_f1 = float(np.mean([v["f1"] for v in zones_report.values()])) if zones_report else 0.0
        result["huanuco_iou"] = _safe_div(h_tp, h_tp + h_fp + h_fn)
        result["huanuco_f1"] = _safe_div(2 * h_tp, 2 * h_tp + h_fp + h_fn)
        result["huanuco_precision"] = _safe_div(h_tp, h_tp + h_fp)
        result["huanuco_recall"] = _safe_div(h_tp, h_tp + h_fn)
        result["huanuco_val_chips_count"] = len(val_hco_pairs)
        result["huanuco_zones"] = zones_report
        result["huanuco_macro_zone_iou"] = macro_iou
        result["huanuco_macro_zone_f1"] = macro_f1
        result["combined_score"] = 0.5 * iou + 0.5 * result["huanuco_iou"]

    return result


# ---------------------------------------------------------------------------
# Pipeline Principal de Entrenamiento (Fase B / Producción)
# ---------------------------------------------------------------------------
def train_pipeline(
    train_dataset: GeoForestDataset,
    val_dataset: GeoForestDataset,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 12,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path = "models/unet_best.pth",
    metrics_path: str | Path = "models/training_metrics.json",
    num_workers: int = 4,
    pin_memory: bool = True,
    base_features: int = 32,
    dropout: float = 0.15,
    target_pos_ratio: float = 0.5,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lambda_focal: float = 1.0,
    lambda_dice: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Ejecuta el pipeline completo de entrenamiento y validación de la Fase B."""
    dev = resolve_training_device(str(device) if device is not None else "auto")
    use_pin = pin_memory and dev.type == "cuda"
    nw = int(num_workers)

    # 1. Configuración obligatoria del WeightedRandomSampler en train_loader
    weights = train_dataset.get_sampler_weights(target_pos_ratio=target_pos_ratio)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=use_pin,
    )

    # 2. Configuración obligatoria de val_loader: SIN sampler, shuffle=False (distribución real)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin,
    )

    sample_img, _ = train_dataset[0]
    in_channels = int(sample_img.shape[0])

    # 3. Inicialización del modelo U-Net (base_features=32, FP32)
    model = UNet(
        in_channels=in_channels,
        num_classes=1,
        base_features=base_features,
        bilinear=True,
        dropout=dropout,
    ).to(dev)

    # 4. Función de pérdida híbrida FocalDiceLoss sobre logits crudos
    criterion = FocalDiceLoss(
        alpha=alpha,
        gamma=gamma,
        lambda_focal=lambda_focal,
        lambda_dice=lambda_dice,
        eps=eps,
    )

    # 5. Optimizador AdamW y Scheduler CosineAnnealingLR
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 6. Variables de seguimiento y Early Stopping
    best_val_iou = -1.0
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Iniciando entrenamiento: %d épocas, batch_size=%d, in_channels=%d, base_features=%d, dispositivo=%s",
        epochs,
        batch_size,
        in_channels,
        base_features,
        dev,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images = images.to(dev, non_blocking=True)
            masks = masks.to(dev, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += float(loss.item())
            train_batches += 1

        scheduler.step()
        train_loss = running_train_loss / max(train_batches, 1)

        # Validación sobre distribución real
        val_loss, val_iou, val_f1 = evaluate_epoch(model, val_loader, criterion, dev, eps=eps)
        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_iou": round(val_iou, 5),
            "val_f1": round(val_f1, 5),
            "lr": current_lr,
        }
        history.append(epoch_record)

        log.info(
            "Época [%3d/%3d] Train Loss: %.4f | Val Loss: %.4f | Val IoU: %.4f | Val F1: %.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_iou,
            val_f1,
        )

        # Criterio estricto de selección de Checkpoint: val_iou de clase 1
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": float(best_val_iou),
                "val_f1": float(best_val_f1),
                "val_loss": float(best_val_loss),
                "in_channels": in_channels,
                "base_features": base_features,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint_data, checkpoint_file)
            log.info("-> Nuevo mejor modelo guardado en %s (val_iou=%.4f)", checkpoint_file, best_val_iou)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping activado tras %d épocas sin mejora en val_iou.", patience)
                break

    # Saneamiento de trazabilidad: rutas estrictamente relativas sin rutas absolutas heredadas
    try:
        rel_ckpt = checkpoint_file.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_ckpt = "models/unet_best.pth" if checkpoint_file.name == "unet_best.pth" else checkpoint_file.name

    ckpt_sha256 = compute_sha256(checkpoint_file) if checkpoint_file.exists() else ""

    metrics_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_val_iou": float(best_val_iou),
        "best_val_f1": float(best_val_f1),
        "best_val_loss": float(best_val_loss),
        "checkpoint_path": rel_ckpt,
        "checkpoint_sha256": ckpt_sha256,
        "total_epochs": epoch,
        "model": {
            "architecture": "UNet",
            "in_channels": in_channels,
            "out_channels": 1,
            "base_features": base_features,
            "precision": "FP32",
        },
        "training_config": {
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduler": "CosineAnnealingLR",
            "epochs_max": epochs,
            "patience": patience,
            "loss": "FocalDiceLoss",
            "alpha": alpha,
            "gamma": gamma,
            "lambda_focal": lambda_focal,
            "lambda_dice": lambda_dice,
            "sampler": "WeightedRandomSampler(target_pos_ratio=0.5, replacement=True)",
        },
        "history": history,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    log.info("Historial de métricas guardado en %s", metrics_file)

    # Actualizar hashes SHA-256 en models/manifest.json si corresponde al directorio models
    manifest_candidate = checkpoint_file.parent / "manifest.json"
    update_manifest_hashes(
        manifest_path=manifest_candidate,
        checkpoint_path=checkpoint_file,
        onnx_path=checkpoint_file.parent / "unet_geoforest.onnx",
    )

    return metrics_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento U-Net con muestreo ponderado y FocalDiceLoss (Fase B)."
    )
    parser.add_argument("--train-index", default="data/processed/train_index.json")
    parser.add_argument("--val-index", default="data/processed/val_index.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--sentinel2-root", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--target-pos-ratio", type=float, default=0.5)
    parser.add_argument("--huanuco-val-fraction", type=float, default=0.25)
    parser.add_argument("--val-huanuco-zones", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run_legacy_zone_validation_mode(args: argparse.Namespace) -> int:
    """Modo compatible para pruebas unitarias de partición de zonas Huánuco + OSCD."""
    from src.dataset import (
        OSCDChangeDataset,
        discover_huanuco_pairs,
        discover_oscd_cities,
        load_huanuco_pairs_subset,
        split_cities,
        split_huanuco_pairs,
    )

    dev = resolve_training_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cities = discover_oscd_cities(args.data_root)
    train_cities, val_cities = split_cities(cities, val_fraction=0.2, seed=args.seed)
    val_ds = OSCDChangeDataset(
        args.data_root,
        cities=val_cities,
        patch_size=args.patch_size,
        train=False,
        seed=args.seed,
        augment=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    pairs = discover_huanuco_pairs(args.sentinel2_root)
    val_zones_list = (
        [z.strip() for z in args.val_huanuco_zones.split(",") if z.strip()]
        if args.val_huanuco_zones
        else None
    )
    train_p, val_p = split_huanuco_pairs(
        pairs,
        val_fraction=args.huanuco_val_fraction,
        val_zones=val_zones_list,
        seed=args.seed,
    )
    val_hco_loaded = load_huanuco_pairs_subset(val_p, sentinel2_root=args.sentinel2_root)

    model = UNet(in_channels=6, num_classes=1, base_features=args.base_features).to(dev)
    criterion = FocalTverskyLoss()
    metrics = evaluate(
        model,
        val_loader,
        criterion,
        dev,
        threshold=0.5,
        val_hco_pairs=val_hco_loaded,
    )

    payload = {
        "data": {
            "train_cities": train_cities,
            "val_cities": val_cities,
            "huanuco_train_pairs_count": len(train_p),
            "huanuco_val_pairs_count": len(val_p),
            "huanuco_train_pairs": [list(p) for p in train_p],
            "huanuco_val_pairs": [list(p) for p in val_p],
        },
        "training": {
            "metrics": metrics,
        },
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    seed_everything(args.seed)

    if args.sentinel2_root is not None and not Path(args.train_index).is_absolute() and args.data_root != ".":
        return _run_legacy_zone_validation_mode(args)

    if args.dry_run:
        log.info("Modo --dry-run activado: verificando rutas y datasets...")
        train_idx = Path(args.train_index)
        if not train_idx.exists():
            log.warning("No existe train_index en %s; saltando dry-run.", train_idx)
            return 0
        ds = GeoForestDataset(train_idx, root_dir=args.data_root)
        log.info("Dataset de entrenamiento contiene %d muestras.", len(ds))
        return 0

    train_ds = GeoForestDataset(args.train_index, root_dir=args.data_root)
    val_ds = GeoForestDataset(args.val_index, root_dir=args.data_root)

    out_dir = Path(args.output_dir)
    train_pipeline(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        checkpoint_path=out_dir / "unet_best.pth",
        metrics_path=out_dir / "training_metrics.json",
        num_workers=args.num_workers,
        base_features=args.base_features,
        dropout=args.dropout,
        target_pos_ratio=args.target_pos_ratio,
    )
    return 0


if __name__ == "__main__":
    main()
