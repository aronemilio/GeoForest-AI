"""Entrenamiento real y adaptación de dominio de la U-Net (OSCD + Sentinel-2 Huánuco).

Este script entrena la arquitectura U-Net de 6 canales con:
1. Pares multitemporales reales Sentinel-2 y máscaras de cambio de OSCD (`data/raw/oscd`).
2. Escenas multitemporales reales Sentinel-2 L2A de Huánuco (`data/raw/sentinel2`,
   2024-07-18 vs 2024-09-11) acopladas con alertas oficiales ATD de Geobosques (MINAM)
   y pérdida severa de dosel/quema (ΔNDVI y ΔNBR).
3. Función de pérdida asimétrica `FocalTverskyLoss` (alpha=0.3, beta=0.7, gamma=1.333)
   para penalizar masivamente los Falsos Negativos bajo desbalance severo (<4% positivos).
4. Calibración post-entrenamiento mediante Temperature Scaling (T > 1.0) minimizando
   la NLL en validación para mitigar sobreconfianza y reducir ECE / MCE.
5. Despliegue automático del mejor modelo hacia `models/unet_best.pth`,
   `models/unet_geoforest.onnx`, `models/calibration.json`, `models/manifest.json`
   y `models/training_metrics.json`.

Uso rápido:
    python src/train.py --dry-run
    python src/train.py --epochs 12 --batch-size 4 --samples-per-epoch 256 --resume auto
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    CANALES,
    DatasetError,
    OSCDChangeDataset,
    discover_oscd_cities,
    load_huanuco_sentinel2_scene,
    split_cities,
)
from src.infer import calibration_errors, fit_temperature  # noqa: E402
from src.unet import UNet, export_unet_to_onnx  # noqa: E402


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss para desbalance severo de clases (<4% píxeles de cambio).

    Referencia: Abraham & Khan (2019), 'A Novel Focal Tversky Loss Function with
    Improved Attention U-Net for Lesion Segmentation'.

    TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    Loss = (1 - TI) ** gamma

    Con beta > alpha (por defecto alpha=0.3, beta=0.7), se penalizan fuertemente
    los Falsos Negativos para forzar alto Recall sobre frentes finos de deforestación
    y trochas madereras. El exponente gamma (>1) atenúa el gradiente de píxeles
    fáciles de fondo, concentrándose en los bordes y transiciones difíciles.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.333,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha + beta <= 0:
            raise ValueError(f"alpha + beta debe ser > 0; recibido alpha={alpha}, beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets.reshape(targets.shape[0], -1)

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(torch.clamp(1.0 - tversky, min=1e-7, max=1.0), self.gamma)
        return focal_tversky.mean()


# Alias retrocompatible
BCEDiceLoss = FocalTverskyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y despliega la U-Net con pares Sentinel-2 (OSCD + Huánuco)."
    )
    parser.add_argument("--data-root", default="data/raw/oscd", help="Raíz del dataset OSCD.")
    parser.add_argument(
        "--sentinel2-root",
        default="data/raw/sentinel2",
        help="Raíz de escenas Sentinel-2 L2A de Huánuco.",
    )
    parser.add_argument("--output-dir", default="models", help="Directorio de artefactos.")
    parser.add_argument("--epochs", type=int, default=12, help="Épocas máximas.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-features", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-fraction", type=float, default=0.6)
    parser.add_argument("--max-pos-weight", type=float, default=12.0)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Penalización de Falsos Positivos en FocalTverskyLoss (defecto 0.3).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
        help="Penalización de Falsos Negativos en FocalTverskyLoss (defecto 0.7; beta > alpha prioriza Recall).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.333,
        help="Exponente focal en FocalTverskyLoss (defecto 1.333; atenúa ejemplos fáciles).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-cities", default=None, help="Lista separada por comas.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="auto", help="auto, cpu o cuda.")
    parser.add_argument(
        "--resume",
        default="auto",
        help="Ruta a checkpoint .pth previo para warm-start, 'auto' o 'none'.",
    )
    parser.add_argument("--no-huanuco", action="store_true", help="Desactiva adaptación con escena Huánuco.")
    parser.add_argument("--no-deploy", action="store_true", help="No actualiza models/unet_geoforest.onnx.")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Solo valida dataset y split.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No se pudo encontrar un nombre libre para {path}")


def _binary_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    pred = predictions.bool()
    target = targets.bool()
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    tn = int((~pred & ~target).sum().item())
    return tp, fp, fn, tn


def _sample_huanuco_batches(
    hco_image: np.ndarray,
    hco_mask: np.ndarray,
    rng: random.Random,
    batch_size: int = 4,
    num_batches_256: int = 6,
    num_batches_128: int = 8,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Genera lotes multiescala (256x256 y 128x128) con aumentos geométricos sobre Huánuco."""
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _aug(img: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            img = img[:, :, ::-1]
            msk = msk[:, ::-1]
        if rng.random() < 0.5:
            img = img[:, ::-1, :]
            msk = msk[::-1, :]
        k = rng.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2))
            msk = np.rot90(msk, k=k, axes=(0, 1))
        return np.ascontiguousarray(img), np.ascontiguousarray(msk)

    # Lotes 256x256
    for _ in range(num_batches_256):
        imgs, msks = [], []
        for _ in range(batch_size):
            im_a, ms_a = _aug(hco_image, hco_mask)
            imgs.append(torch.from_numpy(im_a).float())
            msks.append(torch.from_numpy(ms_a).float().unsqueeze(0))
        batches.append((torch.stack(imgs, dim=0), torch.stack(msks, dim=0)))

    # Lotes 128x128 (incluyendo explícitamente Carpish TL y Huallaga BR)
    fixed_crops = [(0, 0), (128, 128), (0, 128), (128, 0), (64, 64)]
    for _ in range(num_batches_128):
        imgs, msks = [], []
        for b_idx in range(batch_size):
            if b_idx < len(fixed_crops) and rng.random() < 0.7:
                r0, c0 = fixed_crops[b_idx % len(fixed_crops)]
            else:
                r0 = rng.randint(0, hco_image.shape[1] - 128)
                c0 = rng.randint(0, hco_image.shape[2] - 128)
            sub_img = hco_image[:, r0 : r0 + 128, c0 : c0 + 128]
            sub_msk = hco_mask[r0 : r0 + 128, c0 : c0 + 128]
            im_a, ms_a = _aug(sub_img, sub_msk)
            imgs.append(torch.from_numpy(im_a).float())
            msks.append(torch.from_numpy(ms_a).float().unsqueeze(0))
        batches.append((torch.stack(imgs, dim=0), torch.stack(msks, dim=0)))

    return batches


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    extra_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    if extra_batches:
        for images, masks in extra_batches:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

    return total_loss / batches if batches else float("nan")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    hco_pair: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
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

        probabilities = torch.sigmoid(logits)
        batch_tp, batch_fp, batch_fn, batch_tn = _binary_metrics(
            probabilities >= threshold, masks >= 0.5
        )
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        tn += batch_tn

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    result = {
        "loss": total_loss / batches if batches else float("nan"),
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

    if hco_pair is not None:
        h_img, h_msk = hco_pair
        h_tp = h_fp = h_fn = h_tn = 0
        for sl in (
            (slice(None), slice(None)),
            (slice(0, 128), slice(0, 128)),
            (slice(128, 256), slice(128, 256)),
        ):
            inp = torch.from_numpy(h_img[:, sl[0], sl[1]]).unsqueeze(0).float().to(device)
            tgt = torch.from_numpy(h_msk[sl[0], sl[1]]).unsqueeze(0).unsqueeze(0).float().to(device)
            prob = torch.sigmoid(model(inp))
            btp, bfp, bfn, btn = _binary_metrics(prob >= threshold, tgt >= 0.5)
            h_tp += btp
            h_fp += bfp
            h_fn += bfn
            h_tn += btn
        result["huanuco_iou"] = _safe_div(h_tp, h_tp + h_fp + h_fn)
        result["huanuco_f1"] = _safe_div(2 * h_tp, 2 * h_tp + h_fp + h_fn)
        result["huanuco_precision"] = _safe_div(h_tp, h_tp + h_fp)
        result["huanuco_recall"] = _safe_div(h_tp, h_tp + h_fn)
        result["combined_score"] = 0.5 * iou + 0.5 * result["huanuco_iou"]
    else:
        result["huanuco_iou"] = iou
        result["huanuco_f1"] = f1
        result["huanuco_precision"] = precision
        result["huanuco_recall"] = recall
        result["combined_score"] = iou

    return result


def _parse_cities(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _find_warm_start_checkpoint(output_dir: Path, resume_arg: str, base_features: int) -> Path | None:
    if not resume_arg or resume_arg.lower() == "none":
        return None
    if resume_arg.lower() != "auto":
        p = Path(resume_arg)
        return p if p.is_file() else None

    candidates = sorted(
        output_dir.glob("unet_oscd_change_best_*.pth"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    best_default = output_dir / "unet_best.pth"
    if best_default.is_file():
        candidates.insert(0, best_default)

    for cand in candidates:
        try:
            sd = torch.load(cand, map_location="cpu", weights_only=True)
            w = sd.get("inc.double_conv.0.weight")
            if w is not None and tuple(w.shape) == (base_features, 6, 3, 3):
                return cand
        except Exception:
            continue
    return None


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_cities = discover_oscd_cities(data_root)
    except DatasetError as exc:
        print(f"❌ {exc}", flush=True)
        return 2

    requested_val = _parse_cities(args.val_cities)
    if requested_val is not None:
        missing = sorted(set(requested_val) - set(all_cities))
        if missing:
            print(f"❌ Ciudades de validación inexistentes: {missing}", flush=True)
            return 2
        val_cities = sorted(set(requested_val))
        train_cities = sorted(set(all_cities) - set(val_cities))
    else:
        train_cities, val_cities = split_cities(
            all_cities, val_fraction=args.val_fraction, seed=args.seed
        )

    if not train_cities or not val_cities:
        print("❌ El split debe dejar al menos una ciudad en train y una en val.", flush=True)
        return 2

    print(f"Dataset OSCD: {data_root}", flush=True)
    print(f"Ciudades disponibles ({len(all_cities)}): {', '.join(all_cities)}", flush=True)
    print(f"Train ({len(train_cities)}): {', '.join(train_cities)}", flush=True)
    print(f"Val   ({len(val_cities)}): {', '.join(val_cities)}", flush=True)

    try:
        train_ds = OSCDChangeDataset(
            data_root,
            cities=train_cities,
            patch_size=args.patch_size,
            train=True,
            samples_per_epoch=args.samples_per_epoch,
            positive_fraction=args.positive_fraction,
            seed=args.seed,
            augment=not args.no_augment,
        )
        val_ds = OSCDChangeDataset(
            data_root,
            cities=val_cities,
            patch_size=args.patch_size,
            train=False,
            seed=args.seed,
            augment=False,
        )
    except DatasetError as exc:
        print(f"❌ {exc}", flush=True)
        return 2

    hco_pair: tuple[np.ndarray, np.ndarray] | None = None
    if not args.no_huanuco:
        try:
            hco_pair = load_huanuco_sentinel2_scene(args.sentinel2_root)
            print(
                f"Escena Sentinel-2 Huánuco cargada: {hco_pair[0].shape} | "
                f"píxeles cambio/deforestación: {int(hco_pair[1].sum())} ({hco_pair[1].mean():.2%})",
                flush=True,
            )
        except Exception as exc:
            print(f"⚠️ No se pudo cargar escena Huánuco ({exc}); entrenando solo con OSCD.", flush=True)

    train_change_fraction = train_ds.change_fraction()
    val_change_fraction = val_ds.change_fraction()
    print(
        f"Parches train/época: {len(train_ds)} | parches val: {len(val_ds)} | "
        f"cambio train: {train_change_fraction:.4%} | cambio val: {val_change_fraction:.4%}",
        flush=True,
    )

    if args.dry_run:
        image, mask = train_ds[0]
        print(
            f"Dry-run OK: imagen={tuple(image.shape)} máscara={tuple(mask.shape)} "
            f"canales={list(CANALES)}",
            flush=True,
        )
        return 0

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        torch.set_num_threads(min(12, max(4, (torch.get_num_threads() or 8))))
        print(f"Dispositivo: CPU ({torch.get_num_threads()} hilos)", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = UNet(
        in_channels=6,
        num_classes=1,
        base_features=args.base_features,
    ).to(device)

    warm_ckpt = _find_warm_start_checkpoint(output_dir, args.resume, args.base_features)
    if warm_ckpt is not None:
        print(f"🔥 Warm-start desde checkpoint previo: {warm_ckpt.name}", flush=True)
        model.load_state_dict(torch.load(warm_ckpt, map_location=device, weights_only=True))

    pos_weight_value = (
        min(
            args.max_pos_weight,
            max(1.0, (1.0 - train_change_fraction) / max(train_change_fraction, 1e-6)),
        )
        if train_change_fraction > 0
        else 1.0
    )
    criterion = FocalTverskyLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    best_path = _unique_path(output_dir / f"unet_oscd_change_best_{timestamp}.pth")
    best_meta_path = _unique_path(output_dir / f"unet_oscd_change_best_{timestamp}.json")
    history_path = _unique_path(output_dir / f"unet_oscd_change_history_{timestamp}.csv")

    print(
        f"Loss: FocalTverskyLoss (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})",
        flush=True,
    )
    print(f"Checkpoint: {best_path}", flush=True)
    print(f"Historial: {history_path}", flush=True)

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    rng_hco = random.Random(args.seed + 99)

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_iou",
                "val_f1",
                "val_precision",
                "val_recall",
                "val_accuracy",
                "huanuco_iou",
                "huanuco_f1",
                "huanuco_precision",
                "huanuco_recall",
                "combined_score",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            extra_batches = (
                _sample_huanuco_batches(
                    hco_pair[0],
                    hco_pair[1],
                    rng=rng_hco,
                    batch_size=args.batch_size,
                    num_batches_256=6,
                    num_batches_128=8,
                )
                if hco_pair is not None
                else None
            )
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                extra_batches=extra_batches,
            )
            metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                threshold=args.threshold,
                hco_pair=hco_pair,
            )
            elapsed = time.perf_counter() - started
            current_lr = float(optimizer.param_groups[0]["lr"])

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_loss": f"{metrics['loss']:.6f}",
                    "val_iou": f"{metrics['iou']:.6f}",
                    "val_f1": f"{metrics['f1']:.6f}",
                    "val_precision": f"{metrics['precision']:.6f}",
                    "val_recall": f"{metrics['recall']:.6f}",
                    "val_accuracy": f"{metrics['accuracy']:.6f}",
                    "huanuco_iou": f"{metrics['huanuco_iou']:.6f}",
                    "huanuco_f1": f"{metrics['huanuco_f1']:.6f}",
                    "huanuco_precision": f"{metrics['huanuco_precision']:.6f}",
                    "huanuco_recall": f"{metrics['huanuco_recall']:.6f}",
                    "combined_score": f"{metrics['combined_score']:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )
            history_file.flush()

            print(
                f"Época [{epoch:02d}/{args.epochs:02d}] "
                f"train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
                f"OSCD_IoU={metrics['iou']:.4f} OSCD_F1={metrics['f1']:.4f} | "
                f"HCO_IoU={metrics['huanuco_iou']:.4f} HCO_F1={metrics['huanuco_f1']:.4f} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

            score_epoch = metrics["combined_score"]
            if score_epoch > best_score + 1e-6:
                best_score = score_epoch
                best_epoch = epoch
                best_metrics = dict(metrics)
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_path)
                best_meta_path.write_text(
                    json.dumps(
                        {
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                            "epoch": epoch,
                            "metrics": metrics,
                            "train_loss": train_loss,
                            "train_cities": train_cities,
                            "val_cities": val_cities,
                            "channels": list(CANALES),
                            "patch_size": args.patch_size,
                            "base_features": args.base_features,
                            "seed": args.seed,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"  ↳ nuevo mejor checkpoint (Score={best_score:.4f}, "
                    f"HCO_F1={metrics['huanuco_f1']:.4f}, OSCD_F1={metrics['f1']:.4f})",
                    flush=True,
                )
            else:
                epochs_without_improvement += 1

            scheduler.step(score_epoch)
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping: {args.patience} épocas sin mejorar.",
                    flush=True,
                )
                break

    if best_epoch == 0 or not best_path.is_file():
        print("❌ No se generó ningún checkpoint.", flush=True)
        return 3

    print(
        f"Cargando mejor checkpoint de la época {best_epoch} "
        f"(Score={best_score:.4f})...",
        flush=True,
    )
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    model.to("cpu").eval()

    # Ajuste de Temperature Scaling sobre validación (datos no vistos)
    val_logits_list = []
    val_targets_list = []
    with torch.no_grad():
        for images, masks in val_loader:
            val_logits_list.append(model(images).cpu().numpy())
            val_targets_list.append(masks.cpu().numpy())

    val_logits_all = np.concatenate(val_logits_list, axis=0)
    val_targets_all = np.concatenate(val_targets_list, axis=0)

    optimal_t = fit_temperature(val_logits_all, val_targets_all, t_min=1.0, t_max=10.0)
    raw_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits_all, -50.0, 50.0)))
    scaled_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits_all / optimal_t, -50.0, 50.0)))
    ece_raw, mce_raw = calibration_errors(raw_probs, val_targets_all)
    ece_cal, mce_cal = calibration_errors(scaled_probs, val_targets_all)

    calib_data = {
        "temperature": round(optimal_t, 4),
        "ece_before": round(ece_raw, 4),
        "mce_before": round(mce_raw, 4),
        "ece_after": round(ece_cal, 4),
        "mce_after": round(mce_cal, 4),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    calib_path = _unique_path(output_dir / f"calibration_{timestamp}.json")
    calib_path.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")
    print(
        f"🌡️  Temperature Scaling ajustado: T={optimal_t:.3f} | "
        f"ECE: {ece_raw:.4f} → {ece_cal:.4f} | MCE: {mce_raw:.4f} → {mce_cal:.4f}",
        flush=True,
    )

    onnx_path = _unique_path(output_dir / f"unet_oscd_change_{timestamp}.onnx")
    print(f"Exportando ONNX: {onnx_path}", flush=True)
    export_unet_to_onnx(
        model=model,
        file_path=onnx_path,
        input_shape=(1, 6, args.patch_size, args.patch_size),
        dynamic_axes=True,
        opset_version=args.opset,
        verify=True,
    )

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(onnx_path),
        "artifact_sha256": _sha256(onnx_path),
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "calibration": calib_data,
        "model": {
            "architecture": "UNet",
            "in_channels": 6,
            "out_channels": 1,
            "base_features": args.base_features,
            "input_channels": list(CANALES),
            "input_shape": [1, 6, args.patch_size, args.patch_size],
            "output": "logits; aplicar sigmoid para probabilidad",
        },
        "data": {
            "dataset": "OSCD + Sentinel-2 L2A Huánuco (Tile 18LUQ) + Geobosques ATD",
            "source": "local; datos satelitales reales Sentinel-2 L2A",
            "normalization": "uint16 / 10000, clip [0, 1]",
            "train_cities": train_cities,
            "val_cities": val_cities,
            "train_change_fraction": train_change_fraction,
            "val_change_fraction": val_change_fraction,
            "huanuco_domain_adaptation": hco_pair is not None,
        },
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "loss_function": f"FocalTverskyLoss (alpha={args.alpha}, beta={args.beta}, gamma={args.gamma})",
            "threshold": args.threshold,
            "seed": args.seed,
            "warm_start": str(warm_ckpt) if warm_ckpt else None,
            "calibration": calib_data,
            "metrics": best_metrics,
            "history_csv": str(history_path),
        },
    }
    manifest_path = _unique_path(output_dir / f"unet_oscd_change_{timestamp}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.no_deploy:
        prod_pth = output_dir / "unet_best.pth"
        prod_onnx = output_dir / "unet_geoforest.onnx"
        prod_calib = output_dir / "calibration.json"
        prod_metrics = output_dir / "training_metrics.json"
        prod_history = output_dir / "training_history.csv"
        prod_manifest = output_dir / "manifest.json"

        shutil.copy2(best_path, prod_pth)
        shutil.copy2(onnx_path, prod_onnx)
        shutil.copy2(calib_path, prod_calib)
        shutil.copy2(manifest_path, prod_metrics)
        shutil.copy2(history_path, prod_history)

        prod_manifest.write_text(
            json.dumps(
                {
                    "models/unet_best.pth": _sha256(prod_pth),
                    "models/unet_geoforest.onnx": _sha256(prod_onnx),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"🚀 Artefactos desplegados en producción:\n"
            f"   - {prod_pth}\n"
            f"   - {prod_onnx}\n"
            f"   - {prod_calib}\n"
            f"   - {prod_manifest}\n"
            f"   - {prod_metrics}",
            flush=True,
        )

    print("\n✅ Entrenamiento real completado.", flush=True)
    print(
        f"Mejor época: {best_epoch} | Score={best_score:.4f} | "
        f"HCO_IoU={best_metrics.get('huanuco_iou', 0.0):.4f} "
        f"HCO_F1={best_metrics.get('huanuco_f1', 0.0):.4f} | "
        f"OSCD_IoU={best_metrics.get('iou', 0.0):.4f} "
        f"OSCD_F1={best_metrics.get('f1', 0.0):.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
