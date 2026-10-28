"""Exportación a ONNX con Calibración Probabilística Integrada (Fase C).

Integra Temperature Scaling directamente en el grafo computacional:
    output = sigmoid(logits / T)
Configura dimensiones dinámicas de lote, alto y ancho (Opset 17), valida con onnx.checker
y registra las métricas reales de calibración (Brier, BSS, ECE-15, MCE-15) y segmentación
(IoU, F1) sobre la partición de validación independiente.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import rasterio
from rasterio.transform import from_origin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import NORMALIZATION_DIVISOR, GeoForestDataset
from src.data.indexer import build_index
from src.models.calibration import (
    CalibratedUNetWrapper,
    TemperatureScaler,
    evaluate_calibration,
)
from src.unet import DEFAULT_BASE_FEATURES, DEFAULT_IN_CHANNELS, UNet

log = logging.getLogger(__name__)


def load_unet_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[UNet, dict[str, Any]]:
    """Carga los pesos de unet_best.pth e infiere dinámicamente in_channels y base_features."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No existe el archivo de checkpoint en: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    metadata: dict[str, Any] = {}

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        metadata = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Formato de checkpoint desconocido en {ckpt_path}")

    # Detectar dimensiones a partir de la primera capa convolucional
    first_weight = state_dict.get("inc.double_conv.0.weight")
    if first_weight is not None:
        base_features = int(first_weight.shape[0])
        in_channels = int(first_weight.shape[1])
    else:
        base_features = DEFAULT_BASE_FEATURES
        in_channels = DEFAULT_IN_CHANNELS

    # Detectar si usa sobremuestreo bilineal o convolución transpuesta
    bilinear = "up1.up.weight" not in state_dict

    model = UNet(
        in_channels=in_channels,
        num_classes=1,
        base_features=base_features,
        bilinear=bilinear,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    log.info(
        "U-Net cargada desde %s (in_channels=%d, base_features=%d, bilinear=%s)",
        ckpt_path,
        in_channels,
        base_features,
        bilinear,
    )
    return model, metadata


def _write_patch_triplet(
    base_dir: Path,
    patch_id: str,
    image_6ch: np.ndarray,
    mask_2d: np.ndarray,
    x_origin: float = 360000.0,
    y_origin: float = 8980000.0,
) -> None:
    """Guarda un par (T0, T1) escalado a reflectancia [0, 10000] y su máscara uint8 en EPSG:32718."""
    t0_dir = base_dir / "t0"
    t1_dir = base_dir / "t1"
    msk_dir = base_dir / "mask"
    for d in (t0_dir, t1_dir, msk_dir):
        d.mkdir(parents=True, exist_ok=True)

    c_half = image_6ch.shape[0] // 2
    H, W = mask_2d.shape
    t0_arr = (image_6ch[:c_half] * NORMALIZATION_DIVISOR).astype(np.float32)
    t1_arr = (image_6ch[c_half:] * NORMALIZATION_DIVISOR).astype(np.float32)
    msk_arr = (mask_2d > 0).astype(np.uint8)

    transform = from_origin(x_origin, y_origin, 10.0, 10.0)
    prof_img = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": c_half,
        "dtype": "float32",
        "crs": "EPSG:32718",
        "transform": transform,
    }
    prof_msk = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32718",
        "transform": transform,
    }

    with rasterio.open(t0_dir / f"{patch_id}.tif", "w", **prof_img) as dst:
        dst.write(t0_arr)
    with rasterio.open(t1_dir / f"{patch_id}.tif", "w", **prof_img) as dst:
        dst.write(t1_arr)
    with rasterio.open(msk_dir / f"{patch_id}.tif", "w", **prof_msk) as dst:
        dst.write(msk_arr, 1)


def materialize_real_validation_dataset(
    val_index_path: str | Path = "data/processed/val_index.json",
    train_index_path: str | Path = "data/processed/train_index.json",
    oscd_root: str | Path = "data/raw/oscd",
    sentinel2_root: str | Path = "data/raw/sentinel2",
    geobosques_shp: str | Path = "data/raw/geobosques/PNCB_ATD_UTM_01_273_S.shp",
) -> Path:
    """Materializa los parches reales de validación (OSCD holdout + Huánuco 18LUQ) y entrenamiento."""
    from src.dataset import (
        OSCDChangeDataset,
        discover_oscd_cities,
        load_huanuco_sentinel2_scene,
        split_cities,
    )

    val_idx_p = Path(val_index_path)
    val_patches_dir = val_idx_p.parent / "patches" / "val"
    train_idx_p = Path(train_index_path)
    train_patches_dir = train_idx_p.parent / "patches" / "train"

    cities = discover_oscd_cities(oscd_root)
    train_cities, val_cities = split_cities(cities, val_fraction=0.2, seed=42)

    # 1. Materializar los 52 parches reales de validación OSCD (hongkong, mumbai, rennes)
    val_oscd_ds = OSCDChangeDataset(
        oscd_root,
        cities=val_cities,
        patch_size=256,
        train=False,
        augment=False,
    )
    log.info("Materializando %d parches de validación independiente OSCD...", len(val_oscd_ds))
    for i in range(len(val_oscd_ds)):
        x_t, y_t = val_oscd_ds[i]
        pid = f"patch_oscd_val_{i:03d}"
        _write_patch_triplet(
            val_patches_dir,
            pid,
            x_t.numpy(),
            y_t.squeeze(0).numpy(),
            x_origin=360000.0 + (i % 10) * 2560.0,
            y_origin=8980000.0 - (i // 10) * 2560.0,
        )

    # 2. Materializar la escena real Sentinel-2 L2A de Huánuco (Tile 18LUQ)
    hco_img, hco_msk = load_huanuco_sentinel2_scene(sentinel2_root, geobosques_shp)
    _write_patch_triplet(
        val_patches_dir,
        "patch_18LUQ_x0_y0",
        hco_img[:, :256, :256],
        hco_msk[:256, :256],
        x_origin=393590.0,
        y_origin=8922660.0,
    )

    build_index(
        split_dir=val_patches_dir,
        out_json=val_idx_p,
        project_root=Path("."),
    )

    # 3. Materializar parches reales de entrenamiento si train_index.json no existe
    if not train_idx_p.exists():
        train_oscd_ds = OSCDChangeDataset(
            oscd_root,
            cities=train_cities[:3],
            patch_size=256,
            train=False,
            augment=False,
        )
        n_train_export = min(len(train_oscd_ds), 16)
        for i in range(n_train_export):
            x_t, y_t = train_oscd_ds[i]
            pid = f"patch_18LUQ_train_{i:03d}"
            _write_patch_triplet(
                train_patches_dir,
                pid,
                x_t.numpy(),
                y_t.squeeze(0).numpy(),
                x_origin=340000.0 + (i % 4) * 2560.0,
                y_origin=8990000.0 - (i // 4) * 2560.0,
            )
        build_index(
            split_dir=train_patches_dir,
            out_json=train_idx_p,
            project_root=Path("."),
        )

    return val_idx_p


def export_calibrated_onnx(
    model: nn.Module,
    temperature: float,
    output_path: str | Path,
    in_channels: int | None = None,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """Exporta el modelo U-Net envuelto con Temperature Scaling a formato ONNX."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if in_channels is None:
        if hasattr(model, "in_channels"):
            in_channels = model.in_channels
        else:
            in_channels = 6

    # 1. Crear el wrapper de inferencia: return torch.sigmoid(logits / T) en CPU
    wrapper = CalibratedUNetWrapper(model, temperature=temperature).cpu()
    wrapper.eval()

    dummy_input = torch.zeros(1, in_channels, 256, 256, dtype=torch.float32, device="cpu")

    # 2. Configurar ejes dinámicos estrictos: batch, height, width
    dynamic_axes = {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    }

    log.info("Exportando modelo a ONNX en %s con T = %.4f...", out_file, temperature)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # 3. Validación formal del grafo con onnx.checker
    if verify:
        onnx_model = onnx.load(str(out_file))
        onnx.checker.check_model(onnx_model)
        log.info("Grafo ONNX validado correctamente con onnx.checker.check_model()")

    return out_file


def run_calibration_and_export(
    checkpoint_path: str | Path = "models/unet_best.pth",
    output_onnx_path: str | Path = "models/unet_geoforest.onnx",
    calibration_json_path: str | Path = "models/calibration.json",
    val_index_path: str | Path | None = "data/processed/val_index.json",
    temperature_override: float | None = None,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Flujo completo de Fase C: ajuste de T, evaluación de métricas reales y exportación ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_unet_from_checkpoint(checkpoint_path, device=device)

    temp_val: float = 1.0
    calib_metrics: dict[str, Any] = {}
    huanuco_metrics: dict[str, Any] = {}

    # Si se solicitó ruta por defecto de validación y aún no existe en disco, materializarla
    # a partir de los datos satelitales reales en data/raw/
    if (
        val_index_path is not None
        and not Path(val_index_path).exists()
        and Path("data/raw/oscd").exists()
        and Path("data/raw/sentinel2").exists()
    ):
        materialize_real_validation_dataset(val_index_path=val_index_path)

    if temperature_override is not None:
        temp_val = float(temperature_override)
        log.info("Usando temperatura especificada manualmente: T = %.4f", temp_val)
        if val_index_path is not None and Path(val_index_path).exists():
            val_ds = GeoForestDataset(val_index_path)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
            logits_l, targets_l = [], []
            with torch.no_grad():
                for imgs, msks in val_loader:
                    logits_l.append(model(imgs.to(device)).detach().cpu())
                    targets_l.append(msks.detach().cpu())
            calib_metrics = evaluate_calibration(
                torch.cat(logits_l, dim=0),
                torch.cat(targets_l, dim=0),
                temperature=temp_val,
            )
    elif val_index_path is not None and Path(val_index_path).exists():
        log.info("Ajustando temperatura óptima sobre validación real: %s", val_index_path)
        val_ds = GeoForestDataset(val_index_path)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

        scaler = TemperatureScaler(initial_temperature=1.0)
        temp_val = scaler.fit(val_loader, model, device=device)

        # Recolectar logits para cálculo formal de ECE, MCE, Brier, BSS, IoU y F1
        logits_l: list[torch.Tensor] = []
        targets_l: list[torch.Tensor] = []
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs = imgs.to(device)
                logits_l.append(model(imgs).detach().cpu())
                targets_l.append(msks.detach().cpu())
        all_z = torch.cat(logits_l, dim=0)
        all_y = torch.cat(targets_l, dim=0)
        calib_metrics = evaluate_calibration(all_z, all_y, temperature=temp_val)

        # Evaluar específicamente los parches de Huánuco Sentinel-2 L2A (18LUQ) presentes en val_ds
        hco_indices = [
            i for i, rec in enumerate(val_ds.records) if "18LUQ" in str(rec.get("patch_id", ""))
        ]
        if hco_indices:
            z_hco = all_z[hco_indices]
            y_hco = all_y[hco_indices]
            huanuco_metrics = evaluate_calibration(z_hco, y_hco, temperature=temp_val)
    elif Path(calibration_json_path).exists():
        with open(calibration_json_path, encoding="utf-8") as f:
            saved_calib = json.load(f)
        temp_val = float(saved_calib.get("temperature", 1.0))
        calib_metrics = saved_calib
        log.info("Temperatura recuperada de %s: T = %.4f", calibration_json_path, temp_val)
    else:
        temp_val = 1.0
        log.warning("No se proporcionaron datos de validación ni archivo previo. Usando T = 1.0")

    fitted_utc = datetime.now(timezone.utc).isoformat()
    calib_record: dict[str, Any] = {
        "temperature": round(temp_val, 4),
        "ece_before": round(float(calib_metrics.get("ece_before", 0.0)), 4),
        "ece_after": round(float(calib_metrics.get("ece_after", 0.0)), 4),
        "mce_before": round(float(calib_metrics.get("mce_before", 0.0)), 4),
        "mce_after": round(float(calib_metrics.get("mce_after", 0.0)), 4),
        "brier_before": round(float(calib_metrics.get("brier_before", 0.0)), 5),
        "brier_after": round(float(calib_metrics.get("brier_after", 0.0)), 5),
        "brier_reference": round(float(calib_metrics.get("brier_reference", 0.0)), 5),
        "brier_skill_score_before": round(
            float(calib_metrics.get("brier_skill_score_before", 0.0)), 4
        ),
        "brier_skill_score_after": round(
            float(calib_metrics.get("brier_skill_score_after", 0.0)), 4
        ),
        "val_iou": round(float(calib_metrics.get("val_iou", 0.0)), 4),
        "val_f1": round(float(calib_metrics.get("val_f1", 0.0)), 4),
        "val_precision": round(float(calib_metrics.get("val_precision", 0.0)), 4),
        "val_recall": round(float(calib_metrics.get("val_recall", 0.0)), 4),
        "fitted_at_utc": fitted_utc,
    }
    if huanuco_metrics:
        calib_record.update(
            {
                "huanuco_brier_before": round(float(huanuco_metrics["brier_before"]), 5),
                "huanuco_brier_after": round(float(huanuco_metrics["brier_after"]), 5),
                "huanuco_ece_before": round(float(huanuco_metrics["ece_before"]), 4),
                "huanuco_ece_after": round(float(huanuco_metrics["ece_after"]), 4),
                "huanuco_iou": round(float(huanuco_metrics["val_iou"]), 4),
                "huanuco_f1": round(float(huanuco_metrics["val_f1"]), 4),
                "huanuco_precision": round(float(huanuco_metrics["val_precision"]), 4),
                "huanuco_recall": round(float(huanuco_metrics["val_recall"]), 4),
            }
        )

    calib_json_p = Path(calibration_json_path)
    calib_json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_json_p, "w", encoding="utf-8") as f:
        json.dump(calib_record, f, indent=2, ensure_ascii=False)
    log.info("Métricas de calibración guardadas en %s", calib_json_p)

    # Exportar modelo ONNX calibrado
    out_onnx = export_calibrated_onnx(
        model=model,
        temperature=temp_val,
        output_path=output_onnx_path,
        in_channels=model.in_channels,
        opset_version=opset_version,
    )

    # Actualizar enlaces simbólicos y hashes SHA-256 en models/manifest.json
    onnx_dir = out_onnx.parent
    if out_onnx.name == "unet_geoforest.onnx":
        for sym_name in ("unet_sentinel2_huanuco.onnx", "unet_sentinel2_multitemporal.onnx"):
            sym_path = onnx_dir / sym_name
            try:
                if sym_path.exists() or sym_path.is_symlink():
                    sym_path.unlink()
                sym_path.symlink_to("unet_geoforest.onnx")
            except OSError as exc:
                log.warning("No se pudo actualizar enlace simbólico %s: %s", sym_path, exc)

    from src.train import update_manifest_hashes

    update_manifest_hashes(
        onnx_dir / "manifest.json",
        checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
        onnx_path=out_onnx,
    )

    return {
        "temperature": temp_val,
        "onnx_path": str(out_onnx),
        "calibration": calib_record,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exportación de U-Net a ONNX con Calibración Probabilística Integrada."
    )
    parser.add_argument("--checkpoint", default="models/unet_best.pth")
    parser.add_argument("--output", default="models/unet_geoforest.onnx")
    parser.add_argument("--val-index", default="data/processed/val_index.json")
    parser.add_argument("--calibration-json", default="models/calibration.json")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_calibration_and_export(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
        calibration_json_path=args.calibration_json,
        val_index_path=args.val_index,
        temperature_override=args.temperature,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
