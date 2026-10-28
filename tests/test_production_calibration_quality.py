"""Suite de pruebas de calidad de producción para Calibración Probabilística (Brier/ECE) y Segmentación (IoU/F1).

Evalúa exclusivamente el checkpoint real de producción (`models/unet_best.pth`) y el artefacto
ONNX exportado (`models/unet_geoforest.onnx`) sobre la partición de validación real e
independiente (`data/processed/val_index.json`: 32 escenas Sentinel-2 L2A de Huánuco Tile 18LUQ
en Bosque Montano de Carpish y Valle del Huallaga), exigiendo umbrales explícitos de Brier Score,
Brier Skill Score, ECE-15, IoU y F1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from scripts.export_onnx import load_unet_from_checkpoint
from src.data.dataset import GeoForestDataset
from src.models.calibration import (
    CalibratedUNetWrapper,
    TemperatureScaler,
    compute_brier_score,
    compute_segmentation_metrics,
    evaluate_calibration,
)

CHECKPOINT_PATH = Path("models/unet_best.pth")
ONNX_PATH = Path("models/unet_geoforest.onnx")
CALIBRATION_JSON_PATH = Path("models/calibration.json")
VAL_INDEX_PATH = Path("data/processed/val_index.json")
TRAIN_INDEX_PATH = Path("data/processed/train_index.json")

# Umbrales explícitos de calidad en producción
MAX_ECE_AFTER = 0.0250
MAX_GLOBAL_BRIER_AFTER = 0.0332
MAX_HUANUCO_BRIER_AFTER = 0.0332
MIN_HUANUCO_IOU = 0.8000
MIN_HUANUCO_F1 = 0.8500
MIN_GLOBAL_VAL_IOU = 0.8500
MIN_GLOBAL_VAL_F1 = 0.8500


@pytest.fixture(scope="module")
def real_validation_bundle():
    """Carga el checkpoint real y ejecuta un pase de inferencia sobre la validación real independiente."""
    assert CHECKPOINT_PATH.exists(), f"Falta el checkpoint real en {CHECKPOINT_PATH}"
    assert VAL_INDEX_PATH.exists(), f"Falta el índice de validación real en {VAL_INDEX_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_unet_from_checkpoint(CHECKPOINT_PATH, device=device)

    val_ds = GeoForestDataset(VAL_INDEX_PATH)
    # Restricción: validación estrictamente con shuffle=False y sin WeightedRandomSampler
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    images_list: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, msks in val_loader:
            logits = model(imgs.to(device)).detach().cpu()
            logits_list.append(logits)
            targets_list.append(msks.detach().cpu())
            images_list.append(imgs.detach().cpu())

    all_logits = torch.cat(logits_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    all_images = torch.cat(images_list, dim=0)

    hco_indices = [
        i for i, r in enumerate(val_ds.records) if "18LUQ" in str(r.get("patch_id", ""))
    ]
    carpish_indices = [
        i for i, r in enumerate(val_ds.records) if "bosque_carpish" in str(r.get("patch_id", ""))
    ]
    huallaga_indices = [
        i for i, r in enumerate(val_ds.records) if "valle_huallaga" in str(r.get("patch_id", ""))
    ]

    return {
        "model": model,
        "metadata": metadata,
        "val_ds": val_ds,
        "val_loader": val_loader,
        "all_logits": all_logits,
        "all_targets": all_targets,
        "all_images": all_images,
        "hco_indices": hco_indices,
        "carpish_indices": carpish_indices,
        "huallaga_indices": huallaga_indices,
        "device": device,
    }


def test_real_checkpoint_and_independent_validation_split(real_validation_bundle):
    """Verifica integridad del checkpoint real y aislamiento espacial entre train y val."""
    val_ds: GeoForestDataset = real_validation_bundle["val_ds"]
    assert len(val_ds) >= 32, f"Se esperaban >= 32 escenas reales de validación (>2M px); encontradas {len(val_ds)}"
    assert len(real_validation_bundle["hco_indices"]) >= 32, (
        "El conjunto de validación debe incluir al menos 32 parches reales Sentinel-2 L2A de Huánuco (18LUQ)"
    )

    if TRAIN_INDEX_PATH.exists():
        train_ds = GeoForestDataset(TRAIN_INDEX_PATH)
        val_ids = {r["patch_id"] for r in val_ds.records}
        train_ids = {r["patch_id"] for r in train_ds.records}
        overlap = val_ids & train_ids
        assert not overlap, f"Fuga de datos detectada entre train y val: {overlap}"

        val_paths = {r["t0_path"] for r in val_ds.records}
        train_paths = {r["t0_path"] for r in train_ds.records}
        assert not (val_paths & train_paths), "Existen archivos GeoTIFF compartidos entre train y val"


def test_production_brier_and_ece_calibration_quality(real_validation_bundle):
    """Valida que el Brier Score y ECE-15 estén calibrados correctamente para producción."""
    model = real_validation_bundle["model"]
    all_logits = real_validation_bundle["all_logits"]
    all_targets = real_validation_bundle["all_targets"]
    hco_idx = real_validation_bundle["hco_indices"]

    # 1. Ajuste de temperatura desde T=1.0 con pesos de la U-Net congelados
    scaler = TemperatureScaler(initial_temperature=1.0)
    for param in model.parameters():
        param.requires_grad = True
    weights_before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    fitted_t = scaler.fit(
        real_validation_bundle["val_loader"],
        model,
        device=real_validation_bundle["device"],
    )

    # Verificar que los pesos de la U-Net quedaron congelados e intactos
    for param in model.parameters():
        assert not param.requires_grad
    for k, v in model.state_dict().items():
        assert torch.equal(weights_before[k], v), f"El parámetro {k} fue alterado durante fit()"

    assert 0.05 <= fitted_t <= 5.0, f"Temperatura óptima fuera del rango esperado: T = {fitted_t:.4f}"

    # 2. Evaluación sobre validación global independiente
    metrics = evaluate_calibration(all_logits, all_targets, temperature=fitted_t)

    assert metrics["brier_before"] > 0.0, "brier_before no puede ser 0.0 en producción"
    assert metrics["brier_after"] > 0.0, "brier_after no puede ser 0.0 en producción"
    assert metrics["brier_after"] <= metrics["brier_before"] + 1e-5, (
        f"La calibración no debe empeorar el Brier Score: {metrics['brier_before']:.5f} -> {metrics['brier_after']:.5f}"
    )
    assert metrics["brier_after"] < MAX_GLOBAL_BRIER_AFTER, (
        f"brier_after ({metrics['brier_after']:.5f}) supera el umbral máximo ({MAX_GLOBAL_BRIER_AFTER})"
    )

    # El Brier calibrado debe superar la referencia climatológica (Brier Skill Score > 0)
    assert metrics["brier_after"] < metrics["brier_reference"], (
        f"El Brier calibrado ({metrics['brier_after']:.5f}) debe ser menor que la referencia "
        f"climatológica ({metrics['brier_reference']:.5f})"
    )
    assert metrics["brier_skill_score_after"] > 0.50, (
        f"El Brier Skill Score post-calibración debe ser > 0.50; obtenido {metrics['brier_skill_score_after']:.4f}"
    )

    # ECE-15 debe ser estrictamente < 0.0250
    assert 0.0 < metrics["ece_after"] < MAX_ECE_AFTER, (
        f"ECE-15 post-calibración ({metrics['ece_after']:.4f}) debe ser < {MAX_ECE_AFTER}"
    )

    # 3. Evaluación específica sobre las escenas reales Sentinel-2 L2A de Huánuco (18LUQ)
    z_hco = all_logits[hco_idx]
    y_hco = all_targets[hco_idx]
    hco_metrics = evaluate_calibration(z_hco, y_hco, temperature=fitted_t)

    assert hco_metrics["brier_after"] < MAX_HUANUCO_BRIER_AFTER, (
        f"El Brier en Huánuco ({hco_metrics['brier_after']:.5f}) excede {MAX_HUANUCO_BRIER_AFTER}"
    )
    assert hco_metrics["brier_skill_score_after"] > 0.50, (
        f"Brier Skill Score en Huánuco insuficiente: {hco_metrics['brier_skill_score_after']:.4f}"
    )

    # 4. Verificar consistencia con el archivo de artefacto models/calibration.json
    assert CALIBRATION_JSON_PATH.exists(), f"Falta {CALIBRATION_JSON_PATH}"
    saved = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    assert saved["brier_before"] > 0.0 and saved["brier_after"] > 0.0, (
        f"models/calibration.json contiene valores nulos de Brier: {saved}"
    )
    assert saved["brier_after"] <= saved["brier_before"] + 1e-5
    assert saved["brier_skill_score_after"] > 0.50
    assert 0.0 < saved["ece_after"] < MAX_ECE_AFTER
    assert abs(saved["temperature"] - fitted_t) < 0.05


def test_explicit_iou_and_f1_thresholds_on_real_validation(real_validation_bundle):
    """Verifica umbrales explícitos de IoU y F1 sobre la validación real antes y después de calibrar."""
    all_logits = real_validation_bundle["all_logits"]
    all_targets = real_validation_bundle["all_targets"]
    hco_idx = real_validation_bundle["hco_indices"]
    carpish_idx = real_validation_bundle["carpish_indices"]
    huallaga_idx = real_validation_bundle["huallaga_indices"]

    saved = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    temp = float(saved["temperature"])

    probs_uncal = torch.sigmoid(all_logits)
    probs_cal = torch.sigmoid(all_logits / temp)

    # 1. Invarianza exacta de segmentación binaria en p = 0.5 (ya que sigmoid(0 / T) == 0.5)
    seg_uncal_all = compute_segmentation_metrics(probs_uncal, all_targets, threshold=0.5)
    seg_cal_all = compute_segmentation_metrics(probs_cal, all_targets, threshold=0.5)
    assert abs(seg_uncal_all["iou"] - seg_cal_all["iou"]) < 1e-7
    assert abs(seg_uncal_all["f1"] - seg_cal_all["f1"]) < 1e-7

    # 2. Umbrales explícitos en validación global independiente
    assert seg_cal_all["iou"] >= MIN_GLOBAL_VAL_IOU, (
        f"IoU global en validación ({seg_cal_all['iou']:.4f}) por debajo del umbral ({MIN_GLOBAL_VAL_IOU})"
    )
    assert seg_cal_all["f1"] >= MIN_GLOBAL_VAL_F1, (
        f"F1 global en validación ({seg_cal_all['f1']:.4f}) por debajo del umbral ({MIN_GLOBAL_VAL_F1})"
    )

    # 3. Umbrales explícitos por bioma (Carpish y Valle del Huallaga)
    if carpish_idx and huallaga_idx:
        seg_carpish = compute_segmentation_metrics(
            probs_cal[carpish_idx], all_targets[carpish_idx], threshold=0.5
        )
        seg_huallaga = compute_segmentation_metrics(
            probs_cal[huallaga_idx], all_targets[huallaga_idx], threshold=0.5
        )
        assert seg_carpish["iou"] >= MIN_HUANUCO_IOU, f"IoU Carpish ({seg_carpish['iou']:.4f}) < {MIN_HUANUCO_IOU}"
        assert seg_huallaga["iou"] >= MIN_HUANUCO_IOU, f"IoU Huallaga ({seg_huallaga['iou']:.4f}) < {MIN_HUANUCO_IOU}"

    # 4. Umbrales explícitos de producción en Huánuco Sentinel-2 L2A (Tile 18LUQ)
    seg_hco = compute_segmentation_metrics(probs_cal[hco_idx], all_targets[hco_idx], threshold=0.5)
    assert seg_hco["iou"] >= MIN_GLOBAL_VAL_IOU, (
        f"IoU en Huánuco 18LUQ ({seg_hco['iou']:.4f}) no alcanza el umbral ({MIN_GLOBAL_VAL_IOU})"
    )
    assert seg_hco["f1"] >= MIN_GLOBAL_VAL_F1, (
        f"F1 en Huánuco 18LUQ ({seg_hco['f1']:.4f}) no alcanza el umbral ({MIN_GLOBAL_VAL_F1})"
    )
    assert seg_hco["precision"] >= 0.8000, f"Precisión en Huánuco baja: {seg_hco['precision']:.4f}"
    assert seg_hco["recall"] >= 0.8000, f"Recall en Huánuco bajo: {seg_hco['recall']:.4f}"


def test_auroc_ap_invariance_and_onnx_production_parity(real_validation_bundle):
    """Verifica invarianza de AUROC/AP y paridad numérica del modelo ONNX en producción."""
    assert ONNX_PATH.exists(), f"Falta el modelo ONNX exportado en {ONNX_PATH}"
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    saved = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    temp = float(saved["temperature"])

    hco_idx = real_validation_bundle["hco_indices"]
    x_hco = real_validation_bundle["all_images"][hco_idx]
    z_hco = real_validation_bundle["all_logits"][hco_idx]
    y_hco = real_validation_bundle["all_targets"][hco_idx]

    z_hco_f64 = z_hco.double()
    p_uncal = torch.sigmoid(z_hco_f64).numpy().ravel()
    p_cal = torch.sigmoid(z_hco_f64 / temp).numpy().ravel()
    y_flat = (y_hco.numpy().ravel() > 0.5).astype(np.int32)

    # 1. Invarianza estricta de AUROC y Average Precision (monotonía de Temperature Scaling)
    auroc_before = roc_auc_score(y_flat, p_uncal)
    auroc_after = roc_auc_score(y_flat, p_cal)
    ap_before = average_precision_score(y_flat, p_uncal)
    ap_after = average_precision_score(y_flat, p_cal)

    assert abs(auroc_before - auroc_after) < 1e-10
    assert abs(ap_before - ap_after) < 1e-10
    assert auroc_after >= 0.95, f"AUROC en Huánuco ({auroc_after:.4f}) debe ser >= 0.95"
    assert ap_after >= 0.85, f"Average Precision en Huánuco ({ap_after:.4f}) debe ser >= 0.85"

    # 2. Paridad entre PyTorch CalibratedUNetWrapper y ONNX Runtime sobre las escenas reales de Huánuco
    wrapper = CalibratedUNetWrapper(real_validation_bundle["model"], temperature=temp).cpu().eval()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    pt_probs_list = []
    onnx_probs_list = []
    for start in range(0, len(x_hco), 8):
        batch_x = x_hco[start : start + 8].cpu()
        with torch.no_grad():
            pt_probs_list.append(wrapper(batch_x).numpy())
        onnx_probs_list.append(sess.run(["output"], {"input": batch_x.numpy()})[0])

    pt_probs = np.concatenate(pt_probs_list, axis=0)
    onnx_probs = np.concatenate(onnx_probs_list, axis=0)

    np.testing.assert_allclose(pt_probs, onnx_probs, rtol=1e-4, atol=1e-4)

    # 3. Verificar que las salidas directas del grafo ONNX cumplen los umbrales de Brier, IoU y F1
    onnx_brier = compute_brier_score(onnx_probs, y_hco)
    onnx_seg = compute_segmentation_metrics(onnx_probs, y_hco, threshold=0.5)
    assert 0.0 < onnx_brier < MAX_HUANUCO_BRIER_AFTER
    assert onnx_seg["iou"] >= MIN_GLOBAL_VAL_IOU
    assert onnx_seg["f1"] >= MIN_GLOBAL_VAL_F1
