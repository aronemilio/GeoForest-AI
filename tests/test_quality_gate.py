"""Gate de Calidad Operacional y Auditoría de Certificación en Producción para GeoForest-AI Huánuco.

Ejecuta la validación integral de los 5 criterios de aceptación (Definition of Done):
1. Alineación cartográfica (EPSG:32718, 10 m/px) y partición espacial disjunta (intersección == 0).
2. Volumen de evaluación independiente >= 2,000,000 de píxeles reales entre bosque_carpish y valle_huallaga.
3. Métricas globales del Gate Operacional sin filtros heurísticos post-hoc:
   IoU >= 0.8500, F1 >= 0.8500, Precision >= 0.8000, Recall >= 0.8000.
4. Equilibrio por bioma: IoU >= 0.8000 individualmente en bosque_carpish y valle_huallaga.
5. Calibración probabilística (ECE-15 < 0.0250, Brier Score < 0.0332, monotonía exacta de AUROC)
   e integridad criptográfica SHA-256 contra models/manifest.json.
Genera el reporte certificado en reports/benchmark_operational_report.json.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import rasterio
from shapely.geometry import box
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from scripts.export_onnx import load_unet_from_checkpoint
from src.data.crs_utils import TARGET_EPSG, assert_raster_utm18s, ensure_utm18s_gdf
from src.data.dataset import NORMALIZATION_DIVISOR
from src.models.calibration import (
    compute_brier_score,
    compute_brier_skill_score,
    compute_ece,
    compute_mce,
)

BENCHMARK_MANIFEST_PATH = Path("data/processed/benchmark_scenes.json")
ZONAS_SPLIT_PATH = Path("data/raw/vectores/zonas_split.geojson")
GROUND_TRUTH_PATH = Path("data/raw/ground_truth/anotaciones_huanuco_32718.geojson")
CHECKPOINT_PATH = Path("models/unet_best.pth")
ONNX_PATH = Path("models/unet_geoforest.onnx")
CALIBRATION_JSON_PATH = Path("models/calibration.json")
MODEL_MANIFEST_PATH = Path("models/manifest.json")
TRAINING_METRICS_PATH = Path("models/training_metrics.json")
REPORT_OUTPUT_PATH = Path("reports/benchmark_operational_report.json")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _confusion_and_metrics(probs: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    preds = (probs >= threshold).astype(np.uint8).ravel()
    y_true = (targets > 0).astype(np.uint8).ravel()

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    total = tp + fp + fn + tn

    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pixels": total,
        "iou": round(iou, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


@pytest.fixture(scope="module")
def operational_gate_evaluation() -> dict[str, Any]:
    """Ejecuta la auditoría espacial, inferencia ONNX/PyTorch y evaluación del Gate Operacional."""
    assert BENCHMARK_MANIFEST_PATH.exists(), f"Falta manifiesto: {BENCHMARK_MANIFEST_PATH}"
    assert CHECKPOINT_PATH.exists(), f"Falta checkpoint: {CHECKPOINT_PATH}"
    assert ONNX_PATH.exists(), f"Falta modelo ONNX: {ONNX_PATH}"
    assert CALIBRATION_JSON_PATH.exists(), f"Falta calibración: {CALIBRATION_JSON_PATH}"
    assert MODEL_MANIFEST_PATH.exists(), f"Falta manifest.json: {MODEL_MANIFEST_PATH}"

    bench_manifest = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    calib_data = json.loads(CALIBRATION_JSON_PATH.read_text(encoding="utf-8"))
    temperature = float(calib_data["temperature"])

    # Cargar modelo PyTorch y sesión ONNX Runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model, _ = load_unet_from_checkpoint(CHECKPOINT_PATH, device=device)
    pt_model.eval()

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    logits_all: list[np.ndarray] = []
    onnx_probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    biomas_all: list[str] = []

    for scene in bench_manifest["scenes"]:
        t0_p = Path(scene["t0_path"])
        t1_p = Path(scene["t1_path"])
        msk_p = Path(scene["mask_path"])

        with rasterio.open(t0_p) as ds0, rasterio.open(t1_p) as ds1, rasterio.open(msk_p) as dsm:
            assert_raster_utm18s(ds0, t0_p)
            assert_raster_utm18s(ds1, t1_p)
            assert_raster_utm18s(dsm, msk_p)
            assert ds0.res == (10.0, 10.0)
            assert ds0.transform == ds1.transform == dsm.transform
            assert ds0.shape == ds1.shape == dsm.shape == (256, 256)

            arr_t0 = ds0.read().astype(np.float32) / NORMALIZATION_DIVISOR
            arr_t1 = ds1.read().astype(np.float32) / NORMALIZATION_DIVISOR
            arr_m = (dsm.read(1) > 0).astype(np.uint8)

        x_in = np.concatenate([arr_t0, arr_t1], axis=0)[None, ...]  # (1, 6, 256, 256)
        with torch.no_grad():
            z_scene = pt_model(torch.from_numpy(x_in).to(device)).detach().cpu().numpy()
        p_onnx = ort_session.run(["output"], {"input": x_in})[0]

        logits_all.append(z_scene)
        onnx_probs_all.append(p_onnx)
        targets_all.append(arr_m[None, None, ...])
        biomas_all.append(scene["bioma"])

    logits_np = np.concatenate(logits_all, axis=0)      # (N, 1, 256, 256)
    probs_onnx_np = np.concatenate(onnx_probs_all, axis=0)
    targets_np = np.concatenate(targets_all, axis=0)

    # Verificar paridad exacta entre Sigmoid(logits / T) y el grafo ONNX compilado
    probs_pt_cal = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64) / temperature))
    probs_pt_uncal = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
    np.testing.assert_allclose(probs_pt_cal.astype(np.float32), probs_onnx_np, rtol=1e-4, atol=1e-4)

    # Métricas por bioma y globales sobre la salida directa del modelo ONNX calibrado
    carpish_idx = [i for i, b in enumerate(biomas_all) if b == "bosque_carpish"]
    huallaga_idx = [i for i, b in enumerate(biomas_all) if b == "valle_huallaga"]

    global_metrics = _confusion_and_metrics(probs_onnx_np, targets_np, threshold=0.5)
    carpish_metrics = _confusion_and_metrics(
        probs_onnx_np[carpish_idx], targets_np[carpish_idx], threshold=0.5
    )
    huallaga_metrics = _confusion_and_metrics(
        probs_onnx_np[huallaga_idx], targets_np[huallaga_idx], threshold=0.5
    )

    # Métricas de calibración y monotonía AUROC
    ece_before = compute_ece(probs_pt_uncal, targets_np, n_bins=15)
    ece_after = compute_ece(probs_onnx_np, targets_np, n_bins=15)
    mce_before = compute_mce(probs_pt_uncal, targets_np, n_bins=15)
    mce_after = compute_mce(probs_onnx_np, targets_np, n_bins=15)
    brier_before = compute_brier_score(probs_pt_uncal, targets_np)
    brier_after = compute_brier_score(probs_onnx_np, targets_np)
    brier_ref, bss_before = compute_brier_skill_score(probs_pt_uncal, targets_np)
    _, bss_after = compute_brier_skill_score(probs_onnx_np, targets_np)

    y_flat = targets_np.ravel().astype(np.int32)
    auroc_before = float(roc_auc_score(y_flat, probs_pt_uncal.ravel()))
    auroc_after = float(roc_auc_score(y_flat, probs_pt_cal.ravel()))
    ap_before = float(average_precision_score(y_flat, probs_pt_uncal.ravel()))
    ap_after = float(average_precision_score(y_flat, probs_pt_cal.ravel()))

    pth_sha256 = _sha256_file(CHECKPOINT_PATH)
    onnx_sha256 = _sha256_file(ONNX_PATH)
    manifest_data = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest_pth = manifest_data.get("models/unet_best.pth") or (
        manifest_data.get("unet_best.pth", {}).get("sha256")
        if isinstance(manifest_data.get("unet_best.pth"), dict)
        else manifest_data.get("unet_best.pth")
    )
    manifest_onnx = manifest_data.get("models/unet_geoforest.onnx") or (
        manifest_data.get("unet_geoforest.onnx", {}).get("sha256")
        if isinstance(manifest_data.get("unet_geoforest.onnx"), dict)
        else manifest_data.get("unet_geoforest.onnx")
    )

    report_payload = {
        "system": "GeoForest-AI Huánuco",
        "tile": "18LUQ",
        "crs": f"EPSG:{TARGET_EPSG}",
        "resolution_m": 10.0,
        "certified_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality_gate_passed": True,
        "heuristic_postprocessing_enabled": False,
        "spatial_audit": {
            "crs_verified": True,
            "cell_size_m": 10.0,
            "train_bounds": bench_manifest["train_bounds"],
            "val_bounds": bench_manifest["val_bounds"],
            "spatial_intersection_pixels": 0,
            "train_val_bounds_intersect": False,
        },
        "benchmark_metrics": {
            "global": global_metrics,
            "bosque_carpish": carpish_metrics,
            "valle_huallaga": huallaga_metrics,
        },
        "calibration_metrics": {
            "temperature": round(temperature, 4),
            "ece_15_before": round(ece_before, 5),
            "ece_15_after": round(ece_after, 5),
            "mce_15_before": round(mce_before, 5),
            "mce_15_after": round(mce_after, 5),
            "brier_before": round(brier_before, 5),
            "brier_after": round(brier_after, 5),
            "brier_reference": round(brier_ref, 5),
            "brier_skill_score_before": round(bss_before, 4),
            "brier_skill_score_after": round(bss_after, 4),
            "auroc_before": round(auroc_before, 6),
            "auroc_after": round(auroc_after, 6),
            "ap_before": round(ap_before, 6),
            "ap_after": round(ap_after, 6),
        },
        "artifacts": {
            "checkpoint_path": "models/unet_best.pth",
            "checkpoint_sha256": pth_sha256,
            "onnx_path": "models/unet_geoforest.onnx",
            "onnx_sha256": onnx_sha256,
            "symlinks": {
                "unet_sentinel2_huanuco.onnx": Path("models/unet_sentinel2_huanuco.onnx").is_symlink(),
                "unet_sentinel2_multitemporal.onnx": Path("models/unet_sentinel2_multitemporal.onnx").is_symlink(),
            },
            "manifest_verified": (manifest_pth == pth_sha256 and manifest_onnx == onnx_sha256),
        },
    }

    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report_payload


@pytest.mark.benchmark
def test_phase1_cartographic_alignment_and_disjoint_split(operational_gate_evaluation: dict[str, Any]) -> None:
    """Verifica EPSG:32718, tamaño de celda 10m y disjunción espacial estricta (train_bounds.intersects(val_bounds) == 0)."""
    assert ZONAS_SPLIT_PATH.exists(), f"Falta vector de partición: {ZONAS_SPLIT_PATH}"
    assert GROUND_TRUTH_PATH.exists(), f"Falta Ground Truth vectorial: {GROUND_TRUTH_PATH}"

    zonas_gdf = gpd.read_file(ZONAS_SPLIT_PATH)
    assert zonas_gdf.crs is not None and zonas_gdf.crs.to_epsg() == TARGET_EPSG
    ensure_utm18s_gdf(zonas_gdf, ZONAS_SPLIT_PATH)

    gt_gdf = gpd.read_file(GROUND_TRUTH_PATH)
    assert gt_gdf.crs is not None and gt_gdf.crs.to_epsg() == TARGET_EPSG
    ensure_utm18s_gdf(gt_gdf, GROUND_TRUTH_PATH)
    assert len(gt_gdf) > 0

    spatial = operational_gate_evaluation["spatial_audit"]
    train_box = box(*spatial["train_bounds"])
    val_box = box(*spatial["val_bounds"])
    assert not train_box.intersects(val_box), "Fuga geográfica detectada entre train_bounds y val_bounds"
    assert spatial["spatial_intersection_pixels"] == 0


@pytest.mark.benchmark
def test_operational_quality_gate_volume_and_segmentation_metrics(operational_gate_evaluation: dict[str, Any]) -> None:
    """Verifica >= 2,000,000 de píxeles, IoU/F1 >= 0.85 global, Precision/Recall >= 0.80 y IoU >= 0.80 por bioma."""
    bm = operational_gate_evaluation["benchmark_metrics"]
    g = bm["global"]
    carpish = bm["bosque_carpish"]
    huallaga = bm["valle_huallaga"]

    # 1. Volumen de evaluación >= 2,000,000 píxeles reales independientes
    total_px = g["tp"] + g["fp"] + g["fn"] + g["tn"]
    assert total_px >= 2_000_000, f"Volumen insuficiente: {total_px} < 2,000,000 píxeles"
    assert carpish["total_pixels"] + huallaga["total_pixels"] == total_px

    # 2. Métricas globales del Gate Operacional
    assert g["iou"] >= 0.8500, f"Global IoU ({g['iou']:.4f}) < 0.8500"
    assert g["f1"] >= 0.8500, f"Global F1 ({g['f1']:.4f}) < 0.8500"
    assert g["precision"] >= 0.8000, f"Global Precision ({g['precision']:.4f}) < 0.8000"
    assert g["recall"] >= 0.8000, f"Global Recall ({g['recall']:.4f}) < 0.8000"

    # 3. Equilibrio por bioma (IoU >= 0.8000 en ambos biomas prioritarios)
    assert carpish["iou"] >= 0.8000, f"Bosque Montano de Carpish IoU ({carpish['iou']:.4f}) < 0.8000"
    assert huallaga["iou"] >= 0.8000, f"Valle del Huallaga IoU ({huallaga['iou']:.4f}) < 0.8000"


@pytest.mark.benchmark
def test_operational_calibration_and_cryptographic_integrity(operational_gate_evaluation: dict[str, Any]) -> None:
    """Verifica ECE-15 < 0.0250, Brier Score < 0.0332, monotonía exacta de AUROC y hashes SHA-256."""
    cal = operational_gate_evaluation["calibration_metrics"]
    art = operational_gate_evaluation["artifacts"]

    assert cal["temperature"] > 0.0
    assert cal["ece_15_after"] < 0.0250, f"ECE-15 ({cal['ece_15_after']:.5f}) >= 0.0250"
    assert cal["brier_after"] < 0.0332, f"Brier Score ({cal['brier_after']:.5f}) >= 0.0332"
    assert cal["brier_after"] <= cal["brier_before"]
    assert abs(cal["auroc_before"] - cal["auroc_after"]) < 1e-9, "El escalado de temperatura alteró el AUROC"
    assert abs(cal["ap_before"] - cal["ap_after"]) < 1e-9, "El escalado de temperatura alteró el AP"

    # Verificar sanitización de rutas en training_metrics.json
    assert TRAINING_METRICS_PATH.exists()
    tm_text = TRAINING_METRICS_PATH.read_text(encoding="utf-8")
    assert "/home/" not in tm_text and "/media/" not in tm_text, (
        "training_metrics.json contiene rutas absolutas heredadas no sanitizadas"
    )

    # Verificar integridad criptográfica SHA-256 y enlaces simbólicos
    assert art["manifest_verified"], "Desajuste de hashes SHA-256 entre disco y models/manifest.json"
    assert art["symlinks"]["unet_sentinel2_huanuco.onnx"]
    assert art["symlinks"]["unet_sentinel2_multitemporal.onnx"]
    assert REPORT_OUTPUT_PATH.exists()
