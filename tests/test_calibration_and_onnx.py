"""Pruebas unitarias de la Fase C: Calibración Probabilística y Exportación a ONNX.

Valida:
1. Integridad de AUROC y Ranking: T no altera el orden de probabilidades ni métricas de ranking.
2. Congelamiento estricto de pesos: los parámetros de la U-Net no cambian durante el ajuste de T.
3. Convergencia y ECE-15: L-BFGS optimiza T y reduce ECE-15 a < 0.025.
4. Exportación y Validación de Grafo ONNX: onnx.checker valida el grafo con wrapper y dynamic_axes.
5. Inferencia ONNXRuntime: parches de dimensiones variables respetan los ejes dinámicos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

# Configurar path del proyecto
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_onnx import (
    CalibratedUNetWrapper,
    export_calibrated_onnx,
    load_unet_from_checkpoint,
    run_calibration_and_export,
)
from src.data.dataset import GeoForestDataset
from src.models.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_ece,
    evaluate_calibration,
)
from src.unet import UNet


# ---------------------------------------------------------------------------
# Helpers sintéticos
# ---------------------------------------------------------------------------
def _synthetic_dataset_for_val(n_samples: int = 20, in_channels: int = 6, h: int = 64, w: int = 64):
    """Crea dataset sintético en memoria para pruebas de validación."""
    records = []
    rng = np.random.default_rng(123)
    for i in range(n_samples):
        has_change = int(i % 4 == 0)
        t0 = rng.integers(1000, 4000, size=(in_channels // 2, h, w), dtype=np.int32)
        t1 = rng.integers(1000, 4000, size=(in_channels // 2, h, w), dtype=np.int32)
        mask = np.zeros((1, h, w), dtype=np.uint8)
        if has_change:
            mask[0, 10:20, 10:20] = 1
        records.append({
            "patch_id": f"val_patch_{i}",
            "has_change": has_change,
            "positive_pixels": int(mask.sum()),
            "t0_data": t0,
            "t1_data": t1,
            "mask_data": mask,
        })
    return GeoForestDataset(records)


# ---------------------------------------------------------------------------
# Test 1: Monotonicidad, Ranking y Preservación de AUROC / AP
# ---------------------------------------------------------------------------
def test_temperature_scaling_preserves_ranking_and_auroc():
    """Confirma matemáticamente que T > 0 no altera el ranking de predicciones ni AUROC/AP."""
    rng = np.random.default_rng(42)
    n = 1000
    logits_np = rng.normal(loc=0.0, scale=2.0, size=n).astype(np.float64)
    sorted_idx = np.argsort(logits_np)
    targets = (rng.uniform(0.0, 1.0, size=n) < (1.0 / (1.0 + np.exp(-logits_np)))).astype(int)

    probs_base = 1.0 / (1.0 + np.exp(-logits_np))

    for temp in [0.5, 1.5, 2.8, 5.0]:
        probs_cal = 1.0 / (1.0 + np.exp(-logits_np / temp))

        # 1. Monotonicidad estricta: z_i < z_j implica p_cal_i <= p_cal_j
        sorted_probs = probs_cal[sorted_idx]
        assert np.all(np.diff(sorted_probs) >= 0.0), (
            f"La función no es monótona creciente con T={temp}"
        )

        # 2. Invarianza exacta del AUROC
        auc_base = roc_auc_score(targets, probs_base)
        auc_cal = roc_auc_score(targets, probs_cal)
        assert auc_cal == pytest.approx(auc_base, abs=1e-5), (
            f"AUROC difiere tras aplicar T={temp}: base={auc_base}, cal={auc_cal}"
        )

        # 3. Invarianza exacta del Average Precision (AP)
        ap_base = average_precision_score(targets, probs_base)
        ap_cal = average_precision_score(targets, probs_cal)
        assert ap_cal == pytest.approx(ap_base, abs=1e-5), (
            f"AP difiere tras aplicar T={temp}: base={ap_base}, cal={ap_cal}"
        )


# ---------------------------------------------------------------------------
# Test 2: Convergencia de L-BFGS y Reducción de ECE < 0.025
# ---------------------------------------------------------------------------
def test_temperature_scaler_convergence_and_ece_reduction():
    """Valida que la optimización L-BFGS de T reduzca el ECE-15 a menos de 0.025."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Simular modelo sobreconfiado: probabilidades reales con logits sobreescalados (factor 3.0)
    n_samples = 40_000
    # Probabilidades reales bien calibradas
    p_true = rng.uniform(0.05, 0.95, size=n_samples)
    targets = (rng.uniform(0.0, 1.0, size=n_samples) < p_true).astype(np.float32)

    # Logits reales inversos de p_true
    z_true = np.log(p_true / (1.0 - p_true))
    # Logits sobreconfiados generados por red neuronal profunda sin calibrar
    overconfident_factor = 2.8
    z_overconfident = z_true * overconfident_factor

    logits_t = torch.tensor(z_overconfident, dtype=torch.float32)
    targets_t = torch.tensor(targets, dtype=torch.float32)

    # Medir calibración antes
    probs_uncal = torch.sigmoid(logits_t)
    ece_before = compute_ece(probs_uncal, targets_t, n_bins=15)
    brier_before = compute_brier_score(probs_uncal, targets_t)

    # Ajustar calibrador de temperatura con L-BFGS
    scaler = TemperatureScaler(initial_temperature=1.0)
    fitted_t = scaler.fit_logits(logits_t, targets_t, max_iter=80, lr=0.05)

    # La temperatura debe compensar el factor de sobreconfianza
    assert 2.0 <= fitted_t <= 3.5, f"T ajustado ({fitted_t}) alejado del factor esperado"

    # Medir calibración después
    probs_cal = scaler.predict_proba(logits_t)
    ece_after = compute_ece(probs_cal, targets_t, n_bins=15)
    brier_after = compute_brier_score(probs_cal, targets_t)

    # Criterio estricto de aceptación: ECE-15 < 0.025
    assert ece_after < 0.025, (
        f"ECE posterior a la calibración ({ece_after:.4f}) no cumple el umbral < 0.025"
    )
    assert ece_after < ece_before, "ECE no disminuyó tras calibración"
    assert brier_after <= brier_before, "Brier score empeoró tras calibración"


# ---------------------------------------------------------------------------
# Test 3: Congelamiento Estricto de Pesos de U-Net
# ---------------------------------------------------------------------------
def test_unet_weights_strictly_frozen_during_fit():
    """Verifica que los pesos y sesgos de la U-Net estén congelados y no cambien durante fit()."""
    model = UNet(in_channels=6, num_classes=1, base_features=8, bilinear=True)
    for p in model.parameters():
        p.requires_grad = True

    # Copia bit a bit de todos los parámetros iniciales
    initial_weights = [p.clone().detach() for p in model.parameters()]

    val_ds = _synthetic_dataset_for_val(n_samples=4, in_channels=6, h=32, w=32)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    scaler = TemperatureScaler(initial_temperature=1.0)
    scaler.fit(val_loader, model, device="cpu", max_iter=5)

    # 1. Comprobar que todos los parámetros quedaron con requires_grad == False
    for p in model.parameters():
        assert not p.requires_grad, "Un parámetro del modelo no quedó congelado"

    # 2. Comprobar que los valores de los pesos son estrictamente idénticos
    for p, init_p in zip(model.parameters(), initial_weights):
        assert torch.equal(p, init_p), "Los pesos de la U-Net fueron modificados durante fit()"


# ---------------------------------------------------------------------------
# Test 4: Exportación a ONNX y Validación del Grafo con onnx.checker
# ---------------------------------------------------------------------------
def test_export_calibrated_onnx_and_checker_validation(tmp_path):
    """Valida la exportación con dynamic_axes y la verificación exitosa de onnx.checker."""
    model = UNet(in_channels=6, num_classes=1, base_features=8, bilinear=True)
    onnx_path = tmp_path / "unet_test_calibrated.onnx"

    opt_temp = 2.45
    exported_path = export_calibrated_onnx(
        model=model,
        temperature=opt_temp,
        output_path=onnx_path,
        in_channels=6,
        opset_version=17,
        verify=True,
    )

    assert exported_path.exists()

    # 1. Validación formal con onnx.checker
    loaded_onnx = onnx.load(str(exported_path))
    onnx.checker.check_model(loaded_onnx)

    # 2. Comprobar nombres de entrada y salida
    assert loaded_onnx.graph.input[0].name == "input"
    assert loaded_onnx.graph.output[0].name == "output"


# ---------------------------------------------------------------------------
# Test 5: Inferencia ONNXRuntime y Ejes Dinámicos
# ---------------------------------------------------------------------------
def test_onnxruntime_dynamic_axes_and_numerical_match(tmp_path):
    """Verifica que el modelo ONNX admita batch_size, height y width variables."""
    model = UNet(in_channels=6, num_classes=1, base_features=8, bilinear=True)
    model.eval()
    opt_temp = 2.0
    onnx_path = tmp_path / "unet_dyn.onnx"

    export_calibrated_onnx(
        model=model,
        temperature=opt_temp,
        output_path=onnx_path,
        in_channels=6,
        opset_version=17,
        verify=True,
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wrapper = CalibratedUNetWrapper(model, temperature=opt_temp)
    wrapper.eval()

    # Probar diferentes formas espaciales y lotes
    test_shapes = [
        (1, 6, 256, 256),
        (2, 6, 128, 128),
        (1, 6, 96, 96),
    ]

    for shape in test_shapes:
        dummy = np.random.randn(*shape).astype(np.float32)

        # Inferencia ONNX Runtime
        ort_inputs = {"input": dummy}
        ort_outs = session.run(["output"], ort_inputs)[0]

        # Inferencia PyTorch con Wrapper
        with torch.no_grad():
            py_outs = wrapper(torch.from_numpy(dummy)).numpy()

        # Dimensiones coincidentes
        assert ort_outs.shape == (shape[0], 1, shape[2], shape[3])
        # Valores de probabilidad acotados estrictamente en [0, 1]
        assert np.all(ort_outs >= 0.0) and np.all(ort_outs <= 1.0)
        # Coincidencia numérica estricta entre ONNX y PyTorch
        assert np.allclose(ort_outs, py_outs, atol=1e-5), (
            f"Discrepancia entre ONNX y PyTorch para forma {shape}"
        )


# ---------------------------------------------------------------------------
# Test 6: Flujo Completo run_calibration_and_export
# ---------------------------------------------------------------------------
def test_full_workflow_run_calibration_and_export(tmp_path):
    """Comprueba el flujo integral de guardado de métricas y exportación."""
    model = UNet(in_channels=6, num_classes=1, base_features=8, bilinear=True)
    ckpt_path = tmp_path / "models/unet_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "val_iou": 0.85}, ckpt_path)

    out_onnx = tmp_path / "models/unet_geoforest.onnx"
    calib_json = tmp_path / "models/calibration.json"

    res = run_calibration_and_export(
        checkpoint_path=ckpt_path,
        output_onnx_path=out_onnx,
        calibration_json_path=calib_json,
        val_index_path=None,
        temperature_override=2.15,
    )

    assert out_onnx.exists()
    assert calib_json.exists()
    assert res["temperature"] == 2.15

    with open(calib_json, encoding="utf-8") as f:
        data = json.load(f)
    assert data["temperature"] == 2.15
    assert "fitted_at_utc" in data
