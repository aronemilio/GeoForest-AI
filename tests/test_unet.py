"""
Pruebas unitarias para el módulo U-Net (``src/unet.py``).

Verifica:
- Comportamiento y coherencia dimensional de los bloques (DoubleConv, Down, Up, OutConv).
- Paso hacia adelante (forward) de U-Net bajo distintas configuraciones y dimensiones.
- Validación de canales de entrada y número de clases.
- Manejo de excepciones ante hiperparámetros o entradas incompatibles.
- Exportación formal a formato ONNX y verificación de grafo en disco.
- Fábrica de modelos Sentinel-2 y generación de artefactos reproducibles.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch

from src.unet import (
    DEFAULT_BASE_FEATURES,
    DEFAULT_IN_CHANNELS,
    DEFAULT_NUM_CLASSES,
    DoubleConv,
    Down,
    ModelConfigError,
    OutConv,
    UNet,
    Up,
    create_sentinel2_unet,
    export_unet_to_onnx,
    generate_reference_model,
)


# ---------------------------------------------------------------------------
# Pruebas de Bloques Convolucionales
# ---------------------------------------------------------------------------
def test_double_conv_block() -> None:
    """Verifica que DoubleConv mantenga resolución espacial y mapee canales."""
    block = DoubleConv(in_channels=3, out_channels=16)
    x = torch.randn(2, 3, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)


def test_down_block() -> None:
    """Verifica que Down reduzca las dimensiones espaciales a la mitad."""
    down = Down(in_channels=16, out_channels=32)
    x = torch.randn(2, 16, 64, 64)
    out = down(x)
    assert out.shape == (2, 32, 32, 32)


def test_up_block_conv_transpose() -> None:
    """Verifica sobremuestreo con ConvTranspose2d y concatenación con skip connection."""
    up = Up(in_channels=32, out_channels=16, bilinear=False)
    x1 = torch.randn(2, 32, 16, 16)  # Proviniendo del nivel inferior
    x2 = torch.randn(2, 16, 32, 32)  # Skip connection del encoder
    out = up(x1, x2)
    assert out.shape == (2, 16, 32, 32)


def test_up_block_bilinear() -> None:
    """Verifica sobremuestreo bilineal con concatenación y dimensiones impares."""
    up = Up(in_channels=32, out_channels=16, bilinear=True)
    # x1 y x2 aportan 16 canales cada uno sumando 32 canales tras concatenación
    x1 = torch.randn(2, 16, 15, 15)
    x2 = torch.randn(2, 16, 31, 31)
    out = up(x1, x2)
    assert out.shape == (2, 16, 31, 31)


def test_out_conv_block() -> None:
    """Verifica convolución 1x1 final."""
    outc = OutConv(in_channels=16, out_channels=1)
    x = torch.randn(2, 16, 64, 64)
    out = outc(x)
    assert out.shape == (2, 1, 64, 64)


# ---------------------------------------------------------------------------
# Pruebas de Arquitectura U-Net
# ---------------------------------------------------------------------------
def test_unet_forward_shape() -> None:
    """Verifica la forma de salida (B, num_classes, H, W) para distintas resoluciones."""
    model = UNet(in_channels=5, num_classes=1, base_features=16, bilinear=False)
    model.eval()

    # Tamaño estándar
    x1 = torch.randn(1, 5, 128, 128)
    with torch.no_grad():
        out1 = model(x1)
    assert out1.shape == (1, 1, 128, 128)

    # Batch mayor
    x2 = torch.randn(3, 5, 64, 64)
    with torch.no_grad():
        out2 = model(x2)
    assert out2.shape == (3, 1, 64, 64)

    # Versión con bilinear=True
    model_bilinear = UNet(in_channels=5, num_classes=1, base_features=16, bilinear=True)
    model_bilinear.eval()
    with torch.no_grad():
        out3 = model_bilinear(x1)
    assert out3.shape == (1, 1, 128, 128)


def test_unet_channels() -> None:
    """Verifica configuración arbitraria de canales de entrada y clases de salida."""
    # Entrada de 3 canales (RGB), 2 clases
    model_rgb = UNet(in_channels=3, num_classes=2, base_features=16)
    x_rgb = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out_rgb = model_rgb(x_rgb)
    assert out_rgb.shape == (1, 2, 64, 64)

    # Entrada de 6 canales (Sentinel-2 extendido), 4 clases de cobertura
    model_multi = UNet(in_channels=6, num_classes=4, base_features=16)
    x_multi = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        out_multi = model_multi(x_multi)
    assert out_multi.shape == (2, 4, 64, 64)

    # Error al pasar canal incorrecto
    with pytest.raises(ValueError, match="canales"):
        model_multi(torch.randn(1, 3, 64, 64))

    # Error al pasar tensor no 4D
    with pytest.raises(ValueError, match="4 dimensiones"):
        model_multi(torch.randn(6, 64, 64))


def test_unet_config_errors() -> None:
    """Verifica excepciones ante hiperparámetros no válidos."""
    with pytest.raises(ModelConfigError, match="in_channels"):
        UNet(in_channels=0)

    with pytest.raises(ModelConfigError, match="num_classes"):
        UNet(num_classes=-1)

    with pytest.raises(ModelConfigError, match="base_features"):
        UNet(base_features=4)


# ---------------------------------------------------------------------------
# Pruebas de Exportación y Validación ONNX
# ---------------------------------------------------------------------------
def test_export_onnx_validity(tmp_path: Path) -> None:
    """Verifica que export_unet_to_onnx genere un archivo válido interpretable por ONNX Runtime."""
    model = UNet(in_channels=5, num_classes=1, base_features=16)
    onnx_file = tmp_path / "test_unet.onnx"

    exported_path = export_unet_to_onnx(
        model=model,
        file_path=onnx_file,
        input_shape=(1, 5, 128, 128),
        dynamic_axes=True,
        verify=True,
    )

    assert exported_path.is_file()
    assert exported_path == onnx_file.resolve()

    # Cargar y chequear con ONNX
    onnx_model = onnx.load(str(exported_path))
    onnx.checker.check_model(onnx_model)

    # Inferencia de prueba en ONNX Runtime con forma dinámica
    session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
    dummy_np = torch.randn(2, 5, 64, 64).numpy()
    results = session.run(None, {"input": dummy_np})

    assert len(results) == 1
    assert results[0].shape == (2, 1, 64, 64)


def test_create_sentinel2_unet() -> None:
    """Verifica que la función factory instancie el modelo estándar correctamente."""
    model = create_sentinel2_unet()
    assert isinstance(model, UNet)
    assert model.in_channels == DEFAULT_IN_CHANNELS
    assert model.num_classes == DEFAULT_NUM_CLASSES
    assert model.base_features == DEFAULT_BASE_FEATURES


def test_generate_reference_model(tmp_path: Path) -> None:
    """Verifica la generación del modelo de referencia determinista."""
    target_path = tmp_path / "sub" / "reference_model.onnx"
    out = generate_reference_model(
        output_path=target_path,
        in_channels=5,
        num_classes=1,
        base_features=16,
        seed=123,
    )
    assert out.is_file()
    assert out.stat().st_size > 0


def test_unet_dropout_and_parameter_reduction() -> None:
    """Verifica Dropout2d en UNet y reducción de parámetros frente a base_features=64."""
    heavy = UNet(in_channels=6, num_classes=1, base_features=64, bilinear=False, dropout=0.0)
    reg = UNet(in_channels=6, num_classes=1, base_features=32, bilinear=True, dropout=0.15)

    params_heavy = sum(p.numel() for p in heavy.parameters() if p.requires_grad)
    params_reg = sum(p.numel() for p in reg.parameters() if p.requires_grad)

    # De ~31.0M de parámetros a ~4.3M (>85% menos parámetros propensos a memorizar)
    assert params_reg < params_heavy * 0.20

    x = torch.randn(2, 6, 64, 64)
    reg.train()
    out_train = reg(x)
    assert out_train.shape == (2, 1, 64, 64)

    reg.eval()
    with torch.no_grad():
        out_eval_1 = reg(x)
        out_eval_2 = reg(x)
    assert torch.allclose(out_eval_1, out_eval_2)

    with pytest.raises(ModelConfigError, match="dropout"):
        UNet(in_channels=6, num_classes=1, base_features=16, dropout=1.0)

