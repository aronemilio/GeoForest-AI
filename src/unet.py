"""
Módulo de Arquitectura U-Net y Exportación ONNX para GeoForest-AI Huánuco.

Implementa:
- Bloques convolucionales modulares (DoubleConv, Down, Up, OutConv) en PyTorch.
- Arquitectura U-Net configurable en canales de entrada, filtros base y clases.
- Conexiones 'skip' con ajuste automático de dimensiones espaciales impares.
- Exportación y validación formal de modelos a formato ONNX con ejes dinámicos.
- Fábrica de modelos adaptados a constelaciones Sentinel-2 (B4, B8, B11 + NDVI, NBR).
- Generación de modelo ONNX de referencia determinista para inferencia rápida.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Configuración por defecto para Sentinel-2 en Huánuco:
# 5 canales: B4 (Rojo), B8 (NIR), B11 (SWIR1), NDVI, NBR
DEFAULT_IN_CHANNELS: int = 5
DEFAULT_NUM_CLASSES: int = 1
DEFAULT_BASE_FEATURES: int = 32
DEFAULT_S2_CHANNELS: tuple[str, ...] = ("B4", "B8", "B11", "NDVI", "NBR")


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------
class UNetError(Exception):
    """Excepción base para errores del módulo U-Net."""


class ModelConfigError(UNetError, ValueError):
    """Parámetros de configuración inválidos para la arquitectura U-Net."""


class ModelExportError(UNetError, RuntimeError):
    """Falla al exportar o validar el modelo ONNX en disco."""


# ---------------------------------------------------------------------------
# Bloques Fundamentales de U-Net
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    Bloque convolucional doble con normalización por lotes y activación ReLU:
    ``(Conv2d -> BatchNorm2d -> ReLU) x 2``.

    Mantiene las dimensiones espaciales idénticas usando ``kernel_size=3`` y
    ``padding=1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ModelConfigError("Los canales deben ser enteros positivos >= 1.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ModelConfigError(f"dropout debe estar en [0.0, 1.0), recibido: {dropout}")
        if mid_channels is None:
            mid_channels = out_channels

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    Bloque de submuestreo espacial:
    ``MaxPool2d(2) -> DoubleConv(in_channels, out_channels)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Bloque de sobremuestreo espacial y concatenación con conexión residual (skip):
    Sobremuestreo (``ConvTranspose2d`` o ``Upsample`` bilineal) ->
    Concatenación con skip-connection ->
    ``DoubleConv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Ajuste de bordes por si las dimensiones de x1 y x2 difieren por imparidad
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Capa de salida 1x1:
    ``Conv2d(in_channels, num_classes, kernel_size=1)``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Red Neuronal U-Net Modular
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """
    Arquitectura U-Net modular para segmentación semántica de coberturas forestales.

    Permite configurar número de canales espectrales de entrada (Sentinel-2 L2A),
    capacidad del modelo mediante `base_features` y número de clases de salida.

    Attributes:
        in_channels: Número de canales de entrada (por defecto 5).
        num_classes: Número de clases de predicción (por defecto 1, binario).
        base_features: Número de filtros base en el primer bloque (por defecto 32 o 64).
        bilinear: Utilizar sobremuestreo bilineal en lugar de convolución transpuesta.
        dropout: Tasa de Dropout2d espacial en el cuello de botella y decodificador.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        num_classes: int = DEFAULT_NUM_CLASSES,
        base_features: int = DEFAULT_BASE_FEATURES,
        bilinear: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ModelConfigError(f"in_channels debe ser >= 1, recibido: {in_channels}")
        if num_classes < 1:
            raise ModelConfigError(f"num_classes debe ser >= 1, recibido: {num_classes}")
        if base_features < 8:
            raise ModelConfigError(f"base_features debe ser >= 8, recibido: {base_features}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ModelConfigError(f"dropout debe estar en [0.0, 1.0), recibido: {dropout}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_features = base_features
        self.bilinear = bilinear
        self.dropout = float(dropout)

        factor = 2 if bilinear else 1
        dec_drop = round(self.dropout * 0.5, 4)

        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8, dropout=self.dropout)
        self.down4 = Down(
            base_features * 8,
            (base_features * 16) // factor,
            dropout=self.dropout,
        )

        self.up1 = Up(
            base_features * 16,
            (base_features * 8) // factor,
            bilinear=bilinear,
            dropout=dec_drop,
        )
        self.up2 = Up(
            base_features * 8,
            (base_features * 4) // factor,
            bilinear=bilinear,
            dropout=dec_drop,
        )
        self.up3 = Up(base_features * 4, (base_features * 2) // factor, bilinear=bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear=bilinear)
        self.outc = OutConv(base_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante de la red U-Net.

        Args:
            x: Tensor de entrada con forma ``(batch_size, in_channels, height, width)``.

        Returns:
            Tensor de logits con forma ``(batch_size, num_classes, height, width)``.
        """
        if not torch.jit.is_tracing():
            if x.ndim != 4:
                raise ValueError(
                    f"La entrada debe tener 4 dimensiones (B, C, H, W); recibida forma {x.shape}"
                )
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"Se esperaban {self.in_channels} canales pero la entrada contiene {x.shape[1]} canales."
                )

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ---------------------------------------------------------------------------
# Fábrica de Modelos Sentinel-2
# ---------------------------------------------------------------------------
def create_sentinel2_unet(
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_features: int = DEFAULT_BASE_FEATURES,
    bilinear: bool = False,
    weights_path: str | Path | None = None,
) -> UNet:
    """
    Instancia una arquitectura U-Net configurada para Sentinel-2 (Huánuco).

    Por defecto utiliza 5 canales de entrada (B4, B8, B11, NDVI, NBR) y
    1 canal de salida (probabilidad binaria de deforestación/alerta).

    Args:
        in_channels: Cantidad de canales espectrales y derivados.
        num_classes: Cantidad de clases objetivo.
        base_features: Canales de la primera capa convolucional.
        bilinear: Emplear interpolación bilineal en el decodificador.
        weights_path: Ruta opcional a un archivo de pesos ``.pt`` o ``.pth``.

    Returns:
        Instancia de :class:`UNet`.
    """
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_features=base_features,
        bilinear=bilinear,
    )
    if weights_path is not None:
        p = Path(weights_path)
        if not p.is_file():
            raise FileNotFoundError(f"Archivo de pesos no encontrado: {p}")
        state_dict = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Pesos cargados exitosamente desde %s", p)
    return model


# ---------------------------------------------------------------------------
# Exportación y Validación ONNX
# ---------------------------------------------------------------------------
def export_unet_to_onnx(
    model: nn.Module,
    file_path: str | Path,
    input_shape: tuple[int, ...] = (1, DEFAULT_IN_CHANNELS, 256, 256),
    dynamic_axes: bool | dict[str, dict[int, str]] = True,
    opset_version: int = 17,
    verify: bool = True,
) -> Path:
    """
    Exporta una red U-Net de PyTorch a formato ONNX y valida su integridad.

    Permite tamaños de entrada dinámicos (batch_size, alto, ancho) para
    facilitar la inferencia sobre parches o imágenes de distintas dimensiones.

    Args:
        model: Modelo PyTorch a exportar.
        file_path: Ruta del archivo de destino ``.onnx``.
        input_shape: Forma del tensor de prueba para el trazado del grafo.
        dynamic_axes: Habilitar ejes dinámicos (True/False o mapeo explícito).
        opset_version: Versión del conjunto de operadores ONNX (por defecto 17).
        verify: Si es True, ejecuta la verificación de integridad de ONNX.

    Returns:
        Ruta absoluta resuelta del modelo ONNX generado.

    Raises:
        ModelExportError: Si el modelo no puede exportarse o no pasa la validación.
    """
    out_path = Path(file_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_was_training = model.training
    model.eval()

    if dynamic_axes is True:
        dyn_axes: dict[str, dict[int, str]] | None = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
    elif dynamic_axes is False:
        dyn_axes = None
    else:
        dyn_axes = dynamic_axes

    dummy_input = torch.zeros(input_shape, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dyn_axes,
            dynamo=False,
        )
    except Exception as exc:
        raise ModelExportError(f"Error al exportar U-Net a ONNX en {out_path}: {exc}") from exc
    finally:
        if model_was_training:
            model.train()

    if verify:
        try:
            onnx_model = onnx.load(str(out_path))
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            raise ModelExportError(f"El modelo ONNX generado en {out_path} no es válido: {exc}") from exc

    logger.info("Modelo U-Net exportado exitosamente a ONNX: %s", out_path)
    return out_path


def generate_reference_model(
    output_path: str | Path = "models/unet_sentinel2_huanuco.onnx",
    in_channels: int = DEFAULT_IN_CHANNELS,
    num_classes: int = DEFAULT_NUM_CLASSES,
    base_features: int = DEFAULT_BASE_FEATURES,
    bilinear: bool = False,
    seed: int = 42,
) -> Path:
    """
    Genera un modelo ONNX de referencia para GeoForest-AI Huánuco.

    Inicializa los pesos de manera determinista usando una semilla fija, lo que
    proporciona un artefacto reproducible y listo para ejecutar pruebas de inferencia.

    Args:
        output_path: Destino donde se guardará el modelo ONNX.
        in_channels: Cantidad de canales (por defecto 5: B4, B8, B11, NDVI, NBR).
        num_classes: Cantidad de clases (por defecto 1: probabilidad de alerta).
        base_features: Filtros base (por defecto 32).
        bilinear: Tipo de sobremuestreo.
        seed: Semilla del generador pseudoaleatorio.

    Returns:
        Ruta resuelta del modelo generado.
    """
    torch.manual_seed(seed)
    model = create_sentinel2_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_features=base_features,
        bilinear=bilinear,
    )
    return export_unet_to_onnx(
        model=model,
        file_path=output_path,
        input_shape=(1, in_channels, 256, 256),
        dynamic_axes=True,
    )


if __name__ == "__main__":
    generate_reference_model()
