"""
Módulo de métricas de validación y benchmark del modelo U-Net ONNX cuantizado.

Replica `GeoForest-AI-Web/src/sim/metricas.ts` para GeoForest-AI Huánuco.
Permite evaluar y reportar IoU, F1, Precisión, Recall, Latencia en CPU
y Matriz de Confusión en hectáreas sobre el Bosque de Carpish y Valle del Huallaga.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MetricaClase:
    clase: str
    iou: float
    f1: float
    precision: float
    recall: float
    soporteHa: float


@dataclass
class MatrizConfusionHa:
    verdaderosPositivos: float
    falsosPositivos: float
    falsosNegativos: float
    verdaderosNegativos: float


@dataclass
class MetricasModeloSimuladas:
    arquitectura: str
    runtime: str
    tiempoInferenciaMsPorTile: float
    tamanoTilePx: int
    iouGlobal: float
    f1Global: float
    precisionGlobal: float
    recallGlobal: float
    areaValidadaHa: float
    matrizConfusionHa: MatrizConfusionHa
    clases: list[MetricaClase]


METRICAS_MODELO_SIMULADAS = MetricasModeloSimuladas(
    arquitectura="U-Net ResNet-34 Backbone (4 bandas: RGB + NIR)",
    runtime="ONNX Runtime (CPU / Int8 cuantizado)",
    tiempoInferenciaMsPorTile=18.4,
    tamanoTilePx=256,
    iouGlobal=0.884,
    f1Global=0.938,
    precisionGlobal=0.942,
    recallGlobal=0.935,
    areaValidadaHa=14250.0,
    matrizConfusionHa=MatrizConfusionHa(
        verdaderosPositivos=184.6,
        falsosPositivos=11.4,
        falsosNegativos=12.8,
        verdaderosNegativos=14041.2,
    ),
    clases=[
        MetricaClase(
            clase="Bosque denso (estable)",
            iou=0.942,
            f1=0.970,
            precision=0.975,
            recall=0.966,
            soporteHa=13500.0,
        ),
        MetricaClase(
            clase="Pérdida de cobertura (Deforestación)",
            iou=0.871,
            f1=0.931,
            precision=0.938,
            recall=0.924,
            soporteHa=112.4,
        ),
        MetricaClase(
            clase="Cicatriz de quema agrícola",
            iou=0.849,
            f1=0.918,
            precision=0.925,
            recall=0.912,
            soporteHa=85.0,
        ),
    ],
)


def calcular_metricas_escenario(total_alertas: int, area_total_ha: float) -> dict[str, Any]:
    """Calcula métricas sintéticas ajustadas a partir del volumen de alertas activas."""
    factor = min(1.0, max(0.7, total_alertas / 15.0))
    iou = round(0.85 + 0.04 * factor, 3)
    f1 = round(0.91 + 0.03 * factor, 3)
    precision = round(0.92 + 0.03 * factor, 3)
    recall = round(0.90 + 0.04 * factor, 3)

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "alertasProcesadas": total_alertas,
        "areaMonitoreadaHa": round(area_total_ha, 1),
        "latenciaMs": 18.2,
    }
