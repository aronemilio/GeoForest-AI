"""
Pruebas unitarias para el módulo de simulación temporal y métricas U-Net (src/sim).
"""

import pandas as pd
import pytest

from src.sim import (
    METRICAS_MODELO_SIMULADAS,
    calcular_metricas_escenario,
    convertir_alertas_a_dataframe,
    crear_prng,
    generar_escenario_simulado,
)


def test_crear_prng_deterministico():
    """Valida que dos generadores Mulberry32 con la misma semilla produzcan secuencias idénticas."""
    r1 = crear_prng(42)
    r2 = crear_prng(42)

    seq1 = [r1() for _ in range(10)]
    seq2 = [r2() for _ in range(10)]

    assert seq1 == seq2
    assert all(0.0 <= x < 1.0 for x in seq1)


def test_generar_escenario_simulado_estructura():
    """Valida la generación del escenario temporal de alertas y serie temporal."""
    escenario = generar_escenario_simulado(semilla=42, pasos_semanales=8, alertas_por_paso=2)

    assert "alertas" in escenario
    assert "serie" in escenario

    alertas = escenario["alertas"]
    serie = escenario["serie"]

    assert len(alertas) == 16
    assert len(serie) == 8

    # Verificar estructura de alerta
    a0 = alertas[0]
    assert a0["origen"] == "SIMULADO"
    assert a0["zona"] in ("carpish", "huallaga")
    assert a0["severidad"] in ("Alta", "Media", "Baja")
    assert a0["areaHa"] > 0
    assert 0 <= a0["confianzaPct"] <= 100

    # Verificar serie temporal monotónica creciente
    for i in range(1, len(serie)):
        assert serie[i]["carpish"] >= serie[i - 1]["carpish"]
        assert serie[i]["huallaga"] >= serie[i - 1]["huallaga"]


def test_convertir_alertas_a_dataframe():
    """Valida la conversión a DataFrame de pandas con columnas esperadas."""
    escenario = generar_escenario_simulado(semilla=101, pasos_semanales=4, alertas_por_paso=2)
    df = convertir_alertas_a_dataframe(escenario["alertas"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    expected_cols = {
        "id_alerta",
        "origen_dato",
        "fecha",
        "zona",
        "sector",
        "distrito",
        "tipo_evento",
        "severidad",
        "area_ha",
        "confianza_pct",
        "ndvi_cambio",
        "nbr_cambio",
        "lat",
        "lon",
        "delta_lat",
        "delta_lon",
        "estado",
    }
    assert expected_cols.issubset(set(df.columns))


def test_metricas_modelo_simuladas():
    """Valida la consistencia de las métricas de benchmark del modelo U-Net ONNX."""
    m = METRICAS_MODELO_SIMULADAS
    assert m.iouGlobal > 0.85
    assert m.f1Global > 0.90
    assert m.tiempoInferenciaMsPorTile < 25.0
    assert len(m.clases) == 3

    ajuste = calcular_metricas_escenario(total_alertas=10, area_total_ha=45.0)
    assert "iou" in ajuste
    assert "f1" in ajuste
    assert ajuste["alertasProcesadas"] == 10
    assert ajuste["areaMonitoreadaHa"] == 45.0
