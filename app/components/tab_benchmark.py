"""Componente para la pestaña de Benchmark U-Net ONNX."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.onnx_demo import peso_modelo_onnx_mb
from app.styles.theme import divisor_bosque
from src.sim import METRICAS_MODELO_REFERENCIA


def render_tab_benchmark() -> None:
    """Reporte de desempeño y especificaciones técnicas del motor U-Net ONNX."""
    st.subheader(
        "Benchmark del Motor U-Net ONNX — Métricas de Validación Operacional",
        icon=":material/speed:",
    )
    st.caption(
        "Reporte de precisión, latencia y consumo del motor de inferencia sobre escenas "
        "multitemporales Sentinel-2 L2A en el Bosque de Carpish y Valle del Alto Huallaga."
    )
    st.info(
        "🛰️ **Pipeline de Inferencia Calibrado:** El motor combina pesos entrenados "
        "(`models/unet_best.pth` exportados a `models/unet_geoforest.onnx`) con **Temperature Scaling "
        "($T=0.5$)**, **umbral adaptativo de Otsu** y **filtro biofísico espectral** "
        "($\\Delta\\text{NDVI} < -0.12$ o $\\Delta\\text{NBR} < -0.12$) para maximizar la relación "
        "señal-ruido en bordes de deforestación real.",
        icon=":material/verified:",
    )

    m = METRICAS_MODELO_REFERENCIA
    peso_mb = peso_modelo_onnx_mb()

    bm_c1, bm_c2, bm_c3, bm_c4 = st.columns(4)
    bm_c1.metric(
        "IoU Global (Jaccard)",
        f"{m.iouGlobal * 100:.1f}%",
        "Umbral operacional >= 85%",
        delta_color="normal",
    )
    bm_c2.metric(
        "F1-Score Global (Dice)",
        f"{m.f1Global * 100:.1f}%",
        "Balance precisión / exhaustividad",
        delta_color="normal",
    )
    bm_c3.metric(
        "Latencia CPU por Tile",
        f"{m.tiempoInferenciaMsPorTile:.1f} ms",
        "Tile 256x256 px (< 25 ms)",
        delta_color="normal",
    )
    bm_c4.metric(
        "Peso del grafo .onnx servido",
        f"{peso_mb:.1f} MB" if peso_mb else "no encontrado",
        "FP32 · ONNX Runtime CPU Execution Provider",
        delta_color="off",
    )

    divisor_bosque()

    col_bm1, col_bm2 = st.columns([1.4, 1])

    with col_bm1:
        st.markdown("##### 📊 Desglose de Desempeño por Clase de Cobertura")
        filas_clases = []
        for c in m.clases:
            filas_clases.append({
                "Clase de Cobertura": c.clase,
                "IoU (%)": f"{c.iou * 100:.1f}%",
                "F1-Score (%)": f"{c.f1 * 100:.1f}%",
                "Precisión (%)": f"{c.precision * 100:.1f}%",
                "Recall (%)": f"{c.recall * 100:.1f}%",
                "Área Validada": f"{c.soporteHa:,.1f} ha",
            })
        st.dataframe(pd.DataFrame(filas_clases), width="stretch", hide_index=True)

        st.markdown("##### 🧮 Matriz de Confusión en Hectáreas (Área Validada: 14,250 ha)")
        mc = m.matrizConfusionHa
        df_cm = pd.DataFrame(
            [
                [f"VP: {mc.verdaderosPositivos:.1f} ha", f"FN: {mc.falsosNegativos:.1f} ha"],
                [f"FP: {mc.falsosPositivos:.1f} ha", f"VN: {mc.verdaderosNegativos:,.1f} ha"],
            ],
            index=["Real: Pérdida/Quema", "Real: Bosque Estable"],
            columns=["Pred: Pérdida/Quema", "Pred: Bosque Estable"],
        )
        st.dataframe(df_cm, width="stretch")

    with col_bm2:
        st.markdown("##### ⚙️ Especificaciones Técnicas del Motor")
        st.markdown(
            f"""
            - **Arquitectura:** U-Net convolucional de cambio multitemporal (entrada de 6 canales)
            - **Runtime:** `ONNX Runtime (CPUExecutionProvider)` — precisión **FP32**
            - **Resolución espacial:** `10 metros / píxel (Sentinel-2 L2A)`
            - **Bandas de entrada:** `6 canales: T0 (B4, B8, B11) + T1 (B4, B8, B11)`
            - **Tamaño de ventana:** `{m.tamanoTilePx} x {m.tamanoTilePx} píxeles` (655.36 ha por parche)
            - **Peso del modelo servido:** `{peso_mb:.1f} MB` (`models/unet_geoforest.onnx`)
            - **Post-procesamiento y Calibración:** `Temperature Scaling (T=0.5) + Umbral Otsu + Filtro Biofísico ΔNDVI/ΔNBR`
            - **Fusión de operadores:** `Conv + BatchNorm + ReLU en el grafo ONNX`
            - **Trazabilidad:** hash SHA-256 del tensor de entrada y sus metadatos como huella de integridad reproducible para auditoría técnica.
            """
        )
