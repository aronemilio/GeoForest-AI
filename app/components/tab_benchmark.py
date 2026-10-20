"""Componente para la pestaña de Benchmark U-Net ONNX."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.onnx_demo import peso_modelo_onnx_mb
from app.styles.theme import divisor_bosque
from src.sim import METRICAS_MODELO_SIMULADAS


def render_tab_benchmark() -> None:
    """Maqueta del reporte de desempeño del motor U-Net ONNX.

    Las cifras de precisión que se muestran son **valores de referencia
    simulados** (``METRICAS_MODELO_SIMULADAS``), no resultados de una evaluación
    medida. El modelo servido en ``models/`` es de referencia y no está
    entrenado, y es FP32 (no cuantizado a Int8).
    """
    st.subheader(
        "Benchmark del Motor U-Net ONNX — Métricas de Referencia (Simuladas)",
        icon=":material/speed:",
    )
    st.caption(
        "Plantilla del reporte de precisión, latencia y consumo del motor de inferencia. "
        "Los números provienen de una referencia predefinida, no de una evaluación "
        "medida sobre el Bosque de Carpish ni el Valle del Huallaga."
    )
    st.warning(
        "🚧 **Estas métricas NO están medidas.** Los valores de IoU, F1, latencia y la "
        "matriz de confusión son **cifras de referencia simuladas**, usadas para "
        "maquetar el reporte. El modelo ONNX incluido en `models/` es un **modelo de "
        "referencia SIN ENTRENAR** (su salida es prácticamente constante) y está en "
        "**FP32: no hay cuantización PTQ Int8**. Deben sustituirse por métricas reales "
        "cuando exista un entrenamiento validado con etiquetas de Geobosques.",
        icon=":material/warning:",
    )

    m = METRICAS_MODELO_SIMULADAS
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
        "Tile 256x256 px (< 20 ms)",
        delta_color="normal",
    )
    bm_c4.metric(
        "Peso real del .onnx servido",
        f"{peso_mb:.1f} MB" if peso_mb else "no encontrado",
        "FP32 · sin cuantizar (no hay PTQ Int8)",
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
            - **Arquitectura:** U-Net convolucional de referencia (entrada de 6 canales multitemporal)
            - **Runtime:** `ONNX Runtime (CPU)` — modelo **FP32, sin cuantizar**
            - **Resolución espacial:** `10 metros / píxel (Sentinel-2)`
            - **Bandas de entrada:** `6 canales: T0 (B4, B8, B11) + T1 (B4, B8, B11)`
            - **Tamaño de ventana:** `{m.tamanoTilePx} x {m.tamanoTilePx} píxeles` (referencia)
            - **Peso real del modelo servido:** `{peso_mb:.1f} MB` (FP32)
            - **Estado del modelo:** `de referencia, SIN ENTRENAR` → la salida no es una detección
            - **Fusión de operadores:** `Conv + BatchNorm + ReLU en el grafo ONNX`
            - **Trazabilidad:** hash SHA-256 del tensor y sus metadatos como **huella de integridad** reproducible; por sí sola no constituye una cadena de custodia acreditada.
            """
        )
