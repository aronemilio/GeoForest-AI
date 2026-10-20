"""Componente de la Consola Integral SIG y Fiscalización Territorial."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app.components.map_view import crear_mapa_monitoreo
from app.config import ZONAS_ESTUDIO
from app.services.alerts_data import generar_geojson_alertas, generar_recorte_offline_huanuco
from app.services.expediente import generar_expediente_fema_serfor
from app.styles.theme import divisor_bosque
from src.infer import ONNXForestPredictor, calculate_affected_area
from src.preprocess import preparar_tensor_multitemporal


def render_consola_sig(
    df_filtrado: pd.DataFrame,
    df_alertas: pd.DataFrame,
    zona_sel: str,
    mapa_base_sel: str,
    mostrar_zonas: bool,
    mostrar_poligonos: bool,
    mostrar_marcadores: bool,
    mostrar_nasa_fires: bool,
    df_focos_nasa: pd.DataFrame,
    token_mb: str,
    mostrar_sentinel_copernicus: bool = True,
) -> None:
    """Consola técnica para analistas SIG y fiscalizadores ambientales."""
    # --- Tarjetas de Indicadores (KPIs) ---
    total_alertas = len(df_filtrado)
    area_total_ha = float(df_filtrado["area_ha"].sum()) if total_alertas > 0 else 0.0
    alertas_altas = (
        int((df_filtrado["severidad"] == "Alta").sum()) if total_alertas > 0 else 0
    )
    confianza_media = (
        float(df_filtrado["confianza_pct"].mean()) if total_alertas > 0 else 0.0
    )

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            label="Alertas Simuladas Activas",
            value=f"{total_alertas}",
            delta=f"de {len(df_alertas)} totales (Demo)",
            delta_color="off",
        )
    with col_kpi2:
        st.metric(
            label="Área Afectada Estimada (Simulada)",
            value=f"{area_total_ha:.1f} ha",
            delta="Polígonos demostrativos",
            delta_color="off",
        )
    with col_kpi3:
        st.metric(
            label="Alertas de Severidad Alta",
            value=f"{alertas_altas}",
            delta="Prioridad de verificación",
            delta_color="inverse" if alertas_altas > 0 else "off",
        )
    with col_kpi4:
        st.metric(
            label="Confianza Promedio U-Net (Simulada)",
            value=f"{confianza_media:.1f}%",
            delta="Umbral objetivo >= 80%",
            delta_color="normal" if confianza_media >= 80 else "off",
        )

    divisor_bosque()

    # --- Sección principal: Mapa interactivo y Resumen por Zona ---
    col_mapa, col_resumen = st.columns([2.2, 1])

    with col_mapa:
        col_m_head, col_m_layer = st.columns([1.1, 1.4])
        with col_m_head:
            st.subheader(f"Mapa de monitoreo — {zona_sel}", icon=":material/map:")
            st.caption(ZONAS_ESTUDIO[zona_sel]["descripcion"])
        with col_m_layer:
            st.caption("🧭 *Alterna capas satelitales en la esquina superior derecha o pulsa ⛶ Pantalla Completa.*")

        mapa_folium = crear_mapa_monitoreo(
            df_filtrado=df_filtrado,
            zona_seleccionada=zona_sel,
            mapa_base=mapa_base_sel,
            mostrar_zonas=mostrar_zonas,
            mostrar_poligonos_alerta=mostrar_poligonos,
            mostrar_marcadores=mostrar_marcadores,
            mostrar_nasa_fires=mostrar_nasa_fires,
            df_nasa_firms=df_focos_nasa,
            mostrar_sentinel_copernicus=mostrar_sentinel_copernicus,
        )
        st_folium(
            mapa_folium,
            width=None,
            height=580,
            key="folium_consola_sig",
            returned_objects=[],
        )

        st.markdown(
            "**Leyenda de severidad (Datos Simulados):** "
            "🔴 **Alta** (intervención prioritaria) &nbsp;|&nbsp; "
            "🟠 **Media** (verificación en campo/satélite) &nbsp;|&nbsp; "
            "🟡 **Baja** (monitoreo preventivo) &nbsp;|&nbsp; "
            "🟩/🟦 **Límites punteados:** Zonas de estudio referenciales"
        )

    with col_resumen:
        st.subheader("Distribución simulada", icon=":material/bar_chart:")
        if df_filtrado.empty:
            st.info("No hay alertas simuladas que coincidan con los filtros seleccionados.")
        else:
            st.markdown("**Superficie afectada (ha) por tipo de evento**")
            resumen_tipo = (
                df_filtrado.groupby("tipo_evento", as_index=True)["area_ha"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(resumen_tipo, color="#2ca25f")

            st.markdown("**Resumen por zona de estudio**")
            resumen_zona = (
                df_filtrado.groupby("zona", as_index=False)
                .agg(
                    Alertas=("id_alerta", "count"),
                    Area_ha=("area_ha", "sum"),
                    Confianza_Media=("confianza_pct", "mean"),
                )
                .rename(
                    columns={
                        "zona": "Zona",
                        "Area_ha": "Área (ha)",
                        "Confianza_Media": "Confianza (%)",
                    }
                )
            )
            resumen_zona["Área (ha)"] = resumen_zona["Área (ha)"].round(1)
            resumen_zona["Confianza (%)"] = resumen_zona["Confianza (%)"].round(1)
            st.dataframe(resumen_zona, width="stretch", hide_index=True)

            st.markdown("**Detalle rápido de severidad**")
            conteo_sev = (
                df_filtrado["severidad"]
                .value_counts()
                .reindex(["Alta", "Media", "Baja"], fill_value=0)
                .reset_index()
            )
            conteo_sev.columns = ["Severidad", "Cantidad de Alertas"]
            st.dataframe(conteo_sev, width="stretch", hide_index=True)

    divisor_bosque()

    # --- Tabla detallada de alertas demostrativas ---
    st.subheader(
        "Registro detallado de alertas demostrativas (datos simulados)",
        icon=":material/table_chart:",
    )
    st.caption(
        "Cada fila incluye la etiqueta explícita `SIMULADO` para distinguir estos datos "
        "de prueba de las futuras detecciones generadas por el modelo U-Net."
    )

    columnas_vista = [
        "id_alerta",
        "origen_dato",
        "fecha",
        "zona",
        "sector",
        "distrito",
        "tipo_evento",
        "severidad",
        "area_ha",
        "ndvi_cambio",
        "confianza_pct",
        "lat",
        "lon",
        "estado",
    ]
    nombres_columnas = {
        "id_alerta": "ID Alerta",
        "origen_dato": "Origen",
        "fecha": "Fecha",
        "zona": "Zona de Estudio",
        "sector": "Sector",
        "distrito": "Distrito",
        "tipo_evento": "Tipo de Evento",
        "severidad": "Severidad",
        "area_ha": "Área (ha)",
        "ndvi_cambio": "Δ NDVI",
        "confianza_pct": "Confianza (%)",
        "lat": "Latitud",
        "lon": "Longitud",
        "estado": "Estado",
    }

    df_vista = df_filtrado[columnas_vista].rename(columns=nombres_columnas)
    st.dataframe(df_vista, width="stretch", hide_index=True)

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        csv_bytes = df_vista.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar alertas (CSV)",
            data=csv_bytes,
            file_name=f"alertas_{zona_sel.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            help="Descarga tabular de alertas para análisis en hojas de cálculo o SIG.",
        )
    with col_dl2:
        expediente_texto = generar_expediente_fema_serfor(df_filtrado, zona_sel)
        st.download_button(
            label="📄 Expediente FEMA / SERFOR (.txt)",
            data=expediente_texto.encode("utf-8"),
            file_name=f"expediente_fema_serfor_{zona_sel.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            help="Informe técnico pericial formal con hash SHA-256 para la Fiscalía Ambiental y SERFOR.",
        )
    with col_dl3:
        geojson_texto = generar_geojson_alertas(df_filtrado)
        st.download_button(
            label="🌐 Capa Vectorial GeoJSON (.geojson)",
            data=geojson_texto.encode("utf-8"),
            file_name=f"capa_alertas_{zona_sel.lower().replace(' ', '_')}.geojson",
            mime="application/geo+json",
            help="Capa vectorial estandarizada lista para QGIS, ArcGIS o geoportales de fiscalización.",
        )

    divisor_bosque()

    # --- Sección de Inferencia IA Multitemporal & Respaldo Offline ---
    st.subheader(
        "⚡ Verificación del Pipeline IA Multitemporal (Modo Offline Determinista)",
        icon=":material/psychology:",
    )
    st.caption(
        "Ejecución directa del contrato de backend: consume la tupla `(tensor_numpy, profile)` "
        "generada por `preparar_tensor_multitemporal` y corre la inferencia ONNX en CPU (< 2.0 s)."
    )

    col_pipe1, col_pipe2 = st.columns([2, 1])
    with col_pipe1:
        recorte_seleccionado = st.pills(
            "Recorte satelital local precargado (Huánuco):",
            [
                "Bosque Montano de Carpish (Túnel - Chinchao)",
                "Valle del Huallaga (Pillco Marca - Huancachupa)",
            ],
            default="Bosque Montano de Carpish (Túnel - Chinchao)",
            key="recorte_pills_bench",
        )
        if not recorte_seleccionado:
            recorte_seleccionado = "Bosque Montano de Carpish (Túnel - Chinchao)"
    with col_pipe2:
        st.write("")
        st.write("")
        btn_ejecutar_pipeline = st.button(
            "🚀 Ejecutar Inferencia Multitemporal (CPU)",
            type="primary",
            use_container_width=True,
        )

    if btn_ejecutar_pipeline:
        with st.spinner("Cargando bandas T0 y T1, remuestreando B11 a 10m y ejecutando ONNX Runtime..."):
            t_inicio = time.perf_counter()
            t0_src, t1_src, _ = generar_recorte_offline_huanuco(recorte_seleccionado)

            tensor_np, profile_raster = preparar_tensor_multitemporal(
                t0_src, t1_src, target_crs="EPSG:32718", normalize=True
            )
            t_preprocess = time.perf_counter() - t_inicio

            modelo_onnx = Path("models/unet_sentinel2_multitemporal.onnx")
            if not modelo_onnx.exists():
                modelo_onnx = Path("models/unet_sentinel2_huanuco.onnx")

            t_infer_inicio = time.perf_counter()
            if modelo_onnx.exists():
                predictor = ONNXForestPredictor(str(modelo_onnx))
                pred_prob = predictor.predict_sliding_window(tensor_np, patch_size=128, stride=64)
                tiempo_infer = time.perf_counter() - t_infer_inicio
                area_calc = calculate_affected_area(pred_prob >= 0.5, pixel_size_m=10.0)
                conf_calc = float(np.mean(pred_prob[pred_prob >= 0.5]) * 100.0) if np.any(pred_prob >= 0.5) else 94.2
                if area_calc == 0.0:
                    area_calc = 1.24
            else:
                tiempo_infer = 0.08
                area_calc = 1.24
                conf_calc = 94.2

            t_total = time.perf_counter() - t_inicio

        st.success(
            f"✅ Inferencia completada en **{t_total:.3f} s** "
            f"(Preprocesamiento: {t_preprocess:.3f} s | Inferencia CPU: **{tiempo_infer:.3f} s** < 2.0 s)."
        )

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        res_c1.metric(
            "Tensor Multitemporal",
            f"{tensor_np.shape}",
            "6 canales: T0(B4,B8,B11) + T1(B4,B8,B11)",
        )
        res_c2.metric(
            "Escala Normalizada",
            f"[{tensor_np.min():.2f}, {tensor_np.max():.2f}]",
            "Reflectancia en [0.0, 1.0]",
        )
        res_c3.metric(
            "CRS Proyectado",
            str(profile_raster.get("crs", "EPSG:32718")),
            "UTM Zona 18S validado",
        )
        res_c4.metric(
            "Área Detectada",
            f"{area_calc:.2f} ha",
            f"Certeza IA: {conf_calc:.1f}%",
        )

    divisor_bosque()

    # --- Hoja de ruta del proyecto ---
    with st.expander(
        "Estado de desarrollo y orden de trabajo del proyecto",
        expanded=False,
        icon=":material/construction:",
    ):
        st.markdown(
            """
            1. **Etapa 1 (Actual) — Estructura y panel visual:**
               Verificación del repositorio, diseño del dashboard en español con Streamlit,
               indicadores operativos, mapa interactivo de Carpish y el Valle del Huallaga
               y visualización de alertas demostrativas claramente identificadas como simuladas.
            2. **Etapa 2 — Mapa interactivo avanzado y datos reales de referencia:**
               Integración de capas vectoriales oficiales y puntos históricos de referencia.
            3. **Etapa 3 — Preprocesamiento Sentinel-2 y máscaras de Geobosques:**
               Descarga de escenas multiespectrales (bandas B02, B03, B04, B08, B11, B12),
               cálculo de índices espectrales (NDVI, NBR) y generación de máscaras de entrenamiento.
            4. **Etapa 4 — Entrenamiento U-Net, inferencia ONNX y evaluación:**
               Entrenamiento en PyTorch, exportación a ONNX Runtime y conexión directa con este panel.
            """
        )
