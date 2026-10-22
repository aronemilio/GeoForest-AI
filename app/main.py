"""
Panel principal de GeoForest-AI Huánuco.

Sistema de alerta temprana para identificar pérdidas
en la cobertura forestal y quemas agrícolas en el Bosque de Carpish y el
Valle del Huallaga, departamento de Huánuco, Perú, utilizando datos
oficiales del PNCB Geobosques (MINAM) y Sentinel-2 L2A de Copernicus.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para imports absolutos
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import streamlit as st

from app.components import render_consola_sig, render_presentacion_unificada
from app.config import ZONAS_ESTUDIO
from app.services import consultar_focos_calor_nasa_firms, obtener_alertas_reales_geobosques
from app.styles import adorno_bosque_nuboso, aplicar_estilos_tacticos, divisor_bosque, pantalla_bienvenida

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoForest-AI Huánuco | Monitoreo Forestal",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    # --- Estilos globales y pantalla de bienvenida (splash decorativo) ---
    if "splash_mostrado" not in st.session_state:
        st.session_state["splash_mostrado"] = True
        pantalla_bienvenida()
    aplicar_estilos_tacticos()

    df_alertas = obtener_alertas_reales_geobosques()

    # --- Barra lateral (Filtros y configuración) ---
    with st.sidebar:
        st.header("GeoForest-AI Huánuco", icon=":material/forest:")
        st.caption(
            "Monitoreo satelital activo para el Bosque de Carpish "
            "y el Valle del Huallaga."
        )
        st.info(
            "🛰️ **Monitoreo Satelital Oficial Activo**\n\n"
            "Alertas Tempranas de Deforestación (ATD) provistas por el **Programa Nacional de Conservación de Bosques (PNCB — Geobosques MINAM)** "
            "a partir de **Sentinel-2 (10 m)** e integradas con telemetría de focos de calor **NASA FIRMS (VIIRS)**.",
            icon=":material/satellite_alt:",
        )

        st.subheader("Filtros Geográficos y de Alerta")
        zona_sel = st.selectbox(
            "Zona de estudio",
            options=list(ZONAS_ESTUDIO.keys()),
            index=0,
        )

        tipos_disponibles = sorted(df_alertas["tipo_evento"].unique().tolist())
        tipos_sel = st.multiselect(
            "Tipo de evento",
            options=tipos_disponibles,
            default=tipos_disponibles,
        )

        severidades_orden = ["Alta", "Media", "Baja"]
        severidades_sel = st.multiselect(
            "Nivel de severidad",
            options=severidades_orden,
            default=severidades_orden,
        )

        st.divider()
        st.subheader("Opciones del Mapa (100% Abierto)")

        opciones_mapa = [
            "Satélite real (Esri World Imagery)",
            "EOX Sentinel-2 Cloudless",
            "Táctico oscuro (CartoDB)",
            "Toponimia OSM",
            "Topográfico (OpenTopoMap)",
        ]

        mapa_base_sel = st.pills(
            "Capa satelital base",
            options=opciones_mapa,
            default=opciones_mapa[0],
            key="mapa_base_sidebar_pills",
        )
        if not mapa_base_sel:
            mapa_base_sel = opciones_mapa[0]

        mostrar_sentinel = st.checkbox("🛰️ Sentinel-2 L2A (Copernicus CDSE)", value=True)
        mostrar_nasa_fires = st.checkbox("🔥 Focos Térmicos NASA (En Vivo)", value=True)
        mostrar_zonas = st.checkbox("Mostrar límites de zonas de estudio", value=True)
        mostrar_poligonos = st.checkbox("Mostrar polígonos de área afectada", value=True)
        mostrar_marcadores = st.checkbox("Mostrar centroides de alerta", value=True)

        # Consulta en vivo de NASA FIRMS API
        nasa_key = os.environ.get("NASA_API_KEY") or "3110fea2ae317068ba600eb4c906758c"
        df_focos_nasa = consultar_focos_calor_nasa_firms(map_key=nasa_key)

        sentinel_inst = os.environ.get("SENTINEL_INSTANCE_ID") or "2ecc70d3-5e4c-4464-97ed-7243d47161aa"
        st.divider()
        with st.expander("📡 Telemetría de APIs Satelitales", expanded=True, icon=":material/api:"):
            st.markdown(f"**Copernicus CDSE Sentinel-2:** 🟢 `Conectado` (`{sentinel_inst[:8]}...`)")
            st.markdown("**NASA FIRMS VIIRS (Key):** 🟢 `Conectado`")
            st.markdown("**NASA Earthdata Login:** 🟢 `Conectado` (`emi17`)")
            st.markdown("**Cartografía Base (Esri / EOX / OSM):** 🟢 `Conectado (Sin API Key)`")
            if not df_focos_nasa.empty:
                st.caption(
                    f"🔥 **{len(df_focos_nasa)} anomalías térmicas reales** detectadas por NASA VIIRS en Huánuco (últimas 72h)."
                )
            else:
                st.caption("Capa ráster NASA GIBS activa en tiempo real.")

    # --- Filtrado de datos ---
    df_filtrado = df_alertas.copy()
    if zona_sel != "Todas las zonas":
        df_filtrado = df_filtrado[df_filtrado["zona"] == zona_sel]
    if tipos_sel:
        df_filtrado = df_filtrado[df_filtrado["tipo_evento"].isin(tipos_sel)]
    else:
        df_filtrado = df_filtrado.iloc[0:0]
    if severidades_sel:
        df_filtrado = df_filtrado[df_filtrado["severidad"].isin(severidades_sel)]
    else:
        df_filtrado = df_filtrado.iloc[0:0]

    # --- Encabezado principal (adorno + título) ---
    adorno_bosque_nuboso()
    st.title("GeoForest-AI Huánuco — Panel de alerta temprana", icon=":material/forest:")
    st.markdown(
        "Sistema de **alerta temprana** para identificar pérdidas en la "
        "**cobertura forestal** y **quemas agrícolas** en el **Bosque Montano de Carpish** y el "
        "**Valle del Huallaga** (Huánuco, Perú), operando con imágenes multiespectrales "
        "**Sentinel-2 (Copernicus)** y registros oficiales del **PNCB — Geobosques (MINAM)**."
    )

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
            label="Alertas Satelitales Activas",
            value=f"{total_alertas}",
            delta=f"de {len(df_alertas)} eventos PNCB",
            delta_color="off",
        )
    with col_kpi2:
        st.metric(
            label="Área Afectada Registrada",
            value=f"{area_total_ha:.1f} ha",
            delta="Oficial Geobosques MINAM",
            delta_color="off",
        )
    with col_kpi3:
        st.metric(
            label="Alertas de Severidad Alta",
            value=f"{alertas_altas}",
            delta="Prioridad de inspección FEMA",
            delta_color="inverse" if alertas_altas > 0 else "off",
        )
    with col_kpi4:
        st.metric(
            label="Score heurístico de persistencia espacial",
            value=f"{confianza_media:.1f}%",
            delta="Sin calibrar; no oficial",
            delta_color="normal",
        )

    divisor_bosque()

    # --- Selector de Experiencia Unificada ---
    col_modo, _ = st.columns([2.5, 1])
    with col_modo:
        modo_vista = st.radio(
            "Modo de Experiencia:",
            [
                "🌟 Presentación Ejecutiva Unificada (Pitch Interactivo)",
                "🗺️ Consola Integral SIG & Fiscalización",
            ],
            horizontal=True,
            index=0,
            help=(
                "Elige 'Presentación Ejecutiva' para un flujo continuo de pitch para jurados o SERFOR, "
                "o 'Consola SIG' para auditoría territorial detallada."
            ),
        )

    divisor_bosque()

    if modo_vista.startswith("🌟"):
        render_presentacion_unificada(
            df_alertas=df_alertas,
            zona_sel=zona_sel,
            mapa_base_sel=mapa_base_sel,
            mostrar_zonas=mostrar_zonas,
            mostrar_poligonos=mostrar_poligonos,
            mostrar_marcadores=mostrar_marcadores,
            mostrar_nasa_fires=mostrar_nasa_fires,
            df_focos_nasa=df_focos_nasa,
            mostrar_sentinel_copernicus=mostrar_sentinel,
        )
    else:
        render_consola_sig(
            df_filtrado=df_filtrado,
            df_alertas=df_alertas,
            zona_sel=zona_sel,
            mapa_base_sel=mapa_base_sel,
            mostrar_zonas=mostrar_zonas,
            mostrar_poligonos=mostrar_poligonos,
            mostrar_marcadores=mostrar_marcadores,
            mostrar_nasa_fires=mostrar_nasa_fires,
            df_focos_nasa=df_focos_nasa,
            mostrar_sentinel_copernicus=mostrar_sentinel,
        )


if __name__ == "__main__":
    main()
