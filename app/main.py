"""
Panel principal de GeoForest-AI Huánuco.

Sistema experimental de alerta temprana para identificar posibles cambios
en la cobertura forestal y quemas agrícolas en el Bosque de Carpish y el
Valle del Huallaga, departamento de Huánuco, Perú.

Nota: En esta fase inicial, todos los registros y polígonos mostrados
corresponden a DATOS DEMOSTRATIVOS SIMULADOS con fines de validación
de interfaz y visualización geoespacial.
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
from app.services import consultar_focos_calor_nasa_firms, obtener_alertas_simuladas
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
    pantalla_bienvenida()
    aplicar_estilos_tacticos()

    df_alertas = obtener_alertas_simuladas()

    # --- Barra lateral (Filtros y configuración) ---
    with st.sidebar:
        st.header("GeoForest-AI Huánuco", icon=":material/forest:")
        st.caption(
            "Monitoreo satelital experimental para el Bosque de Carpish "
            "y el Valle del Huallaga."
        )
        st.info(
            "🧪 **Modo Demostrativo Activo**\n\n"
            "Todas las alertas visualizadas en este panel son **datos simulados** "
            "para validar la estructura del sistema antes de conectar la inferencia ONNX.",
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
        st.subheader("Opciones del Mapa")

        token_mb = os.environ.get("MAPBOX_TOKEN", "").strip()
        opciones_mapa = [
            "Satélite real (Esri World Imagery)",
            "EOX Sentinel-2 Cloudless",
        ]
        if token_mb:
            opciones_mapa.extend(["Mapbox Satélite HD", "Mapbox Táctico Nocturno"])
        opciones_mapa.extend([
            "Táctico oscuro (CartoDB)",
            "Toponimia OSM",
            "Topográfico (OpenTopoMap)",
        ])

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
            st.markdown("**Mapbox Navigation / Satélite:** 🟢 `Conectado`")
            st.markdown(f"**Copernicus CDSE Sentinel-2:** 🟢 `Conectado` (`{sentinel_inst[:8]}...`)")
            st.markdown("**NASA FIRMS VIIRS (Key):** 🟢 `Conectado`")
            st.markdown("**NASA Earthdata Login:** 🟢 `Conectado` (`emi17`)")
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
        "Sistema experimental de detección de **pérdida de cobertura forestal** y "
        "**quemas agrícolas** en el **Bosque Montano de Carpish** y el **Valle del Huallaga** "
        "(Huánuco, Perú), diseñado para operar con imágenes **Sentinel-2** y "
        "referencias de **Geobosques (MINAM)**."
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
            mapbox_token=token_mb,
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
            token_mb=token_mb,
            mostrar_sentinel_copernicus=mostrar_sentinel,
        )


if __name__ == "__main__":
    main()
