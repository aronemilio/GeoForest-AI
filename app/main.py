"""
Panel principal de GeoForest-AI Huánuco.

Sistema de alerta temprana para identificar pérdidas
en la cobertura forestal y quemas agrícolas en el Bosque de Carpish y el
Valle del Huallaga, departamento de Huánuco, Perú, utilizando datos
oficiales del PNCB Geobosques (MINAM) y Sentinel-2 L2A de Copernicus.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import sys
import time
from pathlib import Path

# Garantizar que la raíz del proyecto esté en sys.path para imports absolutos
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))


def _cargar_env_local() -> None:
    """Carga variables desde .env si existe sin sobreescribir variables del sistema."""
    env_file = _ROOT_DIR / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


_cargar_env_local()

import streamlit as st

from app.components import render_consola_sig, render_presentacion_unificada
from app.config import ZONAS_ESTUDIO
from app.services import (
    TZ_PERU,
    limpiar_cache_alertas_tiempo_real,
    limpiar_cache_focos_tiempo_real,
    obtener_alertas_deforestacion_tiempo_real,
    obtener_overlays_sentinel2_tiempo_real,
    obtener_quemas_tiempo_real_integradas,
)
from app.styles import adorno_bosque_nuboso, aplicar_estilos_tacticos, divisor_bosque, pantalla_bienvenida

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoForest-AI Huánuco | Monitoreo Forestal en Tiempo Real",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIPOS_EVENTO_MONITOREO: list[str] = [
    "Deforestación Antrópica",
    "Pérdida de Cobertura + Quema Activa Cercana",
    "Pérdida Natural de Cobertura",
]


def _forzar_recarga_feeds_tiempo_real() -> None:
    """Invalida cachés de quemas NRT y alertas ATD y marca el instante de sincronización."""
    limpiar_cache_focos_tiempo_real()
    limpiar_cache_alertas_tiempo_real()
    st.session_state["last_live_sync_epoch"] = 0.0


def main() -> None:
    # --- Estilos globales y pantalla de bienvenida (splash decorativo) ---
    if "splash_mostrado" not in st.session_state:
        st.session_state["splash_mostrado"] = True
        pantalla_bienvenida()
    aplicar_estilos_tacticos()

    nasa_key = os.environ.get("NASA_API_KEY", "")
    sentinel_inst = os.environ.get("SENTINEL_INSTANCE_ID", "")

    # --- Barra lateral (Sincronización en Tiempo Real, Filtros y Capas) ---
    with st.sidebar:
        st.header("GeoForest-AI Huánuco", icon=":material/forest:")
        st.caption(
            "Monitoreo satelital en tiempo real para el Bosque de Carpish "
            "y el Valle del Huallaga."
        )
        st.info(
            "🛰️ **Feeds Satelitales Oficiales NRT Activos**\n\n"
            "• **Pérdida de cobertura forestal:** API & WMS 2026 **PNCB Geobosques (MINAM)** + cálculo físico **ΔNDVI Sentinel-2 L2A (10 m)**.\n"
            "• **Quemas e incendios activos:** **NASA FIRMS** (VIIRS SNPP/NOAA-20/21, MODIS, Landsat NRT) + **SERFOR UFMS (SAMI)** + **ΔNBR Sentinel-2**.\n\n"
            "*Nota de dominio: Un cambio de cobertura o foco térmico requiere verificación en campo; no implica tala ilegal certificada.*",
            icon=":material/satellite_alt:",
        )

        st.subheader("⚡ Control en Tiempo Real")
        auto_refresco = st.toggle(
            "Auto-refresco periódico en vivo",
            value=True,
            key="sidebar_auto_refresco",
            help="Actualiza periódicamente las alertas de pérdida de cobertura y los focos de quema sin recargar toda la página.",
        )
        opciones_intervalo = {
            "Cada 1 min": 60,
            "Cada 5 min (Recomendado)": 300,
            "Cada 15 min": 900,
        }
        intervalo_lbl = st.selectbox(
            "Frecuencia de actualización",
            options=list(opciones_intervalo.keys()),
            index=1,
            disabled=not auto_refresco,
            key="sidebar_intervalo_refresco",
        )
        intervalo_seg: int | None = opciones_intervalo[intervalo_lbl] if auto_refresco else None

        dias_quemas = st.slider(
            "Ventana de quemas / focos NRT (días)",
            min_value=1,
            max_value=5,
            value=5,
            key="sidebar_dias_quemas",
            help="Ventana temporal NRT oficial (1 a 5 días en NASA FIRMS y SERFOR SAMI).",
        )

        if st.button(
            "🔄 Actualizar ahora",
            type="primary",
            width="stretch",
            key="sidebar_btn_refresh_now",
            help="Consulta inmediatamente NASA FIRMS, SERFOR SAMI y PNCB Geobosques.",
        ):
            _forzar_recarga_feeds_tiempo_real()

        if st.button(
            "📥 Sincronizar última pasada Sentinel-2 (18LUQ)",
            width="stretch",
            key="sidebar_btn_sync_s2",
            help="Consulta STAC Earth Search / Copernicus y descarga el último par limpio B04/B08/B11 (10 m) para Carpish y Huallaga.",
        ):
            with st.spinner("Descargando último recorte Sentinel-2 L2A (Tile 18LUQ) para Carpish y Huallaga..."):
                _, telem_s2_sync = obtener_overlays_sentinel2_tiempo_real(
                    zona_seleccionada="Todas las zonas",
                    sincronizar_red=True,
                    forzar_descarga=True,
                )
                st.session_state["s2_sync_feedback"] = telem_s2_sync

        if "s2_sync_feedback" in st.session_state:
            fb = st.session_state["s2_sync_feedback"]
            if fb.get("errores"):
                st.warning(f"Sincronización S2 parcial: {'; '.join(fb['errores'])}")
            else:
                st.success(
                    f"Sentinel-2 18LUQ actualizado: {', '.join(fb.get('fechas_pares', []))}"
                )

        st.divider()
        st.subheader("Filtros Geográficos y de Alerta")
        zona_sel = st.selectbox(
            "Zona de estudio",
            options=list(ZONAS_ESTUDIO.keys()),
            index=0,
        )

        tipos_sel = st.multiselect(
            "Tipo de evento de cobertura",
            options=TIPOS_EVENTO_MONITOREO,
            default=TIPOS_EVENTO_MONITOREO,
        )

        severidades_orden = ["Alta", "Media", "Baja"]
        severidades_sel = st.multiselect(
            "Nivel de severidad ATD",
            options=severidades_orden,
            default=severidades_orden,
        )

        st.divider()
        st.subheader("Capas del Mapa en Tiempo Real")

        opciones_mapa = [
            "Satélite real (Esri World Imagery)",
            "Toponimia OSM (OpenStreetMap)",
        ]

        mapa_base_sel = st.pills(
            "Capa base del mapa",
            options=opciones_mapa,
            default=opciones_mapa[0],
            key="mapa_base_sidebar_pills",
        )
        if not mapa_base_sel:
            mapa_base_sel = opciones_mapa[0]

        mostrar_sentinel_copernicus = st.checkbox(
            "🛰️ Cambio Espectral Sentinel-2 L2A Real (ΔNDVI / ΔNBR)",
            value=True,
            help="Proyecta los recortes GeoTIFF Sentinel-2 L2A reales (10 m) con pérdida de cobertura (rojo ΔNDVI) y cicatriz de quema (naranja ΔNBR).",
        )
        mostrar_wms_geobosques = st.checkbox(
            "🌲 Ráster WMS PNCB Geobosques 2026 (En Vivo)",
            value=True,
            help="Superpone el servicio WMS oficial de Alertas Tempranas 2026 del MINAM.",
        )
        mostrar_nasa_fires = st.checkbox(
            "🔥 Quemas y Focos Térmicos FIRMS + SERFOR (En Vivo)",
            value=True,
            help="Muestra focos activos NRT de NASA FIRMS (VIIRS/MODIS/Landsat), alertas de incendio SERFOR SAMI y capa NASA GIBS.",
        )
        mostrar_zonas = st.checkbox("Mostrar límites de zonas de estudio", value=True)
        mostrar_poligonos = st.checkbox("Mostrar polígonos de pérdida de cobertura (ATD)", value=True)
        mostrar_marcadores = st.checkbox("Mostrar centroides de alerta ATD", value=True)

        st.divider()
        with st.expander("📡 Telemetría de APIs Satelitales", expanded=True, icon=":material/api:"):
            if sentinel_inst:
                st.markdown(f"**Copernicus / STAC Sentinel-2 (18LUQ):** 🟢 `Activo` (`{sentinel_inst[:8]}...`)")
            else:
                st.markdown("**Copernicus STAC Sentinel-2 (18LUQ):** 🟢 `Recortes L2A Reales Activos`")
            st.markdown("**PNCB Geobosques MINAM (ATD 2026):** 🟢 `API REST + WMS En Vivo`")
            if nasa_key:
                st.markdown("**NASA FIRMS Multi-Sensor NRT:** 🟢 `VIIRS + MODIS + Landsat`")
            else:
                st.markdown("**NASA FIRMS VIIRS:** 🟡 `Capa WMS Pública GIBS`")
            st.markdown("**SERFOR Perú UFMS (SAMI v2.1):** 🟢 `Incendios Forestales En Vivo`")
            st.markdown("**Cartografía Base (Esri / OSM):** 🟢 `Conectado`")

    # --- Encabezado principal (adorno + título) ---
    adorno_bosque_nuboso()
    st.title("GeoForest-AI Huánuco — Monitoreo en Tiempo Real", icon=":material/forest:")
    st.markdown(
        "Sistema de **alerta temprana en tiempo real** para identificar **pérdidas de cobertura forestal** "
        "(ATD oficiales **PNCB Geobosques MINAM** y anomalía física **ΔNDVI Sentinel-2 L2A**) y **quemas / incendios activos** "
        "(**NASA FIRMS NRT**, **SERFOR SAMI** y cicatrices **ΔNBR Sentinel-2**) en el **Bosque Montano de Carpish** "
        "y el **Valle del Huallaga** (Huánuco, Perú)."
    )

    run_every_td = timedelta(seconds=intervalo_seg) if intervalo_seg else None

    @st.fragment(run_every=run_every_td)
    def _panel_monitoreo_en_vivo() -> None:
        ahora_epoch = time.time()
        ultimo_epoch = float(st.session_state.get("last_live_sync_epoch", 0.0))
        umbral_expiracion = float(intervalo_seg - 2) if intervalo_seg else 300.0

        if ultimo_epoch <= 0.0 or (intervalo_seg and (ahora_epoch - ultimo_epoch) >= umbral_expiracion):
            limpiar_cache_focos_tiempo_real()
            limpiar_cache_alertas_tiempo_real()
            st.session_state["last_live_sync_epoch"] = ahora_epoch
            st.session_state["last_live_sync_pet"] = datetime.now(TZ_PERU).strftime(
                "%H:%M:%S PET (%Y-%m-%d)"
            )

        df_focos_vivo, telem_quemas = obtener_quemas_tiempo_real_integradas(
            map_key=nasa_key,
            dias=dias_quemas,
            incluir_serfor=True,
        )
        df_alertas, telem_alertas = obtener_alertas_deforestacion_tiempo_real(
            df_quemas_vivo=df_focos_vivo,
            usar_api_vivo=True,
        )

        if mostrar_sentinel_copernicus:
            overlays_s2, telem_s2 = obtener_overlays_sentinel2_tiempo_real(
                zona_seleccionada=zona_sel,
                sincronizar_red=False,
            )
        else:
            overlays_s2 = []
            telem_s2 = {
                "overlays_activos": 0,
                "area_perdida_s2_ha": 0.0,
                "area_quema_s2_ha": 0.0,
                "fechas_pares": [],
                "errores": [],
            }

        sync_pet = st.session_state.get(
            "last_live_sync_pet", telem_quemas.get("timestamp_pet", "En vivo")
        )

        # Filtrado geográfico y de severidad/tipo para alertas ATD
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

        if zona_sel != "Todas las zonas" and not df_focos_vivo.empty:
            df_focos_kpi = df_focos_vivo[df_focos_vivo["zona"] == zona_sel]
        else:
            df_focos_kpi = df_focos_vivo

        # Barra de estado de sincronización en tiempo real + botón rápido
        col_status, col_btn_inline = st.columns([4.4, 1.0])
        with col_status:
            estado_api_minam = (
                f"🟢 API MINAM 2026 ({telem_alertas['alertas_api_vivo_count']} clústeres)"
                if telem_alertas.get("api_geobosques_conectada")
                else "🟡 Respaldo Geobosques Local"
            )
            pares_txt = (
                " · ".join(telem_s2["fechas_pares"])
                if telem_s2.get("fechas_pares")
                else "capa S2 desactivada"
            )
            modo_auto_txt = intervalo_lbl if auto_refresco else "Manual"
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.09); border: 1px solid rgba(16, 185, 129, 0.38);
                            border-radius: 8px; padding: 8px 13px; font-size: 12.5px; margin-bottom: 10px;">
                  <span style="color:#10b981; font-weight:700;">● EN VIVO</span>
                  &nbsp;|&nbsp; <b>Última sincronización:</b> <code>{sync_pet}</code>
                  &nbsp;|&nbsp; <b>Refresco:</b> {modo_auto_txt}
                  &nbsp;|&nbsp; {estado_api_minam}
                  &nbsp;|&nbsp; 🔥 <b>FIRMS:</b> {telem_quemas['focos_nasa_count']} · <b>SERFOR:</b> {telem_quemas['alertas_serfor_count']}
                  &nbsp;|&nbsp; 🛰️ <b>S2 18LUQ:</b> <code>{pares_txt}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_btn_inline:
            if st.button(
                "🔄 Actualizar ahora",
                key="inline_btn_refresh_now",
                width="stretch",
                help="Recargar feeds satelitales en vivo inmediatamente.",
            ):
                _forzar_recarga_feeds_tiempo_real()
                st.rerun(scope="fragment")

        # --- Tarjetas de Indicadores en Tiempo Real (KPIs) ---
        total_alertas = len(df_filtrado)
        area_total_ha = float(df_filtrado["area_ha"].sum()) if total_alertas > 0 else 0.0
        con_quema_cercana = (
            int(
                (
                    df_filtrado["tipo_evento"]
                    == "Pérdida de Cobertura + Quema Activa Cercana"
                ).sum()
            )
            if total_alertas > 0
            else 0
        )
        quemas_zona_count = int(len(df_focos_kpi))
        quemas_reg_count = int(len(df_focos_vivo))
        perdida_s2_ha = float(telem_s2.get("area_perdida_s2_ha", 0.0))
        quema_s2_ha = float(telem_s2.get("area_quema_s2_ha", 0.0))

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric(
                label="🌲 Pérdida de Cobertura Forestal (ATD)",
                value=f"{total_alertas}",
                delta=(
                    f"Última ATD: {telem_alertas['ultima_fecha_atd']}"
                    if total_alertas > 0
                    else "Sin alertas con filtros actuales"
                ),
                delta_color="off",
                help="Clústeres oficiales de pérdida de bosque detectados por PNCB Geobosques (MINAM) sobre Sentinel-2 (10 m).",
            )
        with col_kpi2:
            st.metric(
                label="📐 Área ATD Registrada (Geobosques)",
                value=f"{area_total_ha:.1f} ha",
                delta=(
                    f"{con_quema_cercana} con foco de quema < 1.5 km"
                    if con_quema_cercana > 0
                    else "Oficial PNCB Geobosques MINAM"
                ),
                delta_color="inverse" if con_quema_cercana > 0 else "off",
                help="Superficie acumulada de pérdida de cobertura forestal. No implica certificación de tala ilegal sin acta de campo.",
            )
        with col_kpi3:
            delta_fuego = (
                f"{quemas_reg_count} en región Huánuco ({dias_quemas}d NRT)"
                if quemas_reg_count > 0
                else f"Sin focos activos ({dias_quemas}d NRT)"
            )
            st.metric(
                label="🔥 Quemas y Focos Térmicos Activos",
                value=f"{quemas_zona_count}",
                delta=delta_fuego,
                delta_color="inverse" if quemas_zona_count > 0 else "off",
                help="Focos térmicos integrados en tiempo real desde NASA FIRMS (VIIRS/MODIS/Landsat NRT) y SERFOR UFMS (SAMI).",
            )
        with col_kpi4:
            st.metric(
                label="🛰️ Cambio Físico Sentinel-2 Real (10 m)",
                value=f"🔴 {perdida_s2_ha:.2f} ha · 🔥 {quema_s2_ha:.2f} ha",
                delta=(
                    "Rojo: ΔNDVI pérdida | Naranja: ΔNBR quema"
                    if overlays_s2
                    else "Sin recorte Sentinel-2 activo"
                ),
                delta_color="off",
                help="Superficie medida en EPSG:32718 sobre GeoTIFF reales Sentinel-2 L2A mediante LabelRule física explícita (ΔNDVI y ΔNBR+SWIR).",
            )

        telemetria_mapa = {
            "sync_pet": sync_pet,
            "alertas_visibles": total_alertas,
            "quemas_activas": quemas_reg_count,
            "area_perdida_s2_ha": perdida_s2_ha,
            "area_quema_s2_ha": quema_s2_ha,
        }

        divisor_bosque()

        # --- Selector de Experiencia ---
        col_modo, _ = st.columns([2.7, 1])
        with col_modo:
            modo_vista = st.radio(
                "Modo de Experiencia:",
                [
                    "🗺️ Consola Integral SIG & Fiscalización (Mapa en Tiempo Real)",
                    "🌟 Presentación Ejecutiva Unificada (Pitch Interactivo)",
                ],
                horizontal=True,
                index=0,
                help=(
                    "Elige 'Consola Integral SIG' para operar directamente sobre el mapa satelital en tiempo real "
                    "de pérdida de cobertura y quemas activas, o 'Presentación Ejecutiva' para flujo de pitch."
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
                df_focos_nasa=df_focos_vivo,
                mostrar_sentinel_copernicus=mostrar_sentinel_copernicus,
                overlays_sentinel2_reales=overlays_s2,
                mostrar_wms_geobosques=mostrar_wms_geobosques,
                telemetria_tiempo_real=telemetria_mapa,
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
                df_focos_nasa=df_focos_vivo,
                mostrar_sentinel_copernicus=mostrar_sentinel_copernicus,
                overlays_sentinel2_reales=overlays_s2,
                mostrar_wms_geobosques=mostrar_wms_geobosques,
                telemetria_tiempo_real=telemetria_mapa,
            )

    _panel_monitoreo_en_vivo()


if __name__ == "__main__":
    main()
