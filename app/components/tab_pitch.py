"""Componentes de Simulación Temporal multitemporal (Modo Pitch) y Presentación Ejecutiva."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app.components.map_view import crear_mapa_monitoreo
from app.components.tab_benchmark import render_tab_benchmark
from app.components.tab_inferencia import render_tab_laboratorio_inferencia
from app.services.alerts_data import generar_geojson_alertas
from app.services.expediente import generar_expediente_fema_serfor
from app.styles.theme import divisor_bosque
from src.sim import convertir_alertas_a_dataframe, generar_escenario_simulado


def render_tab_pitch() -> None:
    """Renderiza el Simulador Temporal (Modo Pitch) multitemporal de 8 semanas."""
    st.subheader(
        "Modo Pitch — Simulador Temporal Multitemporal (8 Semanas)",
        icon=":material/schedule:",
    )
    st.caption(
        "Evolución semanal multitemporal de alertas de cambio de cobertura y quemas agrícolas "
        "con motor pseudoaleatorio Mulberry32 determinista (`seed=42`)."
    )

    escenario = generar_escenario_simulado(
        semilla=42, pasos_semanales=8, fecha_inicio="2026-08-16", alertas_por_paso=2
    )
    serie = escenario["serie"]
    todas_alertas = escenario["alertas"]

    if "pitch_semana_idx" not in st.session_state:
        st.session_state.pitch_semana_idx = 0

    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    if col_b1.button("⏮ Inicio", use_container_width=True, help="Volver a la Semana 1"):
        st.session_state.pitch_semana_idx = 0
    if col_b2.button("◀ Anterior", use_container_width=True, help="Semana previa"):
        st.session_state.pitch_semana_idx = max(0, st.session_state.pitch_semana_idx - 1)
    if col_b3.button("▶ Siguiente", use_container_width=True, type="primary", help="Avanzar 1 semana"):
        st.session_state.pitch_semana_idx = min(len(serie) - 1, st.session_state.pitch_semana_idx + 1)
    if col_b4.button("⏭ Final", use_container_width=True, help="Ir a la última semana (Semana 8)"):
        st.session_state.pitch_semana_idx = len(serie) - 1
    if col_b5.button("↺ Reiniciar", use_container_width=True, help="Reiniciar simulación"):
        st.session_state.pitch_semana_idx = 0

    paso_sel = st.slider(
        "Línea temporal multitemporal (Paso semanal):",
        min_value=1,
        max_value=len(serie),
        value=st.session_state.pitch_semana_idx + 1,
        help="Arrastra el control para proyectar la evolución de alertas en Huánuco.",
    )
    st.session_state.pitch_semana_idx = paso_sel - 1
    idx = st.session_state.pitch_semana_idx

    punto_actual = serie[idx]
    fecha_actual = punto_actual["fecha"]
    alertas_acum = todas_alertas[: (idx + 1) * 2]
    alertas_paso = todas_alertas[idx * 2 : (idx + 1) * 2]

    area_carpish = punto_actual["carpish"]
    area_huallaga = punto_actual["huallaga"]
    area_acum_total = round(area_carpish + area_huallaga, 1)

    area_prev = (serie[idx - 1]["carpish"] + serie[idx - 1]["huallaga"]) if idx > 0 else 0.0
    tasa_semanal = round(area_acum_total - area_prev, 1)

    # Métricas de telemetría del paso activo
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    t_c1.metric(
        "Semana Activa",
        f"Semana {idx + 1} de {len(serie)}",
        f"Corte: {fecha_actual}",
        delta_color="off",
    )
    t_c2.metric(
        "Alertas Acumuladas",
        f"{len(alertas_acum)}",
        f"+{len(alertas_paso)} nuevas en este ciclo",
        delta_color="inverse" if len(alertas_paso) > 0 else "off",
    )
    t_c3.metric(
        "Área Acumulada Afectada",
        f"{area_acum_total:.1f} ha",
        f"Carpish: {area_carpish:.1f} ha | Huallaga: {area_huallaga:.1f} ha",
        delta_color="off",
    )
    t_c4.metric(
        "Tasa de Expansión Semanal",
        f"{tasa_semanal:.1f} ha / sem",
        "Variación vs semana previa",
        delta_color="inverse" if tasa_semanal > 0 else "off",
    )

    divisor_bosque()

    # Gráficos de evolución
    col_g1, col_g2 = st.columns([1.6, 1])
    with col_g1:
        st.markdown("**📈 Proyección Acumulada de Superficie Afectada (ha)**")
        df_serie = pd.DataFrame(serie[: idx + 1]).rename(
            columns={
                "fecha": "Fecha",
                "carpish": "Bosque Montano Carpish (ha)",
                "huallaga": "Valle del Huallaga (ha)",
            }
        )
        st.line_chart(
            df_serie.set_index("Fecha"),
            color=["#1b7837", "#2166ac"],
        )

    with col_g2:
        st.markdown("**🔥 Distribución de Eventos Acumulados**")
        df_acum = convertir_alertas_a_dataframe(alertas_acum)
        if not df_acum.empty:
            conteo_tipo = df_acum["tipo_evento"].value_counts()
            st.bar_chart(conteo_tipo, color="#2ca25f")
        else:
            st.info("Sin eventos acumulados.")

    st.markdown(f"**📋 Alertas generadas en la Semana {idx + 1} ({fecha_actual})**")
    df_paso = convertir_alertas_a_dataframe(alertas_paso)
    if not df_paso.empty:
        columnas_pitch = [
            "id_alerta",
            "origen_dato",
            "fecha",
            "zona",
            "sector",
            "tipo_evento",
            "severidad",
            "area_ha",
            "confianza_pct",
            "estado",
        ]
        nombres_pitch = {
            "id_alerta": "ID",
            "origen_dato": "Origen",
            "fecha": "Fecha",
            "zona": "Zona",
            "sector": "Sector",
            "tipo_evento": "Evento",
            "severidad": "Severidad",
            "area_ha": "Área (ha)",
            "confianza_pct": "Confianza (%)",
            "estado": "Estado",
        }
        st.dataframe(
            df_paso[columnas_pitch].rename(columns=nombres_pitch),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay nuevas alertas en este paso.")


def render_presentacion_unificada(
    df_alertas: pd.DataFrame,
    zona_sel: str,
    mapa_base_sel: str,
    mostrar_zonas: bool,
    mostrar_poligonos: bool,
    mostrar_marcadores: bool,
    mostrar_nasa_fires: bool,
    df_focos_nasa: pd.DataFrame,
    mapbox_token: str,
    mostrar_sentinel_copernicus: bool = True,
) -> None:
    """Renderiza una presentación ejecutiva unificada y fluida."""
    # HUD Banner Táctico
    st.markdown(
        """
        <div class="gf-hud-card">
          <div class="gf-hud-title">
            <span class="gf-hud-pulse"></span>
            <span>GEOFOREST-AI HUÁNUCO &nbsp;|&nbsp; PLATAFORMA INTEGRADA DE ALERTA TEMPRANA</span>
          </div>
          <div class="gf-hud-chips">
            <span class="gf-hud-chip">🛰️ SENTINEL-2 L2A (10m)</span>
            <span class="gf-hud-chip">🧠 U-NET + ONNX RUNTIME (FP32)</span>
            <span class="gf-hud-chip">📍 CARPISH & VALLE DEL HUALLAGA</span>
            <span class="gf-hud-chip">⚖️ EXPEDIENTE DEMOSTRATIVO (SIMULADO)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Stack tecnológico del proyecto ---
    with st.expander(
        "🛠️ Stack tecnológico empleado", expanded=False, icon=":material/build:"
    ):
        st.markdown(
            """
            | Capa | Tecnologías |
            |---|---|
            | **Lenguaje** | Python 3.14.8 |
            | **Geoespacial / GIS** | Rasterio 1.5.2 · GeoPandas 1.2.0 · Shapely · PyProj |
            | **Deep Learning / Inferencia** | PyTorch 2.14.1+cpu · ONNX 1.23.2 · ONNX Runtime 1.31.0 |
            | **Interfaz y cartografía** | Streamlit 1.66.0 · Folium 0.20.0 · Streamlit-Folium 0.27.4 |
            | **Animación / video** | HyperFrames (HTML/JS/CSS → MP4) |
            | **Simulación** | PRNG Mulberry32 determinista (paridad bit-a-bit con el visor React) |
            | **Pruebas** | Pytest (77 tests: 46 GIS · 11 U-Net · 16 inferencia · 4 simulación) |
            | **Tema visual** | «Niebla de dosel» (`.streamlit/config.toml`: Fraunces + Karla) |

            > **Nota honesta:** el modelo ONNX servido es de **referencia, sin entrenar** (FP32, ~31,1 MB, salida constante). Las métricas de benchmark son una plantilla de maquetación, no una evaluación medida. No hay cuantización Int8.
            """
        )

    # Bloque 1: Simulación Temporal de 8 Semanas & Mapa Satelital Sincronizado
    st.subheader(
        "1. Evolución Temporal & Monitoreo Satelital Sincronizado",
        icon=":material/schedule:",
    )
    st.caption(
        "Proyección semanal determinista basada en el PRNG Mulberry32 (`seed=42`). "
        "Avanza en la línea temporal para evaluar la propagación de eventos sobre el mapa satelital en vivo."
    )

    escenario = generar_escenario_simulado(
        semilla=42, pasos_semanales=8, fecha_inicio="2026-08-16", alertas_por_paso=2
    )
    serie = escenario["serie"]
    todas_alertas = escenario["alertas"]

    if "pitch_semana_idx" not in st.session_state:
        st.session_state.pitch_semana_idx = 0

    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    if col_b1.button("⏮ Inicio", use_container_width=True, key="pres_btn_inicio"):
        st.session_state.pitch_semana_idx = 0
    if col_b2.button("◀ Anterior", use_container_width=True, key="pres_btn_ant"):
        st.session_state.pitch_semana_idx = max(0, st.session_state.pitch_semana_idx - 1)
    if col_b3.button("▶ Siguiente Semana", use_container_width=True, type="primary", key="pres_btn_sig"):
        st.session_state.pitch_semana_idx = min(len(serie) - 1, st.session_state.pitch_semana_idx + 1)
    if col_b4.button("⏭ Final", use_container_width=True, key="pres_btn_fin"):
        st.session_state.pitch_semana_idx = len(serie) - 1
    if col_b5.button("↺ Reiniciar", use_container_width=True, key="pres_btn_rst"):
        st.session_state.pitch_semana_idx = 0

    paso_sel = st.slider(
        "Línea de tiempo de la proyección (Semanas proyectadas):",
        min_value=1,
        max_value=len(serie),
        value=st.session_state.pitch_semana_idx + 1,
        key="pres_slider_semana",
        help="Avanza en la línea temporal para evaluar la propagación de eventos en Huánuco.",
    )
    st.session_state.pitch_semana_idx = paso_sel - 1
    idx = st.session_state.pitch_semana_idx

    punto_actual = serie[idx]
    fecha_actual = punto_actual["fecha"]
    alertas_acum = todas_alertas[: (idx + 1) * 2]
    alertas_paso = todas_alertas[idx * 2 : (idx + 1) * 2]

    area_carpish = punto_actual["carpish"]
    area_huallaga = punto_actual["huallaga"]
    area_acum_total = round(area_carpish + area_huallaga, 1)

    area_prev = (serie[idx - 1]["carpish"] + serie[idx - 1]["huallaga"]) if idx > 0 else 0.0
    tasa_semanal = round(area_acum_total - area_prev, 1)

    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    t_c1.metric(
        "Semana Activa",
        f"Semana {idx + 1} de {len(serie)}",
        f"Corte: {fecha_actual}",
        delta_color="off",
    )
    t_c2.metric(
        "Alertas Acumuladas",
        f"{len(alertas_acum)}",
        f"+{len(alertas_paso)} nuevas en este ciclo",
        delta_color="inverse" if len(alertas_paso) > 0 else "off",
    )
    t_c3.metric(
        "Superficie Afectada Acumulada",
        f"{area_acum_total:.1f} ha",
        f"Carpish: {area_carpish:.1f} ha | Huallaga: {area_huallaga:.1f} ha",
        delta_color="off",
    )
    t_c4.metric(
        "Tasa de Expansión Semanal",
        f"{tasa_semanal:.1f} ha / sem",
        "Variación vs semana previa",
        delta_color="inverse" if tasa_semanal > 0 else "off",
    )

    col_mapa_pres, col_info_pres = st.columns([1.8, 1.2])
    with col_mapa_pres:
        col_m_t, col_m_p = st.columns([1.1, 1.4])
        with col_m_t:
            st.markdown(f"**🗺️ Mapa Satelital Activo — Semana {idx + 1}**")
        with col_m_p:
            st.caption("🧭 *Alterna capas satelitales en la esquina superior derecha o pulsa ⛶ Pantalla Completa.*")

        df_mapa_pitch = convertir_alertas_a_dataframe(alertas_acum)
        mapa_pres = crear_mapa_monitoreo(
            df_filtrado=df_mapa_pitch,
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
            mapa_pres,
            width=None,
            height=540,
            key=f"folium_pres_semana_{idx}",
            returned_objects=[],
        )

    with col_info_pres:
        st.markdown("**📈 Curva de Afectación Acumulada (ha)**")
        df_serie = pd.DataFrame(serie[: idx + 1]).rename(
            columns={
                "fecha": "Fecha",
                "carpish": "Bosque Montano Carpish (ha)",
                "huallaga": "Valle del Huallaga (ha)",
            }
        )
        st.line_chart(
            df_serie.set_index("Fecha"),
            color=["#1b7837", "#2166ac"],
            height=210,
        )

        st.markdown(f"**🚨 Nuevas Alertas en Semana {idx + 1} ({fecha_actual})**")
        df_paso = convertir_alertas_a_dataframe(alertas_paso)
        if not df_paso.empty:
            columnas_p = ["id_alerta", "sector", "severidad", "area_ha", "confianza_pct"]
            nombres_p = {
                "id_alerta": "ID",
                "sector": "Sector",
                "severidad": "Severidad",
                "area_ha": "Área (ha)",
                "confianza_pct": "Confianza (%)",
            }
            st.dataframe(df_paso[columnas_p].rename(columns=nombres_p), width="stretch", hide_index=True)
        else:
            st.info("Sin nuevas alertas en este corte.")

    divisor_bosque()

    # Bloque 2: Demostración en Tiempo Real de Inferencia IA (U-Net ONNX)
    render_tab_laboratorio_inferencia()

    divisor_bosque()

    # Bloque 3: Respaldo Científico & Benchmark Cuantizado
    render_tab_benchmark()

    divisor_bosque()

    # Bloque 4: Acción Legal Inmediata & Cadena de Custodia (FEMA / SERFOR)
    st.subheader(
        "4. Acción Legal Inmediata & Cadena de Custodia Probatoria",
        icon=":material/gavel:",
    )
    st.caption(
        "Generación automática de expedientes periciales con huella de integridad SHA-256 "
        "para intervención de la Fiscalía Especializada en Materia Ambiental (FEMA) y SERFOR."
    )

    df_legal = convertir_alertas_a_dataframe(alertas_acum)
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        expediente_txt = generar_expediente_fema_serfor(df_legal, zona_sel)
        st.download_button(
            label="📄 Descargar Expediente FEMA / SERFOR (.txt)",
            data=expediente_txt.encode("utf-8"),
            file_name=f"expediente_pericial_{zona_sel.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
            help="Informe pericial formal con hash SHA-256.",
        )
    with c_dl2:
        geojson_txt = generar_geojson_alertas(df_legal)
        st.download_button(
            label="🌐 Descargar Capa Vectorial (.geojson)",
            data=geojson_txt.encode("utf-8"),
            file_name=f"capa_alertas_{zona_sel.lower().replace(' ', '_')}.geojson",
            mime="application/geo+json",
            use_container_width=True,
            help="Capa vectorial estandarizada para QGIS o geoportales.",
        )
    with c_dl3:
        csv_bytes = df_legal.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Descargar Registro Tabular (.csv)",
            data=csv_bytes,
            file_name=f"registro_alertas_{zona_sel.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Datos tabulares completos.",
        )
