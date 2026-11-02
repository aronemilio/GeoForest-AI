"""Componentes de Evolución Temporal multitemporal (Modo Pitch) y Presentación Ejecutiva."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from app.components.map_view import crear_mapa_monitoreo
from app.components.tab_benchmark import render_tab_benchmark
from app.components.tab_inferencia import render_tab_laboratorio_inferencia
from app.services.alerts_data import generar_escenario_geobosques, generar_geojson_alertas
from app.services.expediente import generar_expediente_fema_serfor
from app.services.onnx_demo import (
    calcular_telemetria_espectral_tensor,
    generar_capas_satelitales_para_mapa,
    generar_parche_demostracion,
    matriz_a_imagen_delta,
    matriz_a_imagen_mascara,
    matriz_a_imagen_rgb,
    obtener_secuencia_cambio_drastico_huanuco,
    obtener_zonas_impacto_notable,
    poligono_para_zona,
)
from app.styles.theme import divisor_bosque
from src.sim import convertir_alertas_a_dataframe


def render_tab_pitch() -> None:
    """Renderiza la Evolución Temporal (Modo Pitch) multitemporal de 8 semanas."""
    st.subheader(
        "Modo Pitch — Evolución Temporal Oficial (8 Semanas)",
        icon=":material/schedule:",
    )
    st.caption(
        "Evolución semanal multitemporal de alertas de deforestación y quemas agrícolas "
        "basada en registros satelitales oficiales del PNCB — Geobosques MINAM."
    )

    escenario = generar_escenario_geobosques(pasos_semanales=8)
    serie = escenario["serie"]
    todas_alertas = escenario["alertas"]

    if "pitch_tab_semana" not in st.session_state:
        st.session_state["pitch_tab_semana"] = 1

    def _tab_inicio():
        st.session_state["pitch_tab_semana"] = 1

    def _tab_ant():
        st.session_state["pitch_tab_semana"] = max(1, st.session_state["pitch_tab_semana"] - 1)

    def _tab_sig(limite: int):
        st.session_state["pitch_tab_semana"] = min(limite, st.session_state["pitch_tab_semana"] + 1)

    def _tab_fin(limite: int):
        st.session_state["pitch_tab_semana"] = limite

    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
    col_b1.button("⏮ Inicio", key="tab_btn_inicio", on_click=_tab_inicio, width="stretch", help="Volver a la Semana 1")
    col_b2.button("◀ Anterior", key="tab_btn_ant", on_click=_tab_ant, width="stretch", help="Semana previa")
    col_b3.button("▶ Siguiente", key="tab_btn_sig", type="primary", on_click=_tab_sig, args=(len(serie),), width="stretch", help="Avanzar 1 semana")
    col_b4.button("⏭ Final", key="tab_btn_fin", on_click=_tab_fin, args=(len(serie),), width="stretch", help="Ir a la última semana (Semana 8)")
    col_b5.button("↺ Reiniciar", key="tab_btn_rst", on_click=_tab_inicio, width="stretch", help="Reiniciar línea temporal")

    paso_sel = st.slider(
        "Línea temporal multitemporal (Paso semanal):",
        min_value=1,
        max_value=len(serie),
        key="pitch_tab_semana",
        help="Arrastra el control para proyectar la evolución de alertas en Huánuco.",
    )
    idx = paso_sel - 1

    punto_actual = serie[idx]
    fecha_actual = punto_actual["fecha"]
    if todas_alertas and "paso" in todas_alertas[0]:
        alertas_acum = [a for a in todas_alertas if a.get("paso", 0) <= idx]
        alertas_paso = [a for a in todas_alertas if a.get("paso", 0) == idx]
    else:
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
        df_serie = pd.DataFrame(serie[: idx + 1])[["fecha", "carpish", "huallaga"]].rename(
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

    st.markdown(f"**📋 Alertas oficiales registradas en la Semana {idx + 1} ({fecha_actual})**")
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
            "confianza_pct": "Score heurístico de persistencia espacial (%)",
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
    mostrar_sentinel_copernicus: bool = True,
    overlays_sentinel2_reales: list[dict[str, Any]] | None = None,
    mostrar_wms_geobosques: bool = True,
    telemetria_tiempo_real: dict[str, Any] | None = None,
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
            <span class="gf-hud-chip">⚖️ EXPEDIENTE PERICIAL DIGITAL</span>
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
            | **Lenguaje** | Python 3.10+ |
            | **Geoespacial / GIS** | Rasterio 1.4.4 · GeoPandas 1.1.4 · Shapely 2.1.2 · PyProj 3.7.1 |
            | **Deep Learning / Inferencia** | PyTorch 2.5.1+cu121 · ONNX 1.23.2 · ONNX Runtime 1.23.2 |
            | **Interfaz y cartografía** | Streamlit 1.65.0 · Folium 0.20.0 · Streamlit-Folium 0.26.2 |
            | **Animación / video** | HyperFrames (HTML/JS/CSS → MP4) |
            | **Series Temporales** | PRNG Mulberry32 determinista |
            | **Pruebas** | Pytest (103 tests automatizados con 100% de éxito) |
            | **Tema visual** | «Niebla de dosel» (`.streamlit/config.toml`) |
            """
        )

    # Bloque 1: Comparativa Satelital Multitemporal & Evolución Temporal
    st.subheader(
        "1. Detección Multitemporal de Cambio: Tala y Quemas Notables en Huánuco",
        icon=":material/compare:",
    )
    st.caption(
        "Auditoría visual del par satelital Sentinel-2 L2A (T0: Antes vs T1: Después) "
        "y propagación temporal de eventos oficiales del PNCB — Geobosques MINAM."
    )

    modo_analisis_temporal = st.radio(
        "Enfoque de Análisis Territorial:",
        [
            "🛰️ Zonas Críticas con Tala y Quemas Notables (Comparativa Antes vs Después)",
            "📅 Proyección Semanal Sincronizada (Evolución de 8 Semanas)",
        ],
        horizontal=True,
        index=0,
        key="pitch_modo_analisis_temporal",
        help="Elige 'Zonas Críticas' para comparar fotografías satelitales directas de antes vs después, o 'Proyección Semanal' para ver el avance acumulado de alertas.",
    )

    if modo_analisis_temporal.startswith("🛰️"):
        zonas_dict = obtener_zonas_impacto_notable()
        opciones_zonas = list(zonas_dict.keys())

        col_z_sel, col_z_badge = st.columns([2.5, 1.2])
        with col_z_sel:
            zona_activa_key = st.selectbox(
                "🎯 Seleccionar Zona de Impacto Notorio para Auditar:",
                options=opciones_zonas,
                format_func=lambda k: f"{zonas_dict[k]['titulo']} [{zonas_dict[k]['tipo_tag']}]",
                key="pitch_sel_zona_critica",
                help="Zonas prioritarias con eventos severos de deforestación o quema agrícola confirmada.",
            )
        info_z = zonas_dict[zona_activa_key]

        with col_z_badge:
            st.write("")
            st.markdown(
                f"""
                <div style="background: rgba(225, 29, 72, 0.12); border: 1px solid rgba(225, 29, 72, 0.45); border-radius: 8px; padding: 10px; text-align: center;">
                    <span style="font-weight: 700; color: {info_z['tipo_color']}; font-size: 13px;">🚨 {info_z['tipo_tag']}</span>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 3px;">📍 {info_z['coordenadas']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Cargar parches de imágenes del dataset y computar telemetría espectral directa del tensor
        tensor_zona, meta_zona, nom_zona = generar_parche_demostracion(info_z["sector_idx"])
        telem_z = calcular_telemetria_espectral_tensor(tensor_zona)

        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        c_k1.metric(
            "Diagnóstico Biofísico",
            telem_z["diagnostico"].split(" / ")[0],
            telem_z["severidad"],
            help="Clasificación biofísica calculada directamente del tensor.",
        )
        c_k2.metric(
            "Superficie Afectada",
            f"{telem_z['area_afectada_ha']:.1f} ha",
            f"{telem_z['num_px_afectados']:,} píxeles de 10m",
            delta_color="off",
            help="Superficie estimada por acumulación de píxeles con ΔNDVI < -0.15 o ΔNBR < -0.18.",
        )
        c_k3.metric(
            "Anomalía Biofísica en Núcleo",
            f"ΔNDVI {telem_z['delta_ndvi_nucleo']:.2f}",
            f"ΔNBR {telem_z['delta_nbr_nucleo']:.2f}",
            delta_color="inverse",
            help="Caída fotosintética en el centro de la perturbación.",
        )
        c_k4.metric(
            "Caída de Canopia NIR (B8)",
            f"-{telem_z['b8_caida_pct']:.1f}%",
            f"Alza Térmica SWIR: +{telem_z['b11_alza_pct']:.1f}%",
            delta_color="inverse" if telem_z["b8_caida_pct"] > 25 else "off",
            help="Contraste de reflectancia espectral entre T0 (bosque denso) y T1 (suelo expuesto/fuego).",
        )

        img_t0 = matriz_a_imagen_rgb(tensor_zona[1], tensor_zona[0], tensor_zona[2])
        img_t1 = matriz_a_imagen_rgb(tensor_zona[4], tensor_zona[3], tensor_zona[5])
        img_dndvi = matriz_a_imagen_delta(telem_z["delta_ndvi_matriz"])
        anomalia_gt = meta_zona.get("anomalia_gt")
        if anomalia_gt is not None:
            img_mask = matriz_a_imagen_mascara(anomalia_gt.astype(np.uint8))
        else:
            img_mask = matriz_a_imagen_mascara(telem_z["mascara_cambio"].astype(np.uint8))

        st.markdown(f"#### 🛰️ Centro de Mando Territorial — {info_z['titulo']}")

        # Layout en 2 Columnas de Alta Eficiencia: Mapa a la izquierda, Tensores y Firma Espectral a la derecha
        col_mapa_left, col_tensor_right = st.columns([1.65, 1.35])

        with col_mapa_left:
            col_ctrl1, col_ctrl2 = st.columns([1.5, 1.3])
            with col_ctrl1:
                modo_capa_sat = st.pills(
                    "🛰️ Proyección Satelital sobre Terreno:",
                    options=[
                        "🚨 T1 Después (Tala / Quema Notoria)",
                        "🌲 T0 Antes (Bosque Intacto)",
                        "🔍 Segmentación U-Net",
                    ],
                    default="🚨 T1 Después (Tala / Quema Notoria)",
                    key="pitch_sel_modo_sat",
                    help="Proyecta la imagen satelital multiespectral directamente sobre el relieve de Huánuco.",
                )
                if not modo_capa_sat:
                    modo_capa_sat = "🚨 T1 Después (Tala / Quema Notoria)"
            with col_ctrl2:
                usar_s2_real_pitch = st.toggle(
                    "🛰️ Capa Sentinel-2 L2A Real (2026)",
                    value=bool(overlays_sentinel2_reales),
                    key="pitch_toggle_s2_real",
                    help="Activa el recorte GeoTIFF Sentinel-2 L2A real (ΔNDVI pérdida / ΔNBR quema). Al desactivarlo, proyecta la escena demostrativa del Modo Pitch (simulada).",
                )
                st.caption("🧭 *⛶ Pantalla completa (sup. izq.) | 🎛️ Capas (sup. der.)*")

            vista_sat = "t1" if "T1" in modo_capa_sat else ("t0" if "T0" in modo_capa_sat else "mascara")
            capas_sat = None if (usar_s2_real_pitch and overlays_sentinel2_reales) else generar_capas_satelitales_para_mapa(info_z["id"])
            overlays_activos = overlays_sentinel2_reales if (usar_s2_real_pitch and overlays_sentinel2_reales) else None
            secuencia = obtener_secuencia_cambio_drastico_huanuco(info_z["id"])
            poly_destacado = poligono_para_zona(info_z["lat"], info_z["lon"], radio_deg=0.009)

            mapa_zona = crear_mapa_monitoreo(
                df_filtrado=df_alertas,
                zona_seleccionada=zona_sel,
                mapa_base=mapa_base_sel,
                mostrar_zonas=mostrar_zonas,
                mostrar_poligonos_alerta=mostrar_poligonos,
                mostrar_marcadores=mostrar_marcadores,
                mostrar_nasa_fires=mostrar_nasa_fires,
                df_nasa_firms=df_focos_nasa,
                mostrar_sentinel_copernicus=mostrar_sentinel_copernicus,
                centro_override=[info_z["lat"], info_z["lon"]],
                zoom_override=15,
                poligono_destacado=poly_destacado,
                etiqueta_destacada=f"{info_z['evento']} ({info_z['area_estimada_ha']} ha)",
                secuencia_hitos=secuencia,
                panel_capas_colapsado=True,
                capas_raster_satelital=capas_sat,
                overlays_sentinel2_reales=overlays_activos,
                mostrar_wms_geobosques=mostrar_wms_geobosques,
                telemetria_tiempo_real=telemetria_tiempo_real,
                vista_satelital_activa=vista_sat,
            )
            st_folium(
                mapa_zona,
                width=None,
                height=490,
                key=f"folium_zona_{info_z['id']}_{vista_sat}",
                returned_objects=[],
            )

            st.caption("⏱️ **Secuencia de Avance del Frente (PNCB Geobosques & Sentinel-2):**")
            cols_hitos = st.columns(len(secuencia))
            for col_h, hito in zip(cols_hitos, secuencia):
                with col_h:
                    st.markdown(
                        f"""
                        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(225, 29, 72, 0.35); border-radius: 6px; padding: 6px 3px; text-align: center;">
                          <span style="font-weight: 700; color: {hito['color']}; font-size: 11px;">{hito['numero_romano']} {hito['fecha'][5:]}</span>
                          <div style="font-size: 11px; font-weight: 600; color: #f1f5f9; margin-top: 1px;">+{hito['area_paso_ha']:.1f} ha</div>
                          <div style="font-size: 10px; color: #94a3b8;">{hito['area_acum_ha']:.1f} ha</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with col_tensor_right:
            st.markdown(f"**🔬 Telemetría y Análisis Multiespectral del Tensor**")
            tabs_tensor = st.tabs([
                "📷 Antes vs Después",
                "📊 Firma Espectral",
                "🔍 U-Net & ΔNDVI",
                "📋 Ficha Técnica",
            ])
            with tabs_tensor[0]:
                c_sub0, c_sub1 = st.columns(2)
                with c_sub0:
                    st.image(img_t0, caption=f"T0: 2024-07-18\nDosel Intacto", width="stretch")
                with c_sub1:
                    st.image(img_t1, caption=f"T1: 2024-09-11\n{info_z['tipo_tag']}", width="stretch")
                st.caption(f"📍 **Diagnóstico:** {telem_z['diagnostico']}")

            with tabs_tensor[1]:
                st.caption("📈 **Comparativa de Reflectancia Espectral en Superficie (T0 vs T1):**")
                st.bar_chart(telem_z["df_firmas"].set_index("Banda Espectral"), color=["#10b981", "#ef4444"], height=230)
                st.caption(
                    f"• Caída en Infrarrojo Cercano (B8): **-{telem_z['b8_caida_pct']:.1f}%** (colapso fotosintético).\n"
                    f"• Alza en Onda Corta SWIR (B11): **+{telem_z['b11_alza_pct']:.1f}%** (suelo mineral seco o calor)."
                )

            with tabs_tensor[2]:
                c_sub2, c_sub3 = st.columns(2)
                with c_sub2:
                    st.image(img_mask, caption="Máscara IA U-Net\n(Píxeles > Umbral)", width="stretch")
                with c_sub3:
                    st.image(img_dndvi, caption="Mapa Pericial ΔNDVI\n(Rojo = Deforestado)", width="stretch")
                st.caption(f"Superficie afectada: **{telem_z['area_afectada_ha']} ha** ({telem_z['num_px_afectados']:,} píxeles clasificados).")

            with tabs_tensor[3]:
                st.markdown(
                    f"""
                    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px; font-size: 12px;">
                        <div style="color: #38bdf8; font-weight: 700; margin-bottom: 6px;">📍 DATOS DE TELEMETRÍA DEL PARCHE</div>
                        <div><b>Sector:</b> {info_z['titulo']}</div>
                        <div><b>Coordenadas:</b> {info_z['coordenadas']}</div>
                        <div><b>Dimensiones:</b> 256×256 px a 10m (6.55 km² evaluados)</div>
                        <div><b>NDVI Medio T0:</b> {telem_z['ndvi_t0_media']} ➔ <b>T1:</b> {telem_z['ndvi_t1_media']}</div>
                        <div><b>NBR Medio T0:</b> {telem_z['nbr_t0_media']} ➔ <b>T1:</b> {telem_z['nbr_t1_media']}</div>
                        <div><b>ΔNDVI en Núcleo:</b> <span style="color:#ef4444; font-weight:700;">{telem_z['delta_ndvi_nucleo']:.3f}</span></div>
                        <div><b>ΔNBR en Núcleo:</b> <span style="color:#ef4444; font-weight:700;">{telem_z['delta_nbr_nucleo']:.3f}</span></div>
                        <div style="margin-top: 6px; color:#10b981;"><b>Estado FEMA:</b> Flag de inspección inmediata generado.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        alertas_acum = df_alertas
    else:
        # Modo Proyección Semanal de 8 Semanas
        escenario = generar_escenario_geobosques(pasos_semanales=8)
        serie = escenario["serie"]
        todas_alertas = escenario["alertas"]

        if "pres_slider_semana" not in st.session_state:
            st.session_state["pres_slider_semana"] = 1

        def _pres_inicio():
            st.session_state["pres_slider_semana"] = 1

        def _pres_ant():
            st.session_state["pres_slider_semana"] = max(1, st.session_state["pres_slider_semana"] - 1)

        def _pres_sig(limite: int):
            st.session_state["pres_slider_semana"] = min(limite, st.session_state["pres_slider_semana"] + 1)

        def _pres_fin(limite: int):
            st.session_state["pres_slider_semana"] = limite

        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        col_b1.button("⏮ Inicio", key="pres_btn_inicio", on_click=_pres_inicio, width="stretch", help="Ir a Semana 1")
        col_b2.button("◀ Anterior", key="pres_btn_ant", on_click=_pres_ant, width="stretch", help="Semana previa")
        col_b3.button("▶ Siguiente Semana", key="pres_btn_sig", type="primary", on_click=_pres_sig, args=(len(serie),), width="stretch", help="Avanzar 1 semana")
        col_b4.button("⏭ Final", key="pres_btn_fin", on_click=_pres_fin, args=(len(serie),), width="stretch", help="Ir a la última semana")
        col_b5.button("↺ Reiniciar", key="pres_btn_rst", on_click=_pres_inicio, width="stretch", help="Reiniciar línea temporal")

        paso_sel = st.slider(
            "Línea de tiempo de la proyección (Semanas proyectadas):",
            min_value=1,
            max_value=len(serie),
            key="pres_slider_semana",
            help="Avanza en la línea temporal para evaluar la propagación de eventos en Huánuco.",
        )
        idx = paso_sel - 1

        punto_actual = serie[idx]
        fecha_actual = punto_actual["fecha"]
        if todas_alertas and "paso" in todas_alertas[0]:
            alertas_acum = [a for a in todas_alertas if a.get("paso", 0) <= idx]
            alertas_paso = [a for a in todas_alertas if a.get("paso", 0) == idx]
        else:
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
                st.caption("🧭 *⛶ Expandir pantalla completa (sup. izq.) | 🎛️ Abrir/Cerrar capas y satélites (sup. der.).*")

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
                secuencia_hitos=obtener_secuencia_cambio_drastico_huanuco("carpish_chinchao"),
                panel_capas_colapsado=True,
                capas_raster_satelital=None if overlays_sentinel2_reales else generar_capas_satelitales_para_mapa("carpish_tala"),
                overlays_sentinel2_reales=overlays_sentinel2_reales,
                mostrar_wms_geobosques=mostrar_wms_geobosques,
                telemetria_tiempo_real=telemetria_tiempo_real,
                vista_satelital_activa="t1",
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
            df_serie = pd.DataFrame(serie[: idx + 1])[["fecha", "carpish", "huallaga"]].rename(
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

            st.markdown(f"**🚨 Nuevas Alertas Oficiales en Semana {idx + 1} ({fecha_actual})**")
            df_paso = convertir_alertas_a_dataframe(alertas_paso)
            if not df_paso.empty:
                columnas_p = ["id_alerta", "sector", "severidad", "area_ha", "confianza_pct"]
                nombres_p = {
                    "id_alerta": "ID",
                    "sector": "Sector",
                    "severidad": "Severidad",
                    "area_ha": "Área (ha)",
                    "confianza_pct": "Score heurístico de persistencia espacial (%)",
                }
                st.dataframe(df_paso[columnas_p].rename(columns=nombres_p), width="stretch", hide_index=True)
            else:
                st.info("Sin nuevas alertas en este corte.")

    divisor_bosque()

    # Bloque 2: Inferencia en Tiempo Real IA (U-Net ONNX)
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

    if isinstance(alertas_acum, pd.DataFrame):
        df_legal = alertas_acum
    else:
        df_legal = convertir_alertas_a_dataframe(alertas_acum)
    c_dl1, c_dl2, c_dl3 = st.columns(3)
    with c_dl1:
        expediente_txt = generar_expediente_fema_serfor(df_legal, zona_sel)
        st.download_button(
            label="📄 Descargar Expediente FEMA / SERFOR (.txt)",
            data=expediente_txt.encode("utf-8"),
            file_name=f"expediente_pericial_{zona_sel.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            key="pres_dl_expediente",
            width="stretch",
            help="Informe pericial formal con hash SHA-256.",
        )
    with c_dl2:
        geojson_txt = generar_geojson_alertas(df_legal)
        st.download_button(
            label="🌐 Descargar Capa Vectorial (.geojson)",
            data=geojson_txt.encode("utf-8"),
            file_name=f"capa_alertas_{zona_sel.lower().replace(' ', '_')}.geojson",
            mime="application/geo+json",
            key="pres_dl_geojson",
            width="stretch",
            help="Capa vectorial estandarizada para QGIS o geoportales.",
        )
    with c_dl3:
        csv_bytes = df_legal.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Descargar Registro Tabular (.csv)",
            data=csv_bytes,
            file_name=f"registro_alertas_{zona_sel.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            key="pres_dl_csv",
            width="stretch",
            help="Datos tabulares completos.",
        )
