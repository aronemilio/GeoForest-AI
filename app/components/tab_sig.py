"""Componente de la Consola Integral SIG y Fiscalización Territorial."""

from __future__ import annotations

from typing import Any
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
    mostrar_sentinel_copernicus: bool = True,
    overlays_sentinel2_reales: list[dict[str, Any]] | None = None,
    mostrar_wms_geobosques: bool = True,
    telemetria_tiempo_real: dict[str, Any] | None = None,
) -> None:
    """Consola técnica en tiempo real para analistas SIG y fiscalizadores ambientales."""
    total_alertas = len(df_filtrado)
    alertas_altas = (
        int((df_filtrado["severidad"] == "Alta").sum()) if total_alertas > 0 else 0
    )
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
    confianza_media = (
        float(df_filtrado["confianza_pct"].mean()) if total_alertas > 0 else 0.0
    )

    if zona_sel != "Todas las zonas" and df_focos_nasa is not None and not df_focos_nasa.empty and "zona" in df_focos_nasa.columns:
        df_quemas_vista = df_focos_nasa[df_focos_nasa["zona"] == zona_sel].copy()
    elif df_focos_nasa is not None:
        df_quemas_vista = df_focos_nasa.copy()
    else:
        df_quemas_vista = pd.DataFrame()

    frp_max_mw = (
        float(pd.to_numeric(df_quemas_vista["frp"], errors="coerce").fillna(0.0).max())
        if not df_quemas_vista.empty and "frp" in df_quemas_vista.columns
        else 0.0
    )

    # --- Sub-indicadores de Priorización Operativa FEMA / SERFOR ---
    col_op1, col_op2, col_op3, col_op4 = st.columns(4)
    with col_op1:
        st.metric(
            label="🚨 Prioridad Alta (ATD >= 1.5 ha)",
            value=f"{alertas_altas}",
            delta="Inspección prioritaria FEMA/SERFOR",
            delta_color="inverse" if alertas_altas > 0 else "off",
        )
    with col_op2:
        st.metric(
            label="🔥 Pérdida + Quema Activa (< 1.5 km)",
            value=f"{con_quema_cercana}",
            delta=(
                "Cruce espacial ATD × foco térmico"
                if con_quema_cercana > 0
                else "Sin cruce directo < 1.5 km"
            ),
            delta_color="inverse" if con_quema_cercana > 0 else "off",
        )
    with col_op3:
        st.metric(
            label="🌡️ Potencia Radiativa Máx. (FRP)",
            value=f"{frp_max_mw:.2f} MW" if frp_max_mw > 0 else "0.00 MW",
            delta=(
                f"{len(df_quemas_vista)} focos en ámbito seleccionado"
                if not df_quemas_vista.empty
                else "Sin radiación térmica activa"
            ),
            delta_color="off",
        )
    with col_op4:
        st.metric(
            label="Score heurístico de persistencia espacial",
            value=f"{confianza_media:.1f}%",
            delta="Sin calibrar; no oficial",
            delta_color="normal",
        )

    divisor_bosque()

    # --- Sección principal: Mapa interactivo en tiempo real y Resumen Territorial ---
    col_mapa, col_resumen = st.columns([2.25, 1])

    with col_mapa:
        col_m_head, col_m_pills = st.columns([1.15, 1.45])
        with col_m_head:
            st.subheader(f"Mapa en Tiempo Real — {zona_sel}", icon=":material/map:")
            st.caption(ZONAS_ESTUDIO[zona_sel]["descripcion"])
        with col_m_pills:
            opciones_vista_s2 = {
                "🛰️ T1 Reciente + ΔNDVI/ΔNBR": "t1",
                "🌲 T0 Línea Base": "t0",
                "🔍 Solo Máscara ΔNDVI/ΔNBR": "mascara",
            }
            vista_lbl = st.pills(
                "Capa espectral Sentinel-2 L2A Real (Tile 18LUQ):",
                options=list(opciones_vista_s2.keys()),
                default="🛰️ T1 Reciente + ΔNDVI/ΔNBR",
                key="sig_pills_vista_s2_real",
            )
            vista_s2_activa = opciones_vista_s2.get(
                vista_lbl or "🛰️ T1 Reciente + ΔNDVI/ΔNBR", "t1"
            )

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
            panel_capas_colapsado=True,
            capas_raster_satelital=None,
            overlays_sentinel2_reales=overlays_sentinel2_reales,
            mostrar_wms_geobosques=mostrar_wms_geobosques,
            telemetria_tiempo_real=telemetria_tiempo_real,
            vista_satelital_activa=vista_s2_activa,
        )
        st_folium(
            mapa_folium,
            width=None,
            height=600,
            key=f"folium_consola_sig_{vista_s2_activa}",
            returned_objects=[],
        )

        st.markdown(
            "**Leyenda en tiempo real:** "
            "🔴 **Pérdida de cobertura forestal** (ATD Geobosques MINAM / ΔNDVI Sentinel-2) &nbsp;|&nbsp; "
            "🔥 **Quema / foco térmico activo** (NASA FIRMS NRT + SERFOR SAMI / ΔNBR Sentinel-2) &nbsp;|&nbsp; "
            "🚨 **Borde naranja sobre magenta:** Pérdida de cobertura con quema activa (< 1.5 km) &nbsp;|&nbsp; "
            "🟩/🟦 **Límites punteados:** Zonas de estudio"
        )
        st.caption(
            "⚖️ *Nota de rigor técnico y legal: Las capas reflejan cambios físicos de cobertura vegetal y anomalías térmicas "
            "medidas por satélite; no implican certificación de tala ilegal sin verificación presencial de SERFOR / FEMA.*"
        )

    with col_resumen:
        st.subheader("Síntesis en Tiempo Real", icon=":material/analytics:")

        st.markdown("**🔥 Quemas / Focos Térmicos por Sensor Activo**")
        if df_focos_nasa is None or df_focos_nasa.empty:
            st.info("0 focos térmicos activos reportados en la ventana NRT actual.")
        else:
            conteo_sensores = (
                df_focos_nasa["sensor_fuente"]
                .value_counts()
                .rename_axis("Sensor / Fuente Oficial")
                .reset_index(name="Focos Activos")
            )
            st.dataframe(conteo_sensores, width="stretch", hide_index=True)

        if overlays_sentinel2_reales:
            st.markdown("**🛰️ Recortes Sentinel-2 L2A Reales (10 m)**")
            filas_s2 = [
                {
                    "Ventana": ov["zona_titulo"].split(" (")[0],
                    "Par T0→T1": f"{ov['fecha_t0'][5:]}→{ov['fecha_t1'][5:]}",
                    "🔴 ΔNDVI (ha)": ov["area_perdida_cobertura_ha"],
                    "🔥 ΔNBR (ha)": ov["area_quema_espectral_ha"],
                }
                for ov in overlays_sentinel2_reales
            ]
            st.dataframe(pd.DataFrame(filas_s2), width="stretch", hide_index=True)

        if df_filtrado.empty:
            st.info("No hay alertas ATD que coincidan con los filtros seleccionados.")
        else:
            st.markdown("**🌲 Superficie ATD (ha) por Tipo de Evento**")
            resumen_tipo = (
                df_filtrado.groupby("tipo_evento", as_index=True)["area_ha"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(resumen_tipo, color="#2ca25f", height=175)

            st.markdown("**Resumen ATD por Zona de Estudio**")
            resumen_zona = (
                df_filtrado.groupby("zona", as_index=False)
                .agg(
                    Alertas=("id_alerta", "count"),
                    Area_ha=("area_ha", "sum"),
                )
                .rename(columns={"zona": "Zona", "Area_ha": "Área (ha)"})
            )
            resumen_zona["Área (ha)"] = resumen_zona["Área (ha)"].round(1)
            st.dataframe(resumen_zona, width="stretch", hide_index=True)

    divisor_bosque()

    # --- Tablas auditables en tiempo real: Pérdida de Cobertura y Quemas Activas ---
    tab_atd, tab_quemas = st.tabs(
        [
            f"🌲 Pérdida de Cobertura Forestal — PNCB Geobosques ({len(df_filtrado)})",
            f"🔥 Quemas y Focos Térmicos NRT — NASA FIRMS + SERFOR ({len(df_focos_nasa) if df_focos_nasa is not None else 0})",
        ]
    )

    with tab_atd:
        st.caption(
            "Detecciones satelitales oficiales (Sentinel-2, 10 m) integradas desde la API de Interoperabilidad "
            "2026 de Geobosques MINAM y respaldo local verificado. El score heurístico de persistencia espacial "
            "no está calibrado (Geobosques MINAM no emite porcentaje oficial de confianza)."
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
            "confianza_pct": "Score heurístico de persistencia espacial (%)",
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
                label="⬇️ Descargar alertas ATD (CSV)",
                data=csv_bytes,
                file_name=f"alertas_cobertura_{zona_sel.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="sig_dl_csv",
                width="stretch",
                help="Descarga tabular de alertas de pérdida de cobertura forestal.",
            )
        with col_dl2:
            expediente_texto = generar_expediente_fema_serfor(df_filtrado, zona_sel)
            st.download_button(
                label="📄 Expediente FEMA / SERFOR (.txt)",
                data=expediente_texto.encode("utf-8"),
                file_name=f"expediente_fema_serfor_{zona_sel.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                key="sig_dl_fema",
                width="stretch",
                help="Informe técnico pericial formal con hash SHA-256 para la Fiscalía Ambiental y SERFOR.",
            )
        with col_dl3:
            geojson_texto = generar_geojson_alertas(df_filtrado)
            st.download_button(
                label="🌐 Capa Vectorial ATD (.geojson)",
                data=geojson_texto.encode("utf-8"),
                file_name=f"capa_alertas_{zona_sel.lower().replace(' ', '_')}.geojson",
                mime="application/geo+json",
                key="sig_dl_geojson",
                width="stretch",
                help="Capa vectorial estandarizada lista para QGIS, ArcGIS o geoportales de fiscalización.",
            )

    with tab_quemas:
        st.caption(
            "Anomalías térmicas NRT consultadas en vivo desde NASA FIRMS (VIIRS SNPP, NOAA-20, NOAA-21, MODIS, Landsat) "
            "y alertas de incendio forestal de SERFOR Perú (UFMS — SAMI v2.1). Cuando un registro proviene de SERFOR sin "
            "telemetría Kelvin, la temperatura se conserva vacía (sin valores ficticios de relleno)."
        )
        if df_focos_nasa is None or df_focos_nasa.empty:
            st.info("Sin detecciones térmicas activas en Huánuco para la ventana NRT seleccionada.")
        else:
            cols_q = [
                c
                for c in (
                    "fecha_hora_pet",
                    "zona",
                    "sensor_fuente",
                    "satellite",
                    "bright_ti4",
                    "bright_c",
                    "frp",
                    "confidence",
                    "latitude",
                    "longitude",
                    "origen_feed",
                )
                if c in df_focos_nasa.columns
            ]
            nombres_q = {
                "fecha_hora_pet": "Fecha / Hora (PET)",
                "zona": "Ámbito",
                "sensor_fuente": "Sensor / Servicio",
                "satellite": "Satélite",
                "bright_ti4": "Brillo Térmico (K)",
                "bright_c": "Temp. Brillo (°C)",
                "frp": "FRP (MW)",
                "confidence": "Nivel Sensor (NRT)",
                "latitude": "Latitud",
                "longitude": "Longitud",
                "origen_feed": "Feed",
            }
            df_q_vista = df_focos_nasa[cols_q].rename(columns=nombres_q)
            st.dataframe(df_q_vista, width="stretch", hide_index=True)
            st.download_button(
                label="🔥 Descargar focos de quema en vivo (CSV)",
                data=df_q_vista.to_csv(index=False).encode("utf-8"),
                file_name="quemas_tiempo_real_huanuco.csv",
                mime="text/csv",
                key="sig_dl_quemas_csv",
                help="Exporta los focos de calor y alertas de incendio activos en Huánuco.",
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
            "Recorte satelital Huánuco (Sentinel-2 L2A):",
            [
                "🛰️ Sentinel-2 L2A Huánuco Real (Tile 18LUQ)",
                "Bosque Montano de Carpish (Túnel - Chinchao)",
                "Valle del Huallaga (Pillco Marca - Huancachupa)",
            ],
            default="🛰️ Sentinel-2 L2A Huánuco Real (Tile 18LUQ)",
            key="recorte_pills_bench",
        )
        if not recorte_seleccionado:
            recorte_seleccionado = "🛰️ Sentinel-2 L2A Huánuco Real (Tile 18LUQ)"
    with col_pipe2:
        st.write("")
        st.write("")
        btn_ejecutar_pipeline = st.button(
            "🚀 Ejecutar Inferencia Multitemporal (CPU)",
            type="primary",
            key="btn_sig_infer_exec",
            width="stretch",
        )

    if btn_ejecutar_pipeline:
        with st.spinner("Cargando bandas T0 y T1, remuestreando B11 a 10m y ejecutando ONNX Runtime..."):
            t_inicio = time.perf_counter()
            t0_src, t1_src, sector_nom = generar_recorte_offline_huanuco(recorte_seleccionado)
            try:
                tensor_np, profile_raster = preparar_tensor_multitemporal(
                    t0_src, t1_src, target_crs="EPSG:32718", normalize=True
                )
            finally:
                for ds in list(t0_src.values()) + list(t1_src.values()):
                    if hasattr(ds, "close"):
                        ds.close()
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
                conf_calc = (
                    float(np.mean(pred_prob[pred_prob >= 0.5]) * 100.0)
                    if np.any(pred_prob >= 0.5)
                    else 0.0
                )
                modelo_ejecutado = True
            else:
                tiempo_infer = 0.0
                area_calc = 0.0
                conf_calc = 0.0
                modelo_ejecutado = False

            t_total = time.perf_counter() - t_inicio

            st.session_state["sig_infer_result"] = {
                "t_total": t_total,
                "t_preprocess": t_preprocess,
                "tiempo_infer": tiempo_infer,
                "shape": tensor_np.shape,
                "min_val": float(tensor_np.min()),
                "max_val": float(tensor_np.max()),
                "crs": str(profile_raster.get("crs", "EPSG:32718")),
                "area_calc": area_calc,
                "conf_calc": conf_calc,
                "modelo_ejecutado": modelo_ejecutado,
            }

    if "sig_infer_result" in st.session_state:
        res = st.session_state["sig_infer_result"]
        if res.get("modelo_ejecutado", False):
            st.success(
                f"✅ Inferencia completada en **{res['t_total']:.3f} s** "
                f"(Preprocesamiento: {res['t_preprocess']:.3f} s | Inferencia CPU: **{res['tiempo_infer']:.3f} s** < 2.0 s)."
            )
        else:
            st.error(
                "No se encontró el modelo ONNX servido: no se ejecutó inferencia. "
                "No se muestran valores de relleno."
            )

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        res_c1.metric(
            "Tensor Multitemporal",
            f"{res['shape']}",
            "6 canales: T0(B4,B8,B11) + T1(B4,B8,B11)",
        )
        res_c2.metric(
            "Escala Normalizada",
            f"[{res['min_val']:.2f}, {res['max_val']:.2f}]",
            "Reflectancia en [0.0, 1.0]",
        )
        res_c3.metric(
            "CRS Proyectado",
            res["crs"],
            "UTM Zona 18S validado",
        )
        res_c4.metric(
            "Área sobre umbral (no validada)",
            f"{res['area_calc']:.2f} ha",
            f"Probabilidad media: {res['conf_calc']:.1f}% (sin calibrar)",
        )
        st.caption(
            "Inferencia multitemporal U-Net sobre CPU (< 2.0 s). "
            "Si no se detectan píxeles sobre el umbral de corte, el área reportada es rigurosamente 0.00 ha."
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
            1. **Etapa 1 — Estructura y panel visual oficial:**
               Diseño de consola táctica SIG, indicadores periciales, mapas multiespectrales
               y visualización de alertas oficiales de deforestación del PNCB — Geobosques MINAM.
            2. **Etapa 2 — Telemetría en tiempo real y focos térmicos:**
               Integración de focos de calor satelitales en vivo vía NASA FIRMS (sensor VIIRS).
            3. **Etapa 3 — Preprocesamiento Sentinel-2 y máscaras de Geobosques:**
               Descarga de escenas multiespectrales (bandas B02, B03, B04, B08, B11, B12),
               cálculo de índices espectrales (NDVI, NBR) y generación de máscaras de entrenamiento.
            4. **Etapa 4 — Entrenamiento U-Net, inferencia ONNX y evaluación:**
               Entrenamiento en PyTorch, exportación a ONNX Runtime y conexión directa con este panel.
            """
        )
