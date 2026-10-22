"""Pestaña interactiva de ejecución real de inferencia U-Net sobre CPU con ONNX Runtime."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rasterio.crs import CRS
import streamlit as st
import time

from app.services.copernicus import (
    CATALOGO_ESCENAS_REALES,
    cargar_escena_satelital_seleccionada,
    consultar_catalogo_copernicus,
    descargar_escena_sentinel2,
)
from app.services.onnx_demo import (
    matriz_a_imagen_heatmap,
    matriz_a_imagen_mascara,
    matriz_a_imagen_ndvi,
    matriz_a_imagen_rgb,
)
from app.styles.theme import divisor_bosque
from src.infer import (
    ONNXForestPredictor,
    apply_biophysical_filter,
    binarize,
    calculate_affected_area,
    calibrate_probabilities,
    compute_otsu_threshold,
    extract_alerts,
)


def render_tab_laboratorio_inferencia() -> None:
    """Pestaña interactiva de ejecución real de inferencia U-Net sobre CPU."""
    st.subheader(
        "🔬 Laboratorio de Inferencia U-Net — Ejecución en CPU (ONNX Runtime)",
        icon=":material/biotech:",
    )
    st.caption(
        "Ejecución en tiempo real del modelo de aprendizaje profundo sobre imágenes satelitales multiespectrales "
        "Sentinel-2 (10m) en Huánuco (Carpish y Valle del Huallaga). Mide latencia de CPU, extrae polígonos "
        "vectoriales y calcula superficies afectadas con base matemática formal."
    )
    st.info(
        "🛰️ **Modo Operativo Satelital 100% Real**: Los tensores analizados corresponden a imágenes multiespectrales "
        "Sentinel-2 L2A (10 m/px) adquiridas por la Agencia Espacial Europea (ESA / Copernicus CDSE) "
        "sobre Huánuco y reservas forestales de referencia. Inferencia ejecutada mediante red neuronal U-Net en ONNX Runtime (CPU).",
        icon=":material/satellite_alt:",
    )

    modelo_onnx_path = Path("models/unet_sentinel2_huanuco.onnx")
    if not modelo_onnx_path.exists():
        modelo_onnx_path = Path("models/unet_sentinel2_multitemporal.onnx")

    if not modelo_onnx_path.exists():
        st.error(
            f"❌ No se encontró el modelo ONNX en `{modelo_onnx_path}`. "
            "Asegúrate de que los pesos estén exportados en `models/`."
        )
        return

    # --- Consulta en Vivo API Copernicus CDSE ---
    with st.expander("📡 Telemetría y Consulta en Vivo: Catálogo Copernicus (Sentinel-2 Huánuco)", expanded=False, icon=":material/satellite_alt:"):
        st.markdown(
            "Conexión directa con la API de Copernicus para explorar las pasadas orbitales "
            "de Sentinel-2A y Sentinel-2B sobre el Bounding Box de Huánuco (MGRS Tile `18LUQ`)."
        )
        c_q1, c_q2, c_q3 = st.columns([1.5, 1.5, 1])
        with c_q1:
            f_ini = st.date_input("Fecha Inicio", value=pd.to_datetime("2024-06-01"), key="cop_f_ini")
        with c_q2:
            f_fin = st.date_input("Fecha Fin", value=pd.to_datetime("2024-10-31"), key="cop_f_fin")
        with c_q3:
            st.write("")
            btn_buscar_cop = st.button("🔍 Consultar Catálogo", key="btn_query_copernicus", use_container_width=True)

        if btn_buscar_cop or "catalogo_copernicus_cache" in st.session_state:
            if btn_buscar_cop:
                with st.spinner("Consultando API STAC de Copernicus..."):
                    st.session_state["catalogo_copernicus_cache"] = consultar_catalogo_copernicus(
                        fecha_inicio=str(f_ini),
                        fecha_fin=str(f_fin),
                        max_nubes_pct=30.0,
                        limite=8,
                    )
            escenas = st.session_state.get("catalogo_copernicus_cache", [])
            if escenas:
                st.success(f"Encontradas {len(escenas)} escenas satelitales Sentinel-2 L2A sobre Huánuco (Tile 18LUQ).")
                df_esc = pd.DataFrame(escenas)[["id", "fecha", "plataforma", "tile", "nubes_pct"]]
                df_esc.columns = ["ID Escena", "Fecha Adquisición", "Plataforma", "Tile MGRS", "Cobertura Nubosa (%)"]
                st.dataframe(df_esc, use_container_width=True, hide_index=True)

                col_dl1, col_dl2 = st.columns([2.5, 1])
                with col_dl1:
                    opciones_esc = [f"{e['fecha']} — {e['id']} (nubosidad: {e['nubes_pct']}%)" for e in escenas]
                    sel_idx = st.selectbox(
                        "Seleccionar escena para descargar a data/raw/sentinel2/:",
                        range(len(opciones_esc)),
                        format_func=lambda i: opciones_esc[i],
                        key="sel_esc_descarga_ui",
                    )
                with col_dl2:
                    st.write("")
                    st.write("")
                    btn_descargar = st.button("📥 Descargar GeoTIFFs", key="btn_dl_copernicus", use_container_width=True)

                if btn_descargar:
                    escena_sel = escenas[sel_idx]
                    with st.spinner(f"Descargando GeoTIFFs (B04, B08, B11) para {escena_sel['id']}..."):
                        try:
                            p_out = descargar_escena_sentinel2(escena_sel)
                            st.success(f"✅ Guardado en `{p_out.name}`: B04.tif, B08.tif y B11.tif listos para `src/dataset.py`.")
                        except Exception as dl_err:
                            st.error(f"Error en la descarga: {dl_err}")
            else:
                st.info("No se encontraron escenas con el filtro seleccionado.")

    if "escena_satelital_activa" not in st.session_state:
        st.session_state["escena_satelital_activa"] = "huanuco_completa"

    st.markdown("#### 🛰️ Consola de Misión Satelital — Selección de Escena Multitemporal Real")
    st.caption(
        "Catálogo operativo de escenas satelitales multitemporales **100% reales** (Copernicus Sentinel-2 L2A y ESA OSCD). "
        "Selecciona una misión satelital para cargar sus bandas espectrales (B04 Red, B08 NIR, B11 SWIR) "
        "y procesarlas con la red neuronal U-Net:"
    )

    opciones_claves = list(CATALOGO_ESCENAS_REALES.keys())
    clave_guardada = st.session_state.get("escena_satelital_activa", "huanuco_completa")
    idx_def = opciones_claves.index(clave_guardada) if clave_guardada in opciones_claves else 0

    col_sel, col_status = st.columns([2.4, 1.0])
    with col_sel:
        clave_activa = st.selectbox(
            "🛰️ Misión Satelital y Sector Monitoreado:",
            options=opciones_claves,
            index=idx_def,
            format_func=lambda k: CATALOGO_ESCENAS_REALES[k]["titulo"],
            key="select_escena_real",
            help="Todas las opciones corresponden a pasadas orbitales Sentinel-2 reales calibradas a reflectancia de superficie (L2A).",
        )
        if clave_activa != st.session_state.get("escena_satelital_activa"):
            st.session_state["escena_satelital_activa"] = clave_activa
            st.session_state["ultima_inferencia"] = True

    info_escena = CATALOGO_ESCENAS_REALES[clave_activa]
    tensor_5c, meta_parche, nombre_sector = cargar_escena_satelital_seleccionada(clave_activa)
    sector_idx = clave_activa

    with col_status:
        st.write("")
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 8px; padding: 10px; text-align: center;">
                <span style="font-weight: 700; color: {info_escena['color']}; font-size: 13px;">{info_escena['badge']}</span>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 3px;">📍 {info_escena['coords']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Panel de Telemetría Orbital y Previsualización Espectral Multitemporal
    with st.container():
        c_telem, c_prev0, c_prev1 = st.columns([1.3, 1.35, 1.35])
        with c_telem:
            st.markdown(
                f"""
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #10b981; font-weight: 700; margin-bottom: 6px;">🛰️ Telemetría de Misión</div>
                    <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;"><b>Plataforma:</b> {info_escena['satelite']}</div>
                    <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;"><b>Resolución:</b> {info_escena['resolucion']} (Bandas B4, B8, B11)</div>
                    <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;"><b>Temporalidad:</b> {info_escena['fechas']}</div>
                    <div style="font-size: 12px; color: #e2e8f0; margin-bottom: 4px;"><b>Ámbito:</b> {info_escena['cobertura']}</div>
                    <div style="font-size: 11px; color: #38bdf8; font-family: monospace; margin-top: 6px;">Matriz Ráster: {tensor_5c.shape[1]}×{tensor_5c.shape[2]} px · 6 canales</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_prev0:
            thumb_t0 = matriz_a_imagen_rgb(tensor_5c[1], tensor_5c[0], tensor_5c[2])
            st.image(thumb_t0, caption=f"Línea Base T0 — Falso Color ({info_escena['fechas'].split(' vs ')[0]})", width="stretch")
        with c_prev1:
            thumb_t1 = matriz_a_imagen_rgb(tensor_5c[4], tensor_5c[3], tensor_5c[5])
            st.image(thumb_t1, caption=f"Adquisición Reciente T1 — Falso Color ({info_escena['fechas'].split(' vs ')[-1]})", width="stretch")

    # Parámetros operativos integrados en línea
    col_p1, col_p2, col_p3 = st.columns([1.5, 1.2, 1.3])
    with col_p1:
        st.markdown(f"**Escena en Proceso:** `{info_escena['titulo']}`")
        st.caption(
            f"Origen: **{info_escena['satelite']}** • Matriz: **{tensor_5c.shape[1]}×{tensor_5c.shape[2]} px** • "
            f"Res: **{info_escena['resolucion']}** • Tensor 100% Satelital Real"
        )
    with col_p2:
        modo_umbral = st.radio(
            "Modo de Umbralización:",
            options=["Adaptativo (Otsu)", "Manual"],
            horizontal=True,
            key="radio_modo_umbral",
            help="Adaptativo calcula matemáticamente el corte óptimo bimodal entre fondo y anomalía.",
        )
        if modo_umbral == "Manual":
            umbral_decision = st.slider(
                "Umbral de Decisión (Threshold)",
                min_value=0.40,
                max_value=0.85,
                value=0.68,
                step=0.01,
                help="Probabilidad mínima requerida por píxel para clasificarlo como pérdida de cobertura o quema.",
            )
        else:
            umbral_decision = None

    with col_p3:
        modo_ejecucion = st.pills(
            "Estrategia de Ventana:",
            options=["Directa (256x256)", "Ventana Deslizante (Gaussian Blend)"],
            default="Directa (256x256)",
            key="modo_ejecucion_pills",
        )
        if not modo_ejecucion:
            modo_ejecucion = "Directa (256x256)"

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            aplicar_filtro_bio = st.checkbox(
                "Filtro biofísico (ΔNBR / ΔNDVI)",
                value=True,
                help="Descarta falsos positivos requiriendo caída de humedad o dosel vegetal real.",
            )
        with c_opt2:
            reescalar_score = st.checkbox(
                "Reescalar score continuo (0-100%)",
                value=True,
                help="Reescala el mapa continuo a 0–100 % como score relativo. No es calibración estadística ni una probabilidad de acierto.",
            )

    # Botón principal de ejecución
    btn_ejecutar = st.button(
        "⚡ Ejecutar Inferencia U-Net en Tiempo Real",
        type="primary",
        key="btn_lab_infer_exec",
        width="stretch",
    )

    if btn_ejecutar or "ultima_inferencia" in st.session_state:
        st.session_state["ultima_inferencia"] = True

        try:
            # 1. Cargar predictor ONNX
            predictor = ONNXForestPredictor(str(modelo_onnx_path))

            # 2. Inferencia y cronometraje de alta precisión
            t_inicio = time.perf_counter()

            if "Ventana Deslizante" in modo_ejecucion:
                prob_map = predictor.predict_sliding_window(
                    image=tensor_5c,
                    patch_size=128,
                    stride=64,
                    blend_mode="gaussian",
                )
            else:
                prob_raw = predictor.predict_proba(tensor_5c)
                prob_map = np.squeeze(prob_raw)

            latencia_ms = (time.perf_counter() - t_inicio) * 1000.0

            # Determinación de umbral efectivo
            if modo_umbral == "Adaptativo (Otsu)" or umbral_decision is None:
                umbral_efectivo = compute_otsu_threshold(prob_map)
                etiqueta_umbral = f"{umbral_efectivo:.3f} (Otsu)"
            else:
                umbral_efectivo = float(umbral_decision)
                etiqueta_umbral = f"{umbral_efectivo:.3f} (Manual)"

            # Escala continua (sin calibrar)
            if reescalar_score:
                prob_eval = calibrate_probabilities(prob_map)
            else:
                prob_eval = prob_map

            # 3. Post-procesamiento
            mascara_binaria = binarize(prob_map, threshold=umbral_efectivo)
            px_brutos = int(np.count_nonzero(mascara_binaria))

            # Filtro biofísico multitemporal
            if aplicar_filtro_bio and tensor_5c.shape[0] >= 6:
                mascara_binaria = apply_biophysical_filter(
                    mask=mascara_binaria,
                    tensor_6ch=tensor_5c,
                    min_dnbr=0.15,
                    min_dndvi=0.10,
                )
            px_refinados = int(np.count_nonzero(mascara_binaria))
            descarte_bio_ha = ((px_brutos - px_refinados) * (meta_parche["pixel_size_m"] ** 2)) / 10000.0

            area_afectada_ha = calculate_affected_area(mascara_binaria, pixel_size_m=meta_parche["pixel_size_m"])

            # 4. Extracción vectorial de alertas
            alertas_vectoriales = extract_alerts(
                mask=mascara_binaria,
                prob_map=prob_map,
                calibrated_prob_map=prob_eval if reescalar_score else None,
                transform=meta_parche["transform"],
                crs=CRS.from_epsg(32718),
                min_area_ha=0.05,
            )

            # 5. Panel de Métricas de Inferencia
            st.success(
                f"✅ **Inferencia ejecutada en {latencia_ms:.1f} ms** usando ONNX Runtime sobre CPU. "
                f"Modelo: `{modelo_onnx_path.name}` ({modelo_onnx_path.stat().st_size / (1024 * 1024):.1f} MB) "
                f"— Umbral aplicado: **{etiqueta_umbral}**."
            )

            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric(
                "Latencia Inferencia CPU",
                f"{latencia_ms:.1f} ms",
                f"Matriz {tensor_5c.shape[1]}×{tensor_5c.shape[2]} px",
                delta_color="normal",
            )
            kpi_c2.metric(
                "Superficie Afectada",
                f"{area_afectada_ha:.2f} ha",
                f"Umbral: {etiqueta_umbral}",
                delta_color="off",
            )
            delta_filtro = (
                f"Filtro biofísico: -{descarte_bio_ha:.2f} ha"
                if aplicar_filtro_bio and descarte_bio_ha > 0
                else "Filtro activo (0 falsos)"
            )
            kpi_c3.metric(
                "Alertas Poligonales",
                f"{len(alertas_vectoriales)}",
                delta_filtro,
                delta_color="inverse" if len(alertas_vectoriales) > 0 else "off",
            )
            if alertas_vectoriales:
                conf_media_patch = max(a.confidence_score * 100.0 for a in alertas_vectoriales)
                sub_label = f"Máx: {conf_media_patch:.1f}% · heurístico, sin calibrar"
            elif np.any(mascara_binaria == 1):
                vals_det = prob_eval[mascara_binaria == 1]
                conf_media_patch = float((0.65 * np.percentile(vals_det, 90) + 0.35 * np.max(vals_det)) * 100.0)
                sub_label = "heurístico, sin calibrar (0-100%)"
            else:
                conf_media_patch = 0.0
                sub_label = "Sin anomalías"

            kpi_c4.metric(
                "Score heurístico (sin calibrar)",
                f"{conf_media_patch:.1f}%",
                sub_label,
                delta_color="normal" if conf_media_patch >= 85 else "off",
            )

            divisor_bosque()

            # 6. Cuadrícula de Visualización Espectral y de IA
            st.markdown(f"#### 🛰️ Descomposición Espectral y Diagnóstico de la Red — *{nombre_sector}*")

            img_falso_color = matriz_a_imagen_rgb(tensor_5c[4], tensor_5c[3], tensor_5c[5])
            ndvi_val = np.clip((tensor_5c[4] - tensor_5c[3]) / (tensor_5c[4] + tensor_5c[3] + 1e-6), -1.0, 1.0)
            img_ndvi = matriz_a_imagen_ndvi(ndvi_val)
            img_heat = matriz_a_imagen_heatmap(prob_eval)
            img_mask = matriz_a_imagen_mascara(mascara_binaria)

            v_col1, v_col2, v_col3, v_col4 = st.columns(4)

            with v_col1:
                st.caption("**1. Color Falso Infrarrojo** (B8-B4-B11)")
                st.image(img_falso_color, use_container_width=True)
                st.caption("Vegetación sana en rojo/verde brillante; suelo expuesto en tonos oscuros.")

            with v_col2:
                st.caption("**2. Índice Espectral NDVI**")
                st.image(img_ndvi, use_container_width=True)
                st.caption("Verde = vigor fotosintético alto; rojo = estrés o pérdida de dosel.")

            with v_col3:
                st.caption(f"**3. Heatmap de {'Score Continuo (sin calibrar)' if reescalar_score else 'Probabilidad U-Net (sin calibrar)'}**")
                st.image(img_heat, use_container_width=True)
                st.caption("Gradiente continuo [0.0 a 1.0] con normalización de fondo dinámico (sin calibración probabilística).")

            with v_col4:
                st.caption(f"**4. Máscara sobre Umbral** (>= {etiqueta_umbral})")
                st.image(img_mask, use_container_width=True)
                st.caption("Rojo = polígonos validados por la IA y evidencia biofísica.")

            # 7. Tabla pericial de polígonos extraídos
            if alertas_vectoriales:
                st.divider()
                st.markdown("#### 📋 Registro Pericial de Alertas Extraídas")

                registros_tabla = []
                features_geojson = []

                for i, al in enumerate(alertas_vectoriales, start=1):
                    registros_tabla.append({
                        "Alerta": f"ALERTA-IA-{i:03d}",
                        "Superficie (ha)": f"{al.area_ha:.2f}",
                        "Score heurístico (sin calibrar)": f"{al.confidence_score * 100:.1f}%",
                        "Nivel de Severidad": al.confidence_level.value,
                        "Centroide UTM X": f"{al.centroid[0]:.1f}",
                        "Centroide UTM Y": f"{al.centroid[1]:.1f}",
                    })
                    features_geojson.append({
                        "type": "Feature",
                        "geometry": al.geometry,
                        "properties": al.to_dict(),
                    })

                df_vectorial = pd.DataFrame(registros_tabla)
                st.dataframe(df_vectorial, width="stretch", hide_index=True)

                # Descarga de GeoJSON
                geojson_str = json.dumps({
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32718"}},
                    "features": features_geojson,
                }, indent=2)

                st.download_button(
                    "🗺️ Descargar Capa de Alertas en GeoJSON (QGIS / ArcGIS)",
                    data=geojson_str,
                    file_name=f"alertas_unet_{sector_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                    mime="application/geo+json",
                    key="lab_dl_geojson",
                    width="stretch",
                )

        except Exception as exc:
            st.error(f"Error al ejecutar la inferencia ONNX: {exc}")
