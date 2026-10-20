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

from app.services.onnx_demo import (
    generar_parche_demostracion,
    matriz_a_imagen_heatmap,
    matriz_a_imagen_mascara,
    matriz_a_imagen_ndvi,
    matriz_a_imagen_rgb,
)
from app.styles.theme import divisor_bosque
from src.infer import (
    ONNXForestPredictor,
    binarize,
    calculate_affected_area,
    extract_alerts,
)


def render_tab_laboratorio_inferencia() -> None:
    """Pestaña interactiva de ejecución real de inferencia U-Net sobre CPU."""
    st.subheader(
        "🔬 Laboratorio de Inferencia U-Net — Ejecución Real en CPU (ONNX Runtime)",
        icon=":material/biotech:",
    )
    st.caption(
        "Ejecución en tiempo real del modelo de aprendizaje profundo sobre parches multiespectrales "
        "Sentinel-2 (10m) en los ecosistemas de Huánuco. Mide latencia de CPU, extrae polígonos vectoriales "
        "y calcula superficies afectadas con base matemática formal."
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

    if "sector_idx_lab" not in st.session_state:
        st.session_state["sector_idx_lab"] = 0

    st.markdown("#### 🛰️ Imágenes Satelitales Sentinel-2 en Huánuco (Selección Visual Directa)")
    st.caption(
        "Escenas satelitales Sentinel-2 L2A multiespectrales (10m/px) integradas directamente en la plataforma. "
        "Haz clic en cualquiera de las imágenes para cargarla y evaluarla con la red neuronal U-Net:"
    )

    sectores_cards = [
        {
            "id": 0,
            "titulo": "1. Carpish (Chinchao)",
            "subtitulo": "Bosque Nuboso / Quebrada",
            "evento": "Pérdida de cobertura (Tala)",
            "coords": "9°42'54\"S, 76°05'42\"W",
            "badge": "🚨 Tala no autorizada",
            "color": "#ef4444",
        },
        {
            "id": 1,
            "titulo": "2. Huallaga (Acomayo)",
            "subtitulo": "Valle Interandino",
            "evento": "Quema agrícola de rastrojos",
            "coords": "9°48'00\"S, 76°03'36\"W",
            "badge": "🔥 Quema agrícola",
            "color": "#f97316",
        },
        {
            "id": 2,
            "titulo": "3. Churubamba (Cuenca Media)",
            "subtitulo": "Laderas Andinas",
            "evento": "Tala selectiva y aclareo",
            "coords": "9°45'18\"S, 76°08'24\"W",
            "badge": "⚠️ Tala selectiva",
            "color": "#eab308",
        },
    ]

    cols_img = st.columns(3)
    for sec, col in zip(sectores_cards, cols_img):
        idx = sec["id"]
        esta_activo = (st.session_state["sector_idx_lab"] == idx)
        tensor_tmp, _, _ = generar_parche_demostracion(idx)
        thumb_rgb = matriz_a_imagen_rgb(tensor_tmp[1], tensor_tmp[0], tensor_tmp[2])

        with col:
            border_style = "2px solid #2ca25f" if esta_activo else "1px solid rgba(255, 255, 255, 0.12)"
            bg_style = "rgba(44, 162, 95, 0.14)" if esta_activo else "rgba(255, 255, 255, 0.02)"
            st.markdown(
                f"""
                <div style="border: {border_style}; background: {bg_style}; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:13px; color:#e2e8f0;">{sec['titulo']}</span>
                        <span style="font-size:10px; padding:2px 6px; border-radius:4px; background:rgba(0,0,0,0.4); color:{sec['color']}; font-weight:600;">{sec['badge']}</span>
                    </div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:2px;">{sec['subtitulo']}</div>
                    <div style="font-size:10px; color:#64748b; font-family:monospace; margin-top:2px;">📍 {sec['coords']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(thumb_rgb, caption=f"Falso Color S2 (B8/B4/B11) — {sec['titulo']}", use_container_width=True)

            if esta_activo:
                st.button(
                    "✅ Escena Satelital Activa",
                    key=f"btn_sec_activo_{idx}",
                    type="primary",
                    disabled=True,
                    use_container_width=True,
                )
            else:
                if st.button(
                    "🛰️ Cargar esta Escena",
                    key=f"btn_sec_sel_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["sector_idx_lab"] = idx
                    st.session_state["ultima_inferencia"] = True
                    st.rerun()

    sector_idx = st.session_state["sector_idx_lab"]
    tensor_5c, meta_parche, nombre_sector = generar_parche_demostracion(sector_idx)

    # Parámetros operativos integrados en línea
    col_p1, col_p2, col_p3 = st.columns([1.5, 1.2, 1.3])
    with col_p1:
        st.markdown(f"**Escena en Proceso:** `{sectores_cards[sector_idx]['titulo']}`")
        st.caption("Resolución: **10 m/px** • Matriz: **256x256** • Sensor: **Sentinel-2 L2A**")
    with col_p2:
        umbral_decision = st.slider(
            "Umbral de Decisión (Threshold)",
            min_value=0.25,
            max_value=0.85,
            value=0.50,
            step=0.05,
            help="Probabilidad mínima requerida por píxel para clasificarlo como pérdida de cobertura o quema.",
        )
    with col_p3:
        modo_ejecucion = st.pills(
            "Estrategia de Ventana:",
            options=["Directa (256x256)", "Ventana Deslizante (Gaussian Blend)"],
            default="Directa (256x256)",
            key="modo_ejecucion_pills",
        )
        if not modo_ejecucion:
            modo_ejecucion = "Directa (256x256)"

    # Botón principal de ejecución
    btn_ejecutar = st.button(
        "⚡ Ejecutar Inferencia U-Net en Tiempo Real",
        type="primary",
        use_container_width=True,
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
                prob_map = prob_raw[0, 0] if prob_raw.ndim == 4 else prob_raw[0]

            latencia_ms = (time.perf_counter() - t_inicio) * 1000.0

            # 3. Post-procesamiento
            mascara_binaria = binarize(prob_map, threshold=umbral_decision)
            area_afectada_ha = calculate_affected_area(mascara_binaria, pixel_size_m=meta_parche["pixel_size_m"])

            # 4. Extracción vectorial de alertas
            alertas_vectoriales = extract_alerts(
                mask=mascara_binaria,
                prob_map=prob_map,
                transform=meta_parche["transform"],
                crs=CRS.from_epsg(32718),
                min_area_ha=0.05,
            )

            # 5. Panel de Métricas de Inferencia
            st.success(
                f"✅ **Inferencia completada con éxito en {latencia_ms:.1f} ms** usando ONNX Runtime sobre CPU. "
                f"Modelo: `{modelo_onnx_path.name}` ({modelo_onnx_path.stat().st_size / (1024 * 1024):.1f} MB)."
            )

            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
            kpi_c1.metric(
                "Latencia Inferencia CPU",
                f"{latencia_ms:.1f} ms",
                "Tile 256x256 px",
                delta_color="normal",
            )
            kpi_c2.metric(
                "Superficie Afectada",
                f"{area_afectada_ha:.2f} ha",
                f"Umbral: >= {umbral_decision:.2f}",
                delta_color="off",
            )
            kpi_c3.metric(
                "Alertas Poligonales",
                f"{len(alertas_vectoriales)}",
                "Polígonos segmentados",
                delta_color="inverse" if len(alertas_vectoriales) > 0 else "off",
            )
            conf_media_patch = (
                float(np.mean(prob_map[mascara_binaria == 1]) * 100.0)
                if np.any(mascara_binaria == 1)
                else 0.0
            )
            kpi_c4.metric(
                "Confianza Promedio",
                f"{conf_media_patch:.1f}%",
                "Sobre píxeles detectados",
                delta_color="normal" if conf_media_patch >= 80 else "off",
            )

            divisor_bosque()

            # 6. Cuadrícula de Visualización Espectral y de IA
            st.markdown(f"#### 🛰️ Descomposición Espectral y Diagnóstico de la Red — *{nombre_sector}*")

            img_falso_color = matriz_a_imagen_rgb(tensor_5c[1], tensor_5c[0], tensor_5c[2])
            img_ndvi = matriz_a_imagen_ndvi(tensor_5c[3])
            img_heat = matriz_a_imagen_heatmap(prob_map)
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
                st.caption("**3. Heatmap de Probabilidad U-Net**")
                st.image(img_heat, use_container_width=True)
                st.caption("Gradiente continuo [0.0 a 1.0] generado por la última capa Sigmoid.")

            with v_col4:
                st.caption(f"**4. Máscara de Detección** (>= {umbral_decision:.2f})")
                st.image(img_mask, use_container_width=True)
                st.caption("Rojo = Polígonos de alerta que superan el umbral operativo.")

            # 7. Tabla pericial de polígonos extraídos
            if alertas_vectoriales:
                st.divider()
                st.markdown("#### 📋 Registro Pericial de Alertas Vectoriales Extraídas")

                registros_tabla = []
                features_geojson = []

                for i, al in enumerate(alertas_vectoriales, start=1):
                    registros_tabla.append({
                        "Alerta": f"ALERTA-IA-{i:03d}",
                        "Superficie (ha)": f"{al.area_ha:.2f}",
                        "Confianza Promedio": f"{al.confidence_score * 100:.1f}%",
                        "Nivel Severidad": al.confidence_level.value,
                        "Centroide UTM X": f"{al.centroid[0]:.1f}",
                        "Centroide UTM Y": f"{al.centroid[1]:.1f}",
                    })
                    features_geojson.append({
                        "type": "Feature",
                        "geometry": al.geometry,
                        "properties": al.to_dict(),
                    })

                df_vectorial = pd.DataFrame(registros_tabla)
                st.dataframe(df_vectorial, use_container_width=True, hide_index=True)

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
                )

        except Exception as exc:
            st.error(f"Error al ejecutar la inferencia ONNX: {exc}")
