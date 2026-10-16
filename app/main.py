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

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoForest-AI Huánuco | Monitoreo Forestal",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constantes geográficas y zonas de estudio (Huánuco, Perú)
# ---------------------------------------------------------------------------
ZONAS_ESTUDIO = {
    "Todas las zonas": {
        "centro": [-9.6500, -76.0800],
        "zoom": 10,
        "descripcion": (
            "Vista general del corredor ecológico entre el Valle del Huallaga "
            "y el Bosque Montano de Carpish en el departamento de Huánuco."
        ),
    },
    "Bosque de Carpish": {
        "centro": [-9.7080, -76.0920],
        "zoom": 12,
        "descripcion": (
            "Ecosistema de bosque nuboso montano de alta biodiversidad, "
            "clave para la regulación hídrica entre Huánuco y Tingo María."
        ),
        "poligono": [
            [-9.6550, -76.1500],
            [-9.6550, -76.0300],
            [-9.7600, -76.0300],
            [-9.7600, -76.1500],
        ],
        "color": "#1b7837",
    },
    "Valle del Huallaga": {
        "centro": [-9.5850, -76.0450],
        "zoom": 11,
        "descripcion": (
            "Corredor agrícola y forestal del río Huallaga (Chinchao, Acomayo, "
            "Churubamba y zonas de transición hacia selva alta)."
        ),
        "poligono": [
            [-9.4900, -76.1200],
            [-9.4900, -75.9600],
            [-9.6500, -75.9600],
            [-9.6500, -76.1200],
        ],
        "color": "#2166ac",
    },
}

COLORES_SEVERIDAD = {
    "Alta": "#d73027",
    "Media": "#fc8d59",
    "Baja": "#fee08b",
}


# ---------------------------------------------------------------------------
# Generación de datos demostrativos (simulados)
# ---------------------------------------------------------------------------
@st.cache_data
def obtener_alertas_simuladas() -> pd.DataFrame:
    """Devuelve un conjunto de alertas demostrativas etiquetadas como simuladas."""
    registros = [
        {
            "id_alerta": "SIM-HCO-001",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-10",
            "zona": "Bosque de Carpish",
            "sector": "Sector Túnel de Carpish Norte",
            "distrito": "Chinchao",
            "tipo_evento": "Deforestación",
            "severidad": "Alta",
            "area_ha": 4.8,
            "confianza_pct": 91.5,
            "ndvi_cambio": -0.42,
            "lat": -9.6985,
            "lon": -76.0875,
            "delta_lat": 0.0055,
            "delta_lon": 0.0065,
            "estado": "Alerta activa (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-002",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-11",
            "zona": "Bosque de Carpish",
            "sector": "Cabecera San Pedro de Carpish",
            "distrito": "Chinchao",
            "tipo_evento": "Degradación de bosque nuboso",
            "severidad": "Media",
            "area_ha": 2.3,
            "confianza_pct": 84.0,
            "ndvi_cambio": -0.27,
            "lat": -9.7210,
            "lon": -76.1120,
            "delta_lat": 0.0042,
            "delta_lon": 0.0048,
            "estado": "En verificación (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-003",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-12",
            "zona": "Bosque de Carpish",
            "sector": "Quebrada Sariapampa",
            "distrito": "Chinchao",
            "tipo_evento": "Quema agrícola",
            "severidad": "Alta",
            "area_ha": 5.6,
            "confianza_pct": 94.2,
            "ndvi_cambio": -0.49,
            "lat": -9.6790,
            "lon": -76.0640,
            "delta_lat": 0.0060,
            "delta_lon": 0.0068,
            "estado": "Alerta activa (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-004",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-12",
            "zona": "Bosque de Carpish",
            "sector": "Ladera Oriental Carpish",
            "distrito": "Chinchao",
            "tipo_evento": "Degradación de bosque nuboso",
            "severidad": "Baja",
            "area_ha": 1.4,
            "confianza_pct": 76.8,
            "ndvi_cambio": -0.18,
            "lat": -9.7380,
            "lon": -76.0710,
            "delta_lat": 0.0035,
            "delta_lon": 0.0040,
            "estado": "Monitoreo preventivo (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-005",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-13",
            "zona": "Valle del Huallaga",
            "sector": "Margen Izquierda Acomayo",
            "distrito": "Chinchao",
            "tipo_evento": "Quema agrícola",
            "severidad": "Alta",
            "area_ha": 6.2,
            "confianza_pct": 92.8,
            "ndvi_cambio": -0.51,
            "lat": -9.5920,
            "lon": -76.0580,
            "delta_lat": 0.0065,
            "delta_lon": 0.0072,
            "estado": "Alerta activa (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-006",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-14",
            "zona": "Valle del Huallaga",
            "sector": "Microcuenca Chinobamba",
            "distrito": "Churubamba",
            "tipo_evento": "Deforestación",
            "severidad": "Media",
            "area_ha": 3.1,
            "confianza_pct": 85.4,
            "ndvi_cambio": -0.33,
            "lat": -9.6240,
            "lon": -76.0910,
            "delta_lat": 0.0048,
            "delta_lon": 0.0052,
            "estado": "En verificación (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-007",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-15",
            "zona": "Valle del Huallaga",
            "sector": "Puente Durand - Huallaga Medio",
            "distrito": "Chinchao",
            "tipo_evento": "Quema agrícola",
            "severidad": "Media",
            "area_ha": 2.9,
            "confianza_pct": 81.9,
            "ndvi_cambio": -0.29,
            "lat": -9.5380,
            "lon": -76.0120,
            "delta_lat": 0.0045,
            "delta_lon": 0.0050,
            "estado": "En verificación (Demo)",
        },
        {
            "id_alerta": "SIM-HCO-008",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-16",
            "zona": "Valle del Huallaga",
            "sector": "Ladera Tambogán",
            "distrito": "Churubamba",
            "tipo_evento": "Quema agrícola",
            "severidad": "Baja",
            "area_ha": 1.2,
            "confianza_pct": 74.5,
            "ndvi_cambio": -0.16,
            "lat": -9.5690,
            "lon": -76.0820,
            "delta_lat": 0.0032,
            "delta_lon": 0.0038,
            "estado": "Monitoreo preventivo (Demo)",
        },
    ]
    return pd.DataFrame(registros)


def construir_poligono_alerta(lat: float, lon: float, dlat: float, dlon: float) -> list[list[float]]:
    """Genera un polígono rectangular irregular simulado alrededor del centroide."""
    return [
        [lat + dlat, lon - dlon * 0.8],
        [lat + dlat * 0.7, lon + dlon],
        [lat - dlat, lon + dlon * 0.9],
        [lat - dlat * 0.8, lon - dlon],
    ]


# ---------------------------------------------------------------------------
# Construcción del mapa Folium
# ---------------------------------------------------------------------------
def crear_mapa_monitoreo(
    df_filtrado: pd.DataFrame,
    zona_seleccionada: str,
    mapa_base: str,
    mostrar_zonas: bool,
    mostrar_poligonos_alerta: bool,
    mostrar_marcadores: bool,
) -> folium.Map:
    """Construye el mapa interactivo con zonas de estudio y alertas simuladas."""
    info_zona = ZONAS_ESTUDIO[zona_seleccionada]
    mapa = folium.Map(
        location=info_zona["centro"],
        zoom_start=info_zona["zoom"],
        tiles=None,
        control_scale=True,
    )

    # Capas base
    capas_tiles = {
        "OpenStreetMap": {
            "tiles": "OpenStreetMap",
            "attr": "© OpenStreetMap contributors",
        },
        "Satélite (Esri World Imagery)": {
            "tiles": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            "attr": "Tiles © Esri — Source: Esri, Copernicus, USGS",
        },
        "Topográfico (OpenTopoMap)": {
            "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap",
        },
    }

    # Añadir la capa seleccionada primero para que sea la activa por defecto
    orden_capas = [mapa_base] + [k for k in capas_tiles if k != mapa_base]
    for nombre_capa in orden_capas:
        cfg = capas_tiles[nombre_capa]
        folium.TileLayer(
            tiles=cfg["tiles"],
            attr=cfg["attr"],
            name=nombre_capa,
            overlay=False,
            control=True,
            show=(nombre_capa == mapa_base),
        ).add_to(mapa)

    # 1. Capa de polígonos de zonas de estudio
    if mostrar_zonas:
        grupo_zonas = folium.FeatureGroup(name="Zonas de estudio (Referencia)", show=True)
        for nombre_z, datos_z in ZONAS_ESTUDIO.items():
            if "poligono" not in datos_z:
                continue
            folium.Polygon(
                locations=datos_z["poligono"],
                color=datos_z["color"],
                weight=2,
                dash_array="6, 6",
                fill=True,
                fill_color=datos_z["color"],
                fill_opacity=0.08,
                tooltip=f"Zona de estudio: {nombre_z}",
                popup=folium.Popup(
                    f"<b>{nombre_z}</b><br>{datos_z['descripcion']}<br>"
                    "<i>Delimitación referencial de monitoreo</i>",
                    max_width=280,
                ),
            ).add_to(grupo_zonas)
        grupo_zonas.add_to(mapa)

    # 2. Capa de polígonos de áreas de alerta simulada
    if mostrar_poligonos_alerta and not df_filtrado.empty:
        grupo_poligonos = folium.FeatureGroup(name="Áreas de alerta [SIMULADAS]", show=True)
        for _, fila in df_filtrado.iterrows():
            color_sev = COLORES_SEVERIDAD.get(fila["severidad"], "#e31a1c")
            coords_poligono = construir_poligono_alerta(
                fila["lat"], fila["lon"], fila["delta_lat"], fila["delta_lon"]
            )
            popup_html = (
                f"<div style='font-family: sans-serif; font-size: 13px;'>"
                f"<span style='background:#fef0d9; color:#b30000; padding:2px 6px; "
                f"border-radius:4px; font-weight:bold; font-size:11px;'>"
                f"DATO DEMOSTRATIVO SIMULADO</span><br><br>"
                f"<b>ID:</b> {fila['id_alerta']}<br>"
                f"<b>Evento:</b> {fila['tipo_evento']}<br>"
                f"<b>Zona:</b> {fila['zona']} ({fila['sector']})<br>"
                f"<b>Distrito:</b> {fila['distrito']}<br>"
                f"<b>Severidad:</b> {fila['severidad']}<br>"
                f"<b>Área estimada:</b> {fila['area_ha']:.1f} ha<br>"
                f"<b>Cambio NDVI:</b> {fila['ndvi_cambio']:.2f}<br>"
                f"<b>Confianza U-Net (sim):</b> {fila['confianza_pct']:.1f}%<br>"
                f"<b>Fecha:</b> {fila['fecha']}"
                f"</div>"
            )
            folium.Polygon(
                locations=coords_poligono,
                color=color_sev,
                weight=2.5,
                fill=True,
                fill_color=color_sev,
                fill_opacity=0.45,
                tooltip=(
                    f"[{fila['origen_dato']}] {fila['id_alerta']} — "
                    f"{fila['tipo_evento']} ({fila['area_ha']} ha)"
                ),
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(grupo_poligonos)
        grupo_poligonos.add_to(mapa)

    # 3. Capa de marcadores puntuales
    if mostrar_marcadores and not df_filtrado.empty:
        grupo_marcadores = folium.FeatureGroup(name="Centroides de alerta [SIMULADOS]", show=True)
        for _, fila in df_filtrado.iterrows():
            color_sev = COLORES_SEVERIDAD.get(fila["severidad"], "#e31a1c")
            folium.CircleMarker(
                location=[fila["lat"], fila["lon"]],
                radius=7,
                color="#1a1a1a",
                weight=1.5,
                fill=True,
                fill_color=color_sev,
                fill_opacity=0.9,
                tooltip=(
                    f"{fila['id_alerta']} | {fila['sector']} "
                    f"({fila['severidad']} - SIMULADO)"
                ),
            ).add_to(grupo_marcadores)
        grupo_marcadores.add_to(mapa)

    folium.LayerControl(collapsed=False).add_to(mapa)
    return mapa


# ---------------------------------------------------------------------------
# Interfaz principal de Streamlit
# ---------------------------------------------------------------------------
def main() -> None:
    df_alertas = obtener_alertas_simuladas()

    # --- Barra lateral (Filtros y configuración) ---
    with st.sidebar:
        st.header("🌲 GeoForest-AI Huánuco")
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
        mapa_base_sel = st.selectbox(
            "Mapa base",
            options=[
                "Satélite (Esri World Imagery)",
                "OpenStreetMap",
                "Topográfico (OpenTopoMap)",
            ],
            index=0,
        )
        mostrar_zonas = st.checkbox("Mostrar límites de zonas de estudio", value=True)
        mostrar_poligonos = st.checkbox("Mostrar polígonos de área afectada", value=True)
        mostrar_marcadores = st.checkbox("Mostrar centroides de alerta", value=True)

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

    # --- Encabezado principal ---
    st.title("🌲 GeoForest-AI Huánuco — Panel de Alerta Temprana")
    st.markdown(
        "Sistema experimental de detección de **pérdida de cobertura forestal** y "
        "**quemas agrícolas** en el **Bosque Montano de Carpish** y el **Valle del Huallaga** "
        "(Huánuco, Perú), diseñado para operar con imágenes **Sentinel-2** y "
        "referencias de **Geobosques (MINAM)**."
    )

    st.warning(
        "⚠️ **Aviso de Datos Demostrativos (Simulados):** "
        "Los registros, coordenadas e indicadores presentados a continuación tienen fines "
        "exclusivamente demostrativos (`origen_dato = SIMULADO`). En las siguientes fases "
        "se integrarán recortes reales de Sentinel-2, máscaras de Geobosques y el modelo U-Net en ONNX."
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

    st.divider()

    # --- Sección principal: Mapa interactivo y Resumen por Zona ---
    col_mapa, col_resumen = st.columns([2.2, 1])

    with col_mapa:
        st.subheader(f"🗺️ Mapa de Monitoreo — {zona_sel}")
        st.caption(ZONAS_ESTUDIO[zona_sel]["descripcion"])

        mapa_folium = crear_mapa_monitoreo(
            df_filtrado=df_filtrado,
            zona_seleccionada=zona_sel,
            mapa_base=mapa_base_sel,
            mostrar_zonas=mostrar_zonas,
            mostrar_poligonos_alerta=mostrar_poligonos,
            mostrar_marcadores=mostrar_marcadores,
        )
        st_folium(
            mapa_folium,
            width=None,
            height=520,
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
        st.subheader("📊 Distribución Simulada")
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

    st.divider()

    # --- Tabla detallada de alertas demostrativas ---
    st.subheader("📋 Registro Detallado de Alertas Demostrativas (Datos Simulados)")
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

    csv_bytes = df_vista.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar alertas simuladas (CSV)",
        data=csv_bytes,
        file_name="alertas_simuladas_geoforest_huanuco.csv",
        mime="text/csv",
    )

    st.divider()

    # --- Hoja de ruta del proyecto ---
    with st.expander("🛠️ Estado de Desarrollo y Orden de Trabajo del Proyecto", expanded=False):
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


if __name__ == "__main__":
    main()
