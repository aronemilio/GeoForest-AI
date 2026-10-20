"""Componente del mapa interactivo con Folium, EOX Sentinel-2 Cloudless y NASA FIRMS."""

from __future__ import annotations

import os
import folium
from folium import plugins
import pandas as pd

from app.config import COLORES_SEVERIDAD, ZONAS_ESTUDIO
from app.services.alerts_data import construir_poligono_alerta


def crear_mapa_monitoreo(
    df_filtrado: pd.DataFrame,
    zona_seleccionada: str,
    mapa_base: str = "Satélite real (Esri World Imagery)",
    mostrar_zonas: bool = True,
    mostrar_poligonos_alerta: bool = True,
    mostrar_marcadores: bool = True,
    mostrar_nasa_fires: bool = True,
    df_nasa_firms: pd.DataFrame | None = None,
    mostrar_sentinel_copernicus: bool = True,
) -> folium.Map:
    """Construye el mapa interactivo con vista nítida libre de nubes y alertas en primer plano."""
    info_zona = ZONAS_ESTUDIO.get(zona_seleccionada, ZONAS_ESTUDIO["Todas las zonas"])
    mapa = folium.Map(
        location=info_zona["centro"],
        zoom_start=info_zona["zoom"],
        tiles=None,
        control_scale=True,
    )

    # Inyección de estilos para capas vectoriales, selector flotante y pantalla completa
    mapa.get_root().html.add_child(
        folium.Element(
            """
            <style>
              .leaflet-pane.leaflet-overlay-pane {
                z-index: 650 !important;
              }
              .leaflet-pane.leaflet-marker-pane {
                z-index: 700 !important;
              }
              .leaflet-pane.leaflet-popup-pane {
                z-index: 800 !important;
              }
              /* Selector táctico de capas flotante */
              .leaflet-control-layers {
                background: rgba(14, 28, 22, 0.94) !important;
                border: 1px solid rgba(46, 125, 80, 0.65) !important;
                border-radius: 12px !important;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
                color: #e6f4ea !important;
                font-family: system-ui, -apple-system, sans-serif !important;
                font-size: 11.5px !important;
                padding: 10px 14px !important;
                backdrop-filter: blur(10px);
                max-width: 290px;
              }
              .leaflet-control-layers-expanded {
                max-height: 380px;
                overflow-y: auto;
              }
              .leaflet-control-layers-base label,
              .leaflet-control-layers-overlays label {
                margin-bottom: 5px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 7px;
                color: #d1e7dd;
                font-weight: 500;
              }
              .leaflet-control-layers-base label:hover,
              .leaflet-control-layers-overlays label:hover {
                color: #a7f3d0;
              }
              .leaflet-control-layers-separator {
                border-top: 1px solid rgba(46, 125, 80, 0.4) !important;
                margin: 8px 0 !important;
              }
              /* Botón de pantalla completa y zoom */
              .leaflet-control-fullscreen a,
              .leaflet-control-zoom a {
                background: rgba(14, 28, 22, 0.90) !important;
                border: 1px solid rgba(46, 125, 80, 0.5) !important;
                color: #e6f4ea !important;
                border-radius: 8px !important;
              }
              .leaflet-control-fullscreen a:hover,
              .leaflet-control-zoom a:hover {
                background: rgba(22, 45, 34, 0.98) !important;
                color: #a7f3d0 !important;
              }
            </style>
            """
        )
    )

    mapbox_token = os.environ.get("MAPBOX_TOKEN", "").strip()

    # Capas base ordenadas (Esri World Imagery como predeterminada nítida y libre de nubes)
    capas_tiles = {
        "Satélite real (Esri World Imagery)": {
            "tiles": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            "attr": "Tiles © Esri — Source: Esri, Copernicus, USGS",
            "subdomains": "abc",
        },
        "EOX Sentinel-2 Cloudless": {
            "tiles": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020/default/g/{z}/{y}/{x}.jpg",
            "attr": "Sentinel-2 cloudless - https://s2maps.eu by EOX IT Services GmbH (Contains modified Copernicus Sentinel data)",
            "subdomains": "abc",
        },
        "Táctico oscuro (CartoDB)": {
            "tiles": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            "attr": "© OpenStreetMap contributors © CARTO",
            "subdomains": "abcd",
        },
        "Toponimia OSM": {
            "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attr": "© OpenStreetMap contributors",
            "subdomains": "abc",
        },
        "Topográfico (OpenTopoMap)": {
            "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "attr": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap",
            "subdomains": "abc",
        },
    }

    if mapbox_token:
        capas_tiles["Mapbox Satélite HD"] = {
            "tiles": f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}",
            "attr": "© Mapbox © Maxar",
            "subdomains": "abc",
        }
        capas_tiles["Mapbox Táctico Nocturno"] = {
            "tiles": f"https://api.mapbox.com/styles/v1/mapbox/navigation-night-v1/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}",
            "attr": "© Mapbox © OpenStreetMap",
            "subdomains": "abc",
        }

    # Si mapa_base no coincide exactamente, usar Esri World Imagery como predeterminado
    if mapa_base not in capas_tiles:
        mapa_base = "Satélite real (Esri World Imagery)"

    # Añadir la capa seleccionada primero para que sea la activa por defecto
    orden_capas = [mapa_base] + [k for k in capas_tiles if k != mapa_base]
    for nombre_capa in orden_capas:
        if nombre_capa not in capas_tiles:
            continue
        cfg = capas_tiles[nombre_capa]
        folium.TileLayer(
            tiles=cfg["tiles"],
            attr=cfg["attr"],
            name=nombre_capa,
            overlay=False,
            control=True,
            show=(nombre_capa == mapa_base),
            subdomains=cfg.get("subdomains", "abc"),
        ).add_to(mapa)

    # 1. Capa WMS en vivo de NASA GIBS: Anomalías Térmicas / Incendios VIIRS 375m
    if mostrar_nasa_fires:
        folium.WmsTileLayer(
            url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
            layers="VIIRS_SNPP_Thermal_Anomalies_375m_All",
            fmt="image/png",
            transparent=True,
            name="🔥 NASA GIBS Focos Ráster (En Vivo)",
            overlay=True,
            control=True,
            show=True,
            opacity=0.85,
            attr="NASA Earthdata / GIBS",
        ).add_to(mapa)

        # Focos térmicos vectoriales de NASA FIRMS (API Key)
        if df_nasa_firms is not None and not df_nasa_firms.empty:
            grupo_firms = folium.FeatureGroup(name="🔥 Focos Reales NASA FIRMS (API Key)", show=True)
            for _, row in df_nasa_firms.iterrows():
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    brillo = float(row.get("bright_ti4", 310.0))
                    frp = float(row.get("frp", 1.0))
                    fecha = str(row.get("acq_date", "Reciente"))
                    hora = str(row.get("acq_time", "")).zfill(4)
                    color_fuego = "#ff2200" if brillo >= 325 else "#ff6600"

                    popup_fuego = (
                        f"<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.4;min-width:190px;'>"
                        f"<div style='background:#b30000;color:white;padding:2px 6px;border-radius:3px;font-weight:bold;font-size:10px;margin-bottom:6px;display:inline-block;'>"
                        f"🔥 NASA FIRMS · ANOMALÍA TÉRMICA REAL</div><br>"
                        f"<b>Satélite:</b> Suomi-NPP / VIIRS 375m<br>"
                        f"<b>Fecha:</b> {fecha} · {hora[:2]}:{hora[2:]} UTC<br>"
                        f"<b>Temperatura de Brillo:</b> {brillo:.1f} K ({brillo - 273.15:.1f} °C)<br>"
                        f"<b>Potencia FRP:</b> {frp:.2f} MW<br>"
                        f"<b>Coordenadas:</b> <code>{lat:.5f}, {lon:.5f}</code><br>"
                        f"<div style='margin-top:5px;font-size:10px;color:#888;'>Dato satelital en vivo de la NASA</div>"
                        f"</div>"
                    )

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=7,
                        color="#ffffff",
                        weight=1.5,
                        fill=True,
                        fill_color=color_fuego,
                        fill_opacity=0.9,
                        tooltip=f"🔥 Foco NASA VIIRS: {brillo:.1f} K ({fecha})",
                        popup=folium.Popup(popup_fuego, max_width=260),
                    ).add_to(grupo_firms)
                except Exception:
                    continue
            grupo_firms.add_to(mapa)

    # 2. Capa de polígonos de zonas de estudio (Delimitación referencial)
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

    # 3. Capa de polígonos de áreas de alerta (Primer plano con z-index superior)
    # Tala en rojo (#d73027) y quemas agrícolas en naranja (#ff6600)
    if mostrar_poligonos_alerta and not df_filtrado.empty:
        grupo_poligonos = folium.FeatureGroup(name="Áreas de alerta [Tala y Quemas]", show=True)
        for _, fila in df_filtrado.iterrows():
            tipo_ev = str(fila.get("tipo_evento", "")).lower()
            if "quema" in tipo_ev:
                color_poly = "#ff6600"  # Quemas agrícolas en naranja
            elif "deforest" in tipo_ev or "tala" in tipo_ev:
                color_poly = "#d73027"  # Tala / deforestación en rojo
            else:
                color_poly = COLORES_SEVERIDAD.get(fila["severidad"], "#d73027")

            coords_poligono = construir_poligono_alerta(
                fila["lat"], fila["lon"], fila["delta_lat"], fila["delta_lon"]
            )
            popup_html = (
                f"<div style='font-family: system-ui, -apple-system, sans-serif; font-size: 13px; line-height: 1.5; min-width: 220px;'>"
                f"<div style='background:{color_poly}; color:white; padding:3px 7px; "
                f"border-radius:4px; font-weight:bold; font-size:11px; margin-bottom:8px; display:inline-block;'>"
                f"{fila['tipo_evento'].upper()} — {fila['severidad'].upper()}</div><br>"
                f"<b>ID Alerta:</b> <code>{fila['id_alerta']}</code><br>"
                f"<b>Superficie estimada:</b> <span style='font-weight:bold; color:#1a1a1a;'>{fila['area_ha']:.2f} ha</span><br>"
                f"<b>Confianza algorítmica:</b> <span style='font-weight:bold; color:{color_poly};'>{fila['confianza_pct']:.1f}%</span><br>"
                f"<b>Anomalía ΔNDVI:</b> <code>{fila['ndvi_cambio']:.2f}</code><br>"
                f"<b>Sector / Distrito:</b> {fila['sector']}, {fila['distrito']}<br>"
                f"<b>Zona de estudio:</b> {fila['zona']}<br>"
                f"<b>Fecha de detección:</b> {fila['fecha']}<br>"
                f"<div style='margin-top:6px; font-size:11px; color:#555;'>Capa vectorial en primer plano · {fila.get('estado', '')}</div>"
                f"</div>"
            )
            tooltip_text = (
                f"🚨 [{fila['id_alerta']}] {fila['tipo_evento']} | "
                f"Área: {fila['area_ha']:.2f} ha | Confianza: {fila['confianza_pct']:.1f}%"
            )
            folium.Polygon(
                locations=coords_poligono,
                color=color_poly,
                weight=3,
                fill=True,
                fill_color=color_poly,
                fill_opacity=0.60,
                tooltip=tooltip_text,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(grupo_poligonos)
        grupo_poligonos.add_to(mapa)

    # 4. Capa de marcadores puntuales y centroides de severidad
    if mostrar_marcadores and not df_filtrado.empty:
        grupo_marcadores = folium.FeatureGroup(name="Centroides y Severidad", show=True)
        for _, fila in df_filtrado.iterrows():
            tipo_ev = str(fila.get("tipo_evento", "")).lower()
            if "quema" in tipo_ev:
                color_punto = "#ff6600"
            elif "deforest" in tipo_ev or "tala" in tipo_ev:
                color_punto = "#d73027"
            else:
                color_punto = COLORES_SEVERIDAD.get(fila["severidad"], "#d73027")

            folium.CircleMarker(
                location=[fila["lat"], fila["lon"]],
                radius=8,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color=color_punto,
                fill_opacity=0.95,
                tooltip=(
                    f"[{fila['id_alerta']}] {fila['sector']} "
                    f"({fila['tipo_evento']} - Severidad {fila['severidad']})"
                ),
            ).add_to(grupo_marcadores)
        grupo_marcadores.add_to(mapa)

    # 5. Herramienta de Pantalla Completa (Fullscreen)
    plugins.Fullscreen(
        position="topleft",
        title="Pantalla completa",
        title_cancel="Salir de pantalla completa",
        force_separate_button=True,
    ).add_to(mapa)

    # 6. Selector de capas flotante nativo en Leaflet
    # Permite alternar capas en milisegundos en el navegador sin recargas de Streamlit ni perder el zoom
    folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

    return mapa
