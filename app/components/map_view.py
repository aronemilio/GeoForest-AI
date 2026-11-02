"""Componente del mapa interactivo en tiempo real: pérdida de cobertura forestal y quemas activas."""

from __future__ import annotations

from typing import Any

import folium
from folium import plugins, raster_layers
import numpy as np
import pandas as pd

from app.config import COLORES_SEVERIDAD, ZONAS_ESTUDIO
from app.services.alerts_data import GEOBOSQUES_ATD_WMS_URL, construir_poligono_alerta


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
    centro_override: list[float] | None = None,
    zoom_override: int | None = None,
    poligono_destacado: list[list[float]] | None = None,
    etiqueta_destacada: str | None = None,
    secuencia_hitos: list[dict[str, Any]] | None = None,
    panel_capas_colapsado: bool = True,
    capas_raster_satelital: dict[str, Any] | None = None,
    overlays_sentinel2_reales: list[dict[str, Any]] | None = None,
    mostrar_wms_geobosques: bool = True,
    vista_satelital_activa: str = "t1",
    telemetria_tiempo_real: dict[str, Any] | None = None,
) -> folium.Map:
    """Construye el mapa interactivo en tiempo real de pérdida de cobertura forestal y quemas.

    Renderiza capas vectoriales oficiales (PNCB Geobosques ATD, NASA FIRMS NRT, SERFOR SAMI),
    servicios WMS en vivo (Geobosques 2026, NASA GIBS) y overlays multiespectrales Sentinel-2 L2A
    calibrados físicamente (ΔNDVI para pérdida de dosel y ΔNBR/SWIR para quemas agrícolas).

    Args:
        df_filtrado: DataFrame con alertas oficiales de pérdida de cobertura boscosa (ATD).
        zona_seleccionada: Nombre de la zona de estudio o "Todas las zonas".
        mapa_base: Nombre de la capa base ("Satélite real..." u "Toponimia OSM...").
        mostrar_zonas: Si es True, dibuja los polígonos delimitadores de las zonas de estudio.
        mostrar_poligonos_alerta: Si es True, dibuja los polígonos perimetrales de cada alerta ATD.
        mostrar_marcadores: Si es True, dibuja marcadores circulares en los centroides de alerta.
        mostrar_nasa_fires: Si es True, renderiza focos de calor FIRMS y alertas de incendio SERFOR.
        df_nasa_firms: DataFrame con telemetría de focos térmicos NRT activos.
        mostrar_sentinel_copernicus: Si es True, proyecta recortes e imágenes Sentinel-2 L2A.
        centro_override: Coordenadas [lat, lon] opcionales para centrar el mapa.
        zoom_override: Nivel de zoom entero opcional.
        poligono_destacado: Coordenadas de polígono para destacar un sector prioritario.
        etiqueta_destacada: Texto informativo para el polígono destacado.
        secuencia_hitos: Lista de hitos cronológicos para la visualización del frente de avance.
        panel_capas_colapsado: Si es True, colapsa el control de capas al inicio.
        capas_raster_satelital: Diccionario opcional con capas satelitales demostrativas (Modo Pitch).
        overlays_sentinel2_reales: Lista de recortes GeoTIFF reales procesados con LabelRule física.
        mostrar_wms_geobosques: Si es True, añade la capa WMS 2026 de PNCB Geobosques MINAM.
        vista_satelital_activa: Selector de capa espectral activa ('t1', 't0', 'mascara', 'ambas').
        telemetria_tiempo_real: Diccionario con métricas de sincronización y eventos en vivo para el HUD.

    Returns:
        Objeto `folium.Map` configurado y listo para renderizar en Streamlit.
    """
    info_zona = ZONAS_ESTUDIO.get(zona_seleccionada, ZONAS_ESTUDIO["Todas las zonas"])
    centro = centro_override if centro_override is not None else info_zona["centro"]
    zoom = zoom_override if zoom_override is not None else info_zona["zoom"]
    mapa = folium.Map(
        location=centro,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    # Estilos CSS para jerarquía de paneles, pulso térmico en vivo y HUD flotante
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
              .leaflet-control-layers {
                background: rgba(14, 28, 22, 0.94) !important;
                border: 1px solid rgba(46, 125, 80, 0.65) !important;
                border-radius: 8px !important;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55) !important;
                transition: all 0.25s ease-in-out !important;
              }
              .leaflet-control-layers:hover {
                border-color: #10b981 !important;
                box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35) !important;
              }
              .leaflet-control-layers-toggle {
                background-color: transparent !important;
                width: 38px !important;
                height: 38px !important;
                border-radius: 8px !important;
              }
              .leaflet-control-layers-expanded {
                background: rgba(10, 22, 17, 0.96) !important;
                border: 1px solid rgba(16, 185, 129, 0.7) !important;
                border-radius: 12px !important;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8) !important;
                color: #e6f4ea !important;
                font-family: system-ui, -apple-system, sans-serif !important;
                font-size: 11.5px !important;
                padding: 12px 16px !important;
                backdrop-filter: blur(12px);
                min-width: 260px;
                max-width: 340px;
                max-height: 430px;
                overflow-y: auto;
              }
              .leaflet-control-layers-base label,
              .leaflet-control-layers-overlays label {
                margin-bottom: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 8px;
                color: #d1e7dd;
                font-weight: 500;
                transition: color 0.15s ease;
              }
              .leaflet-control-layers-base label:hover,
              .leaflet-control-layers-overlays label:hover {
                color: #34d399 !important;
              }
              .leaflet-control-layers-separator {
                border-top: 1px solid rgba(46, 125, 80, 0.4) !important;
                margin: 9px 0 !important;
              }
              .leaflet-control-fullscreen a,
              .leaflet-control-zoom a {
                background: rgba(14, 28, 22, 0.90) !important;
                border: 1px solid rgba(46, 125, 80, 0.5) !important;
                color: #e6f4ea !important;
                border-radius: 8px !important;
                width: 34px !important;
                height: 34px !important;
                line-height: 34px !important;
                transition: all 0.2s ease !important;
              }
              .leaflet-control-fullscreen a:hover,
              .leaflet-control-zoom a:hover {
                background: rgba(22, 45, 34, 0.98) !important;
                color: #34d399 !important;
                border-color: #10b981 !important;
                transform: scale(1.04);
              }
              .gf-milestone-marker {
                white-space: nowrap;
                transition: transform 0.15s ease-out;
              }
              .gf-milestone-marker:hover {
                transform: scale(1.15) translateY(-2px);
                z-index: 9999 !important;
              }
              @keyframes gfPulseFire {
                0% { box-shadow: 0 0 0 0 rgba(255, 68, 0, 0.78); }
                70% { box-shadow: 0 0 0 11px rgba(255, 68, 0, 0); }
                100% { box-shadow: 0 0 0 0 rgba(255, 68, 0, 0); }
              }
              .gf-fire-dot {
                width: 14px;
                height: 14px;
                border-radius: 50%;
                border: 2px solid #ffffff;
                animation: gfPulseFire 1.9s infinite;
              }
              @keyframes gfLiveBlink {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.35; }
              }
              .gf-live-hud {
                position: fixed;
                bottom: 28px;
                left: 12px;
                z-index: 900;
                background: rgba(10, 22, 17, 0.92);
                border: 1px solid rgba(16, 185, 129, 0.55);
                border-radius: 8px;
                padding: 7px 11px;
                color: #e6f4ea;
                font-family: system-ui, -apple-system, sans-serif;
                font-size: 11px;
                line-height: 1.45;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
                pointer-events: none;
              }
            </style>
            """
        )
    )

    # Capas base
    capas_tiles = {
        "Satélite real (Esri World Imagery)": {
            "tiles": (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ),
            "attr": "Tiles © Esri — Source: Esri, Copernicus, USGS",
            "subdomains": "abc",
        },
        "Toponimia OSM (OpenStreetMap)": {
            "tiles": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "attr": "© OpenStreetMap contributors",
            "subdomains": "abc",
        },
    }

    if mapa_base == "Toponimia OSM":
        mapa_base = "Toponimia OSM (OpenStreetMap)"
    elif mapa_base not in capas_tiles:
        mapa_base = "Satélite real (Esri World Imagery)"

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

    # 0.4 Overlays espectrales Sentinel-2 L2A 100% reales (Tile 18LUQ: ΔNDVI pérdida cobertura y ΔNBR quema)
    if mostrar_sentinel_copernicus and overlays_sentinel2_reales:
        for ov in overlays_sentinel2_reales:
            bounds_ov = ov["bounds"]
            z_nom = ov.get("zona_titulo", ov.get("zona_slug", "Huánuco"))
            f_t0 = ov.get("fecha_t0", "T0")
            f_t1 = ov.get("fecha_t1", "T1")
            a_perd = float(ov.get("area_perdida_cobertura_ha", 0.0))
            a_quem = float(ov.get("area_quema_espectral_ha", 0.0))

            if ov.get("t0_rgba") is not None:
                raster_layers.ImageOverlay(
                    image=ov["t0_rgba"],
                    bounds=bounds_ov,
                    opacity=0.92,
                    name=f"🌲 Sentinel-2 L2A Real T0 [{z_nom}: {f_t0}]",
                    show=(vista_satelital_activa == "t0"),
                    zindex=602,
                ).add_to(mapa)

            if ov.get("t1_rgba") is not None:
                raster_layers.ImageOverlay(
                    image=ov["t1_rgba"],
                    bounds=bounds_ov,
                    opacity=0.93,
                    name=f"🛰️ Sentinel-2 L2A Real T1 [{z_nom}: {f_t1}]",
                    show=(vista_satelital_activa in ("t1", "ambas")),
                    zindex=606,
                ).add_to(mapa)

            if ov.get("mask_rgba") is not None:
                raster_layers.ImageOverlay(
                    image=ov["mask_rgba"],
                    bounds=bounds_ov,
                    opacity=0.90,
                    name=(
                        f"🔍 Cambio Espectral Real ΔNDVI/ΔNBR [{z_nom}: "
                        f"🔴{a_perd:.2f} ha | 🔥{a_quem:.2f} ha]"
                    ),
                    show=(vista_satelital_activa in ("t1", "mascara", "ambas")),
                    zindex=616,
                ).add_to(mapa)

            # Marco perimetral georreferenciado del recorte Sentinel-2 L2A (256×256 @ 10m)
            (s_lat, w_lon), (n_lat, e_lon) = bounds_ov
            marco_coords = [
                [n_lat, w_lon],
                [n_lat, e_lon],
                [s_lat, e_lon],
                [s_lat, w_lon],
            ]
            popup_s2 = (
                f"<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.45;min-width:240px;'>"
                f"<div style='background:#0f766e;color:white;padding:3px 7px;border-radius:4px;font-weight:700;font-size:10.5px;margin-bottom:6px;display:inline-block;'>"
                f"🛰️ SENTINEL-2 L2A REAL · TILE 18LUQ (10 m)</div><br>"
                f"<b>Ventana:</b> {z_nom}<br>"
                f"<b>Par Multitemporal:</b> <code>{f_t0}</code> ➔ <code>{f_t1}</code><br>"
                f"<b>🔴 Pérdida de cobertura (ΔNDVI):</b> <b>{a_perd:.2f} ha</b> ({ov.get('px_perdida_cobertura', 0)} px)<br>"
                f"<b>🔥 Cicatriz de quema (ΔNBR+SWIR):</b> <b>{a_quem:.2f} ha</b> ({ov.get('px_quema_espectral', 0)} px)<br>"
                f"<b>Nubosidad brillante descartada:</b> {ov.get('nubes_descartadas_pct', 0.0):.1f}%<br>"
                f"<div style='margin-top:6px;padding-top:5px;border-top:1px solid #cbd5e1;font-size:10px;color:#475569;'>"
                f"<b>LabelRule explícita:</b> <code>{ov.get('label_rule', '')}</code><br>"
                f"<b>Reproyección:</b> continuo <code>{ov.get('resampling_continuo', 'bilinear')}</code> · "
                f"categórico <code>{ov.get('resampling_categorico', 'nearest')}</code>.<br>"
                f"<i>Nota de dominio: Cambio físico de reflectancia superficial; no constituye certificación de tala ilegal ni inferencia calibrada.</i>"
                f"</div></div>"
            )
            folium.Polygon(
                locations=marco_coords,
                color="#14b8a6",
                weight=2,
                dash_array="4, 4",
                fill=False,
                tooltip=(
                    f"🛰️ Ventana Sentinel-2 L2A Real ({f_t0} → {f_t1}): "
                    f"Pérdida ΔNDVI {a_perd:.2f} ha | Quema ΔNBR {a_quem:.2f} ha"
                ),
                popup=folium.Popup(popup_s2, max_width=340),
            ).add_to(mapa)

    # 0.5 Capas demostrativas (solo cuando no provienen de GeoTIFF real, p. ej. Modo Pitch)
    elif capas_raster_satelital:
        es_real = bool(capas_raster_satelital.get("es_satelital_real", False))
        sufijo_demo = "" if es_real else " (Simulado — Modo Pitch)"
        bounds_sat = capas_raster_satelital["bounds"]
        t1_arr = capas_raster_satelital.get("t1_rgba")
        t0_arr = capas_raster_satelital.get("t0_rgba")
        mask_arr = capas_raster_satelital.get("mask_rgba")

        if t0_arr is not None:
            raster_layers.ImageOverlay(
                image=t0_arr,
                bounds=bounds_sat,
                opacity=0.94,
                name=f"🌲 Escena T0 Línea Base{sufijo_demo}",
                show=(vista_satelital_activa == "t0"),
                zindex=600,
            ).add_to(mapa)

        if t1_arr is not None:
            raster_layers.ImageOverlay(
                image=t1_arr,
                bounds=bounds_sat,
                opacity=0.94,
                name=f"🚨 Escena T1 Pérdida y Quema{sufijo_demo}",
                show=(vista_satelital_activa in ("t1", "ambas", "mascara")),
                zindex=605,
            ).add_to(mapa)

        if mask_arr is not None:
            raster_layers.ImageOverlay(
                image=mask_arr,
                bounds=bounds_sat,
                opacity=0.85,
                name=f"🔍 Máscara de Cambio{sufijo_demo}",
                show=(vista_satelital_activa in ("t1", "mascara")),
                zindex=615,
            ).add_to(mapa)

    # 0.8 Capa WMS Oficial en Vivo de PNCB Geobosques MINAM (Alertas Tempranas 2026)
    if mostrar_wms_geobosques:
        try:
            folium.WmsTileLayer(
                url=GEOBOSQUES_ATD_WMS_URL,
                layers="0",
                fmt="image/png",
                transparent=True,
                name="🌲 PNCB Geobosques ATD 2026 Ráster (WMS En Vivo)",
                overlay=True,
                control=True,
                show=True,
                opacity=0.85,
                attr="PNCB Geobosques MINAM Perú",
            ).add_to(mapa)
        except Exception:
            pass

    # 1. Capas en Vivo de Quemas e Incendios: NASA GIBS WMS + NASA FIRMS Multi-Sensor + SERFOR SAMI
    if mostrar_nasa_fires:
        folium.WmsTileLayer(
            url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
            layers="VIIRS_SNPP_Thermal_Anomalies_375m_All",
            fmt="image/png",
            transparent=True,
            name="🔥 NASA GIBS Anomalías Térmicas Ráster (WMS En Vivo)",
            overlay=True,
            control=True,
            show=True,
            opacity=0.85,
            attr="NASA Earthdata / GIBS",
        ).add_to(mapa)

        if df_nasa_firms is not None and not df_nasa_firms.empty:
            grupo_firms = folium.FeatureGroup(
                name=f"🔥 Quemas y Focos Activos NRT ({len(df_nasa_firms)} en vivo)",
                show=True,
            )
            for _, row in df_nasa_firms.iterrows():
                try:
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    if not (np.isfinite(lat) and np.isfinite(lon)):
                        continue

                    origen_feed = str(row.get("origen_feed", "NASA_FIRMS"))
                    es_serfor = origen_feed.startswith("SERFOR_")
                    sensor_lbl = str(
                        row.get("sensor_fuente")
                        or ("SERFOR SAMI UFMS" if es_serfor else "VIIRS 375m NRT")
                    )

                    brillo_raw = pd.to_numeric(row.get("bright_ti4", np.nan), errors="coerce")
                    tiene_brillo = pd.notna(brillo_raw)
                    brillo_k = float(brillo_raw) if tiene_brillo else np.nan
                    brillo_c = brillo_k - 273.15 if tiene_brillo else np.nan

                    frp_raw = pd.to_numeric(row.get("frp", 0.0), errors="coerce")
                    frp = float(frp_raw) if pd.notna(frp_raw) else 0.0

                    fecha = str(row.get("acq_date", ""))
                    hora = str(row.get("acq_time", "")).zfill(4)
                    fh_pet = str(
                        row.get("fecha_hora_pet") or f"{fecha} {hora[:2]}:{hora[2:]} UTC"
                    )
                    fh_utc = str(
                        row.get("fecha_hora_utc") or f"{fecha} {hora[:2]}:{hora[2:]} UTC"
                    )
                    conf_foco = str(row.get("confidence", "nominal"))

                    if es_serfor:
                        color_fuego = "#f97316"
                        titulo_badge = "🔥 SERFOR SAMI · ALERTA OFICIAL DE INCENDIO / FOCO"
                    elif (tiene_brillo and brillo_k >= 330.0) or frp >= 10.0:
                        color_fuego = "#ff1e00"
                        titulo_badge = "🔥 NASA FIRMS NRT · QUEMA / FOCO TÉRMICO INTENSO"
                    else:
                        color_fuego = "#ff6600"
                        titulo_badge = "🔥 NASA FIRMS NRT · ANOMALÍA TÉRMICA EN VIVO"

                    linea_temp = (
                        f"<b>Temperatura de Brillo:</b> {brillo_k:.1f} K ({brillo_c:.1f} °C)<br>"
                        if tiene_brillo
                        else "<b>Temperatura de Brillo:</b> No reportada en este feed vectorial<br>"
                    )
                    linea_frp = (
                        f"<b>Potencia Radiativa (FRP):</b> {frp:.2f} MW<br>"
                        if frp > 0.0
                        else "<b>Potencia Radiativa (FRP):</b> s/d<br>"
                    )

                    popup_fuego = (
                        f"<div style='font-family:system-ui,sans-serif;font-size:12px;line-height:1.45;min-width:215px;'>"
                        f"<div style='background:{color_fuego};color:white;padding:3px 7px;border-radius:4px;font-weight:bold;font-size:10px;margin-bottom:6px;display:inline-block;'>"
                        f"{titulo_badge}</div><br>"
                        f"<b>Sensor / Fuente:</b> {sensor_lbl}<br>"
                        f"<b>Hora Perú (PET):</b> <code>{fh_pet}</code><br>"
                        f"<b>Hora Orbital (UTC):</b> <code>{fh_utc}</code><br>"
                        f"{linea_temp}"
                        f"{linea_frp}"
                        f"<b>Nivel del sensor:</b> {conf_foco}<br>"
                        f"<b>Coordenadas:</b> <code>{lat:.5f}, {lon:.5f}</code>"
                        f"</div>"
                    )

                    temp_tip = f"{brillo_c:.1f} °C · {frp:.1f} MW" if tiene_brillo else sensor_lbl
                    folium.Marker(
                        location=[lat, lon],
                        icon=folium.DivIcon(
                            html=(
                                f"<div class='gf-fire-dot' style='background:{color_fuego};'></div>"
                            ),
                            icon_size=(14, 14),
                            icon_anchor=(7, 7),
                        ),
                        tooltip=f"🔥 Quema / Foco en vivo ({fh_pet}): {temp_tip}",
                        popup=folium.Popup(popup_fuego, max_width=290),
                    ).add_to(grupo_firms)
                except Exception:
                    continue
            grupo_firms.add_to(mapa)

    # 2. Capa de polígonos de zonas de estudio
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
                fill_opacity=0.07,
                tooltip=f"Zona de estudio: {nombre_z}",
                popup=folium.Popup(
                    f"<b>{nombre_z}</b><br>{datos_z['descripcion']}<br>"
                    "<i>Delimitación referencial de monitoreo</i>",
                    max_width=280,
                ),
            ).add_to(grupo_zonas)
        grupo_zonas.add_to(mapa)

    # 2.5 Polígono destacado de auditoría
    if poligono_destacado:
        grupo_destacado = folium.FeatureGroup(name="🚨 Zona de Impacto Seleccionada", show=True)
        folium.Polygon(
            locations=poligono_destacado,
            color="#e11d48",
            weight=3.5,
            dash_array="8, 4",
            fill=True,
            fill_color="#f43f5e",
            fill_opacity=0.25,
            tooltip=etiqueta_destacada or "Zona crítica de pérdida de cobertura / quema",
            popup=folium.Popup(
                f"<div style='font-family:system-ui;min-width:220px;padding:4px;'>"
                f"<div style='background:#b91c1c;color:white;padding:3px 6px;border-radius:4px;font-weight:700;font-size:11px;margin-bottom:6px;'>🚨 ZONA DE AUDITORÍA MULTITEMPORAL</div>"
                f"<b>{etiqueta_destacada or 'Área con cambio de cobertura'}</b><br>"
                f"<span style='font-size:11px;color:#64748b;'>Análisis multitemporal Sentinel-2 (T0 vs T1)</span>"
                f"</div>",
                max_width=320,
            ),
        ).add_to(grupo_destacado)
        grupo_destacado.add_to(mapa)

    # 3. Capa de polígonos de alertas de pérdida de cobertura forestal (ATD PNCB MINAM)
    if "paso" in df_filtrado.columns:
        df_render = df_filtrado.tail(400)
    else:
        df_render = df_filtrado.head(400) if len(df_filtrado) > 400 else df_filtrado

    if mostrar_poligonos_alerta and not df_render.empty:
        grupo_poligonos = folium.FeatureGroup(
            name=f"🌲 Pérdida de Cobertura Forestal ATD ({len(df_render)} visibles)",
            show=True,
        )
        for _, fila in df_render.iterrows():
            tipo_ev = str(fila.get("tipo_evento", "")).lower()
            origen_d = str(fila.get("origen_dato", "PNCB_GEOBOSQUES_MINAM"))
            es_api_vivo = origen_d == "PNCB_GEOBOSQUES_API_VIVO"

            if "quema" in tipo_ev:
                color_poly = "#e11d48"   # Pérdida de cobertura + quema activa cercana
                borde_poly = "#ff8800"
            elif "natural" in tipo_ev:
                color_poly = "#d97706"   # Pérdida natural de cobertura
                borde_poly = "#f59e0b"
            elif "deforest" in tipo_ev or "cobertura" in tipo_ev or "tala" in tipo_ev:
                color_poly = "#d73027"   # Pérdida de cobertura forestal / deforestación ATD
                borde_poly = "#ef4444" if es_api_vivo else "#d73027"
            else:
                color_poly = COLORES_SEVERIDAD.get(fila["severidad"], "#d73027")
                borde_poly = color_poly

            if "delta_lat" in fila and "delta_lon" in fila and pd.notna(fila["delta_lat"]):
                d_lat, d_lon = float(fila["delta_lat"]), float(fila["delta_lon"])
            else:
                lado_deg = max(
                    0.0012,
                    float(np.sqrt(max(float(fila.get("area_ha", 0.5)), 0.01))) * 0.0012,
                )
                d_lat, d_lon = lado_deg, lado_deg
            coords_poligono = construir_poligono_alerta(
                float(fila["lat"]), float(fila["lon"]), d_lat, d_lon
            )
            tag_origen = (
                "🟢 API EN VIVO PNCB GEOBOSQUES MINAM"
                if es_api_vivo
                else f"Fuente: {origen_d}"
            )
            popup_html = (
                f"<div style='font-family: system-ui, -apple-system, sans-serif; font-size: 12.5px; line-height: 1.45; min-width: 225px;'>"
                f"<div style='background:{color_poly}; color:white; padding:3px 7px; "
                f"border-radius:4px; font-weight:bold; font-size:10.5px; margin-bottom:7px; display:inline-block;'>"
                f"{fila['tipo_evento'].upper()} — {fila['severidad'].upper()}</div><br>"
                f"<b>ID Alerta:</b> <code>{fila['id_alerta']}</code><br>"
                f"<b>Superficie ATD estimada:</b> <b>{fila['area_ha']:.2f} ha</b> ({int(fila.get('puntos', 1))} pts)<br>"
                f"<b>Fecha de detección:</b> <code>{fila['fecha']}</code><br>"
                f"<b>Sector / Distrito:</b> {fila['sector']}, {fila['distrito']}<br>"
                f"<b>Zona de estudio:</b> {fila['zona']}<br>"
                f"<b>Score heurístico (sin calibrar):</b> {fila['confianza_pct']:.1f}%<br>"
                f"<div style='margin-top:6px; font-size:11px; color:#334155;'><b>{tag_origen}</b> · {fila.get('estado', '')}</div>"
                f"<div style='margin-top:6px; padding-top:5px; border-top:1px solid #e2e8f0; font-size:10px; color:#64748b; line-height:1.3;'>"
                f"ℹ️ <i>Alerta satelital de pérdida de cobertura boscosa; no certifica tala ilegal sin verificación en campo (SERFOR / FEMA). Score heurístico rotulado «sin calibrar».</i>"
                f"</div>"
                f"</div>"
            )
            tooltip_text = (
                f"🌲 [{fila['id_alerta']}] {fila['tipo_evento']} ({fila['fecha']}) | "
                f"Área: {fila['area_ha']:.2f} ha"
            )
            folium.Polygon(
                locations=coords_poligono,
                color=borde_poly,
                weight=3.2 if "quema" in tipo_ev else 2.5,
                fill=True,
                fill_color=color_poly,
                fill_opacity=0.58,
                tooltip=tooltip_text,
                popup=folium.Popup(popup_html, max_width=330),
            ).add_to(grupo_poligonos)
        grupo_poligonos.add_to(mapa)

    # 4. Capa de centroides puntuales de alertas ATD
    if mostrar_marcadores and not df_render.empty:
        grupo_marcadores = folium.FeatureGroup(name="Centroides ATD y Severidad", show=True)
        for _, fila in df_render.iterrows():
            tipo_ev = str(fila.get("tipo_evento", "")).lower()
            if "quema" in tipo_ev:
                color_punto = "#e11d48"
                borde_punto = "#ffedd5"
            elif "natural" in tipo_ev:
                color_punto = "#d97706"
                borde_punto = "#ffffff"
            elif "deforest" in tipo_ev or "cobertura" in tipo_ev or "tala" in tipo_ev:
                color_punto = "#d73027"
                borde_punto = "#ffffff"
            else:
                color_punto = COLORES_SEVERIDAD.get(fila["severidad"], "#d73027")
                borde_punto = "#ffffff"

            folium.CircleMarker(
                location=[fila["lat"], fila["lon"]],
                radius=6.5,
                color=borde_punto,
                weight=1.8,
                fill=True,
                fill_color=color_punto,
                fill_opacity=0.92,
                tooltip=(
                    f"[{fila['id_alerta']}] {fila['fecha']} · {fila['sector']} "
                    f"({fila['tipo_evento']} - {fila['area_ha']:.2f} ha)"
                ),
            ).add_to(grupo_marcadores)
        grupo_marcadores.add_to(mapa)

    # 4.5 Secuencia cronológica de hitos (T0 -> T1)
    if secuencia_hitos:
        grupo_secuencia = folium.FeatureGroup(
            name="📍 Secuencia Temporal de Hitos [T0 ➔ T1]", show=False
        )
        coords_linea = [[float(h["lat"]), float(h["lon"])] for h in secuencia_hitos]
        folium.PolyLine(
            locations=coords_linea,
            color="#f43f5e",
            weight=3.5,
            opacity=0.85,
            dash_array="8, 6",
            tooltip="Vector cronológico de referencia del frente de cambio",
        ).add_to(grupo_secuencia)

        for h in secuencia_hitos:
            lat_h = float(h["lat"])
            lon_h = float(h["lon"])
            col_h = h.get("color", "#e11d48")
            fecha_h = h.get("fecha", "")
            paso_h = h.get("paso", 1)
            rom_h = h.get("numero_romano", f"#{paso_h}")
            area_acum = float(h.get("area_acum_ha", 0.0))
            area_paso = float(h.get("area_paso_ha", 0.0))
            fase_h = h.get("fase", f"Hito {paso_h}")
            diag_h = h.get("diagnostico", "")
            delta_n = float(h.get("delta_ndvi", -0.30))
            pts_m = int(h.get("puntos_minam", 50))

            badge_html = (
                f"<div class='gf-milestone-marker' style='background:{col_h}; color:white; "
                f"padding:3px 9px; border-radius:12px; font-weight:700; font-size:11px; "
                f"font-family:system-ui,-apple-system,sans-serif; border:2px solid #ffffff; "
                f"box-shadow:0 3px 10px rgba(0,0,0,0.65); display:inline-block; cursor:pointer;'>"
                f"{rom_h} {fecha_h[5:]} ({area_acum:.1f} ha)"
                f"</div>"
            )

            popup_h_html = (
                f"<div style='font-family:system-ui,-apple-system,sans-serif;font-size:12px;line-height:1.45;min-width:240px;padding:4px;'>"
                f"<div style='background:{col_h};color:white;padding:4px 8px;border-radius:4px;font-weight:bold;font-size:11px;margin-bottom:6px;'>"
                f"🚨 {fase_h}</div>"
                f"<b>Fecha de Detección:</b> <code>{fecha_h}</code><br>"
                f"<b>Superficie del Hito:</b> <span style='color:#e11d48;font-weight:700;'>+{area_paso:.2f} ha</span><br>"
                f"<b>Área Acumulada:</b> <span style='font-weight:700;'>{area_acum:.2f} ha</span><br>"
                f"<b>Anomalía ΔNDVI:</b> <code>{delta_n:.2f}</code><br>"
                f"<b>Puntos Satelitales:</b> {pts_m} píxeles Sentinel-2 (PNCB MINAM)<br>"
                f"<b>Coordenadas:</b> <code>{lat_h:.5f}, {lon_h:.5f}</code><br>"
                f"<div style='margin-top:6px;font-size:11px;color:#475569;background:rgba(0,0,0,0.04);padding:5px;border-radius:4px;'>"
                f"📝 <i>{diag_h}</i>"
                f"</div>"
                f"</div>"
            )

            folium.Marker(
                location=[lat_h, lon_h],
                icon=folium.DivIcon(
                    html=badge_html,
                    icon_size=(130, 26),
                    icon_anchor=(65, 13),
                ),
                tooltip=f"Hito {paso_h} ({fecha_h}): +{area_paso:.1f} ha | Acumulado: {area_acum:.1f} ha",
                popup=folium.Popup(popup_h_html, max_width=300),
            ).add_to(grupo_secuencia)

        grupo_secuencia.add_to(mapa)

    # 4.8 HUD flotante de telemetría en tiempo real (esquina inferior izquierda)
    if telemetria_tiempo_real:
        sync_lbl = telemetria_tiempo_real.get("sync_pet", "En vivo")
        n_atd = int(telemetria_tiempo_real.get("alertas_visibles", len(df_render)))
        n_fuego = int(
            telemetria_tiempo_real.get(
                "quemas_activas", len(df_nasa_firms) if df_nasa_firms is not None else 0
            )
        )
        hud_html = (
            "<div class='gf-live-hud'>"
            "<div style='font-weight:700;color:#34d399;margin-bottom:3px;'>"
            "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:#10b981;"
            "margin-right:6px;animation:gfLiveBlink 1.5s infinite;'></span>"
            f"TELEMETRÍA EN VIVO · {sync_lbl}</div>"
            f"<div><span style='color:#ef4444;font-weight:700;'>●</span> Pérdida cobertura forestal (ATD / ΔNDVI): <b>{n_atd}</b></div>"
            f"<div><span style='color:#ff6600;font-weight:700;'>●</span> Quemas / focos térmicos NRT (FIRMS+SERFOR): <b>{n_fuego}</b></div>"
            "</div>"
        )
        mapa.get_root().html.add_child(folium.Element(hud_html))

    # 5. Pantalla completa y selector de capas
    plugins.Fullscreen(
        position="topleft",
        title="Pantalla completa (Expandir)",
        title_cancel="Salir de pantalla completa (Cerrar)",
        force_separate_button=True,
    ).add_to(mapa)

    folium.LayerControl(position="topright", collapsed=panel_capas_colapsado).add_to(mapa)

    return mapa
