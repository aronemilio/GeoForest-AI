"""Servicio de gestión y generación de alertas demostrativas simuladas."""

from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st


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
        {
            "id_alerta": "IA-HCO-042",
            "origen_dato": "SIMULADO",
            "fecha": "2026-10-16",
            "zona": "Bosque de Carpish",
            "sector": "Sector Túnel de Carpish (Km 42)",
            "distrito": "Chinchao",
            "tipo_evento": "Deforestación",
            "severidad": "Alta",
            "area_ha": 1.24,
            "confianza_pct": 94.2,
            "ndvi_cambio": -0.46,
            "lat": -9.7020,
            "lon": -76.0890,
            "delta_lat": 0.0035,
            "delta_lon": 0.0039,
            "estado": "Alerta prioritaria (FEMA)",
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


def generar_geojson_alertas(df_alertas: pd.DataFrame) -> str:
    """Construye un FeatureCollection GeoJSON formal a partir de las alertas."""
    features = []
    for _, fila in df_alertas.iterrows():
        coords_poly = construir_poligono_alerta(
            fila["lat"], fila["lon"], fila["delta_lat"], fila["delta_lon"]
        )
        geojson_coords = [[lon, lat] for lat, lon in coords_poly]
        geojson_coords.append(geojson_coords[0])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geojson_coords],
            },
            "properties": {
                "id_alerta": fila["id_alerta"],
                "origen_dato": fila.get("origen_dato", "SIMULADO"),
                "fecha": str(fila["fecha"]),
                "zona": fila["zona"],
                "sector": fila["sector"],
                "distrito": fila["distrito"],
                "tipo_evento": fila["tipo_evento"],
                "severidad": fila["severidad"],
                "area_ha": float(fila["area_ha"]),
                "confianza_pct": float(fila["confianza_pct"]),
                "ndvi_cambio": float(fila["ndvi_cambio"]),
                "lat_centroide": float(fila["lat"]),
                "lon_centroide": float(fila["lon"]),
                "crs": "EPSG:32718",
                "organismo_destino": "FEMA Huánuco / SERFOR ATFFS Huánuco",
            },
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "name": "GeoForest_AI_Huanuco_Alertas",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    return json.dumps(feature_collection, indent=2, ensure_ascii=False)


def generar_recorte_offline_huanuco(zona: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Genera en memoria un recorte satelital multitemporal sintético determinista para Huánuco."""
    from rasterio.crs import CRS
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    crs_utm = CRS.from_epsg(32718)
    if "Carpish" in zona:
        x_orig, y_orig = 380000.0, 8927000.0
        seed = 101
        sector = "Bosque Montano de Carpish (Túnel - Chinchao)"
    else:
        x_orig, y_orig = 365000.0, 8900000.0
        seed = 202
        sector = "Valle del Huallaga (Pillco Marca - Huancachupa)"

    rng = np.random.default_rng(seed)
    h_10m, w_10m = 128, 128
    h_20m, w_20m = 64, 64
    tf_10m = from_origin(x_orig, y_orig, 10.0, 10.0)
    tf_20m = from_origin(x_orig, y_orig, 20.0, 20.0)

    t0_files: dict[str, Any] = {}
    t1_files: dict[str, Any] = {}

    for name in ("B4", "B8"):
        arr_t0 = rng.integers(300, 3500, size=(1, h_10m, w_10m), dtype=np.uint16)
        arr_t1 = arr_t0.copy()
        if name == "B4":
            arr_t1[0, 45:75, 45:75] = np.clip(arr_t1[0, 45:75, 45:75] + 1900, 0, 10000)
        else:
            arr_t1[0, 45:75, 45:75] = np.clip(arr_t1[0, 45:75, 45:75] - 2100, 100, 10000)

        mf0 = MemoryFile()
        with mf0.open(driver="GTiff", height=h_10m, width=w_10m, count=1, dtype="uint16", crs=crs_utm, transform=tf_10m) as ds:
            ds.write(arr_t0)
        t0_files[name] = mf0

        mf1 = MemoryFile()
        with mf1.open(driver="GTiff", height=h_10m, width=w_10m, count=1, dtype="uint16", crs=crs_utm, transform=tf_10m) as ds:
            ds.write(arr_t1)
        t1_files[name] = mf1

    # B11 nativa a 20m (64x64 píxeles) para validar remuestreo bilineal a 10m
    arr_b11_t0 = rng.integers(400, 2800, size=(1, h_20m, w_20m), dtype=np.uint16)
    arr_b11_t1 = arr_b11_t0.copy()
    arr_b11_t1[0, 22:38, 22:38] = np.clip(arr_b11_t1[0, 22:38, 22:38] + 1700, 0, 10000)

    mf0_b11 = MemoryFile()
    with mf0_b11.open(driver="GTiff", height=h_20m, width=w_20m, count=1, dtype="uint16", crs=crs_utm, transform=tf_20m) as ds:
        ds.write(arr_b11_t0)
    t0_files["B11"] = mf0_b11

    mf1_b11 = MemoryFile()
    with mf1_b11.open(driver="GTiff", height=h_20m, width=w_20m, count=1, dtype="uint16", crs=crs_utm, transform=tf_20m) as ds:
        ds.write(arr_b11_t1)
    t1_files["B11"] = mf1_b11

    return t0_files, t1_files, sector
