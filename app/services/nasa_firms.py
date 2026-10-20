"""Servicio de integración con la API en vivo de NASA FIRMS (VIIRS)."""

from __future__ import annotations

import io
import pandas as pd
import requests
import streamlit as st


@st.cache_data(ttl=900)
def consultar_focos_calor_nasa_firms(
    map_key: str = "3110fea2ae317068ba600eb4c906758c",
    bbox: tuple[float, float, float, float] = (-76.6, -10.3, -75.5, -9.2),
    dias: int = 3,
) -> pd.DataFrame:
    """Consulta la API de NASA FIRMS en tiempo real para obtener anomalías térmicas e incendios
    detectados por el sensor VIIRS (Suomi-NPP / NOAA-20) en el corredor de Huánuco.
    """
    if not map_key or len(map_key) < 10:
        return pd.DataFrame()

    west, south, east, north = bbox
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/{dias}"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200 and "latitude" in resp.text:
            df = pd.read_csv(io.StringIO(resp.text))
            return df
    except Exception:
        pass
    return pd.DataFrame()
