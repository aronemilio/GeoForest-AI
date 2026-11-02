"""Servicio de integración en tiempo real con NASA FIRMS (VIIRS/MODIS/Landsat NRT) y SERFOR SAMI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import io
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

TZ_PERU = timezone(timedelta(hours=-5))

SENSORES_NASA_FIRMS_NRT: dict[str, str] = {
    "VIIRS_SNPP_NRT": "VIIRS Suomi-NPP (375m)",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20 (375m)",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21 (375m)",
    "MODIS_NRT": "MODIS Terra/Aqua (1km)",
    "LANDSAT_NRT": "Landsat 8/9 NRT (30m)",
}

SERFOR_SAMI_BASE_URL = (
    "https://geo.serfor.gob.pe/geoservicios/rest/services/UFMS/Incendios_SAMIv21/MapServer"
)

COLUMNAS_QUEMAS_VIVO: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_c",
    "frp",
    "acq_date",
    "acq_time",
    "fecha_hora_utc",
    "fecha_hora_pet",
    "satellite",
    "instrument",
    "confidence",
    "daynight",
    "sensor_fuente",
    "origen_feed",
    "zona",
)


def _quemas_vacias() -> pd.DataFrame:
    """Devuelve un DataFrame vacío con el esquema estandarizado de quemas/focos térmicos."""
    return pd.DataFrame(columns=list(COLUMNAS_QUEMAS_VIVO))


def _clasificar_zona_estudio(lat: float, lon: float) -> str:
    """Clasifica una coordenada geográfica dentro de las zonas de estudio de Huánuco."""
    if -9.7600 <= lat <= -9.6500 and -76.1600 <= lon <= -76.0300:
        return "Bosque de Carpish"
    if -9.6500 < lat <= -9.4800 and -76.1500 <= lon <= -75.9500:
        return "Valle del Huallaga"
    if -9.9500 <= lat <= -9.4800 and -76.3500 <= lon <= -75.8500:
        return "Valle del Huallaga"
    return "Corredor Regional Huánuco"


def _formatear_hora_utc_a_pet(acq_date: str, acq_time: str) -> tuple[str, str]:
    """Convierte fecha YYYY-MM-DD y hora HHMM UTC a cadenas legibles UTC y PET (UTC-5)."""
    hhmm = str(acq_time or "0000").strip().zfill(4)[:4]
    if not hhmm.isdigit():
        hhmm = "0000"
    fecha_limpia = str(acq_date or "").strip()[:10]
    try:
        dt_utc = datetime.strptime(f"{fecha_limpia} {hhmm}", "%Y-%m-%d %H%M").replace(
            tzinfo=timezone.utc
        )
        dt_pet = dt_utc.astimezone(TZ_PERU)
        return dt_utc.strftime("%Y-%m-%d %H:%M UTC"), dt_pet.strftime("%Y-%m-%d %H:%M PET")
    except ValueError:
        return f"{fecha_limpia} {hhmm[:2]}:{hhmm[2:]} UTC", f"{fecha_limpia} PET"


def _fetch_sensor_firms(
    key: str,
    sensor_code: str,
    sensor_label: str,
    bbox: tuple[float, float, float, float],
    dias_nrt: int,
    timeout: int = 7,
) -> pd.DataFrame:
    """Consulta un sensor individual de NASA FIRMS NRT y normaliza sus columnas físicas."""
    west, south, east, north = bbox
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{key}/{sensor_code}/{west},{south},{east},{north}/{dias_nrt}"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or "latitude" not in resp.text:
            return _quemas_vacias()

        df_raw = pd.read_csv(io.StringIO(resp.text))
        if df_raw.empty or "latitude" not in df_raw.columns or "longitude" not in df_raw.columns:
            return _quemas_vacias()

        df = pd.DataFrame()
        df["latitude"] = pd.to_numeric(df_raw["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df_raw["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        if df.empty:
            return _quemas_vacias()

        # Temperatura de brillo en Kelvin (VIIRS usa bright_ti4; MODIS usa brightness)
        if "bright_ti4" in df_raw.columns:
            bright_k = pd.to_numeric(df_raw.loc[df.index, "bright_ti4"], errors="coerce")
        elif "brightness" in df_raw.columns:
            bright_k = pd.to_numeric(df_raw.loc[df.index, "brightness"], errors="coerce")
        else:
            bright_k = pd.Series(np.nan, index=df.index, dtype=float)

        df["bright_ti4"] = bright_k
        df["bright_c"] = np.where(bright_k.notna(), np.round(bright_k - 273.15, 1), np.nan)
        df["frp"] = (
            pd.to_numeric(df_raw.loc[df.index, "frp"], errors="coerce").fillna(0.0)
            if "frp" in df_raw.columns
            else 0.0
        )
        df["acq_date"] = (
            df_raw.loc[df.index, "acq_date"].astype(str).str.slice(0, 10)
            if "acq_date" in df_raw.columns
            else ""
        )
        df["acq_time"] = (
            df_raw.loc[df.index, "acq_time"].astype(str).str.strip().str.zfill(4)
            if "acq_time" in df_raw.columns
            else "0000"
        )

        fechas_fmt = [
            _formatear_hora_utc_a_pet(f_d, f_t)
            for f_d, f_t in zip(df["acq_date"], df["acq_time"])
        ]
        df["fecha_hora_utc"] = [p[0] for p in fechas_fmt]
        df["fecha_hora_pet"] = [p[1] for p in fechas_fmt]
        df["satellite"] = (
            df_raw.loc[df.index, "satellite"].astype(str)
            if "satellite" in df_raw.columns
            else sensor_code.split("_")[0]
        )
        df["instrument"] = (
            df_raw.loc[df.index, "instrument"].astype(str)
            if "instrument" in df_raw.columns
            else sensor_code.split("_")[0]
        )
        df["confidence"] = (
            df_raw.loc[df.index, "confidence"].astype(str)
            if "confidence" in df_raw.columns
            else "nominal"
        )
        df["daynight"] = (
            df_raw.loc[df.index, "daynight"].astype(str)
            if "daynight" in df_raw.columns
            else ""
        )
        df["sensor_fuente"] = sensor_label
        df["origen_feed"] = f"NASA_FIRMS_{sensor_code}"
        df["zona"] = [
            _clasificar_zona_estudio(float(la), float(lo))
            for la, lo in zip(df["latitude"], df["longitude"])
        ]
        return df[list(COLUMNAS_QUEMAS_VIVO)]
    except requests.RequestException as req_err:
        logger.warning("Fallo de conexión al consultar sensor NASA FIRMS %s: %s", sensor_code, req_err)
        return _quemas_vacias()
    except Exception as exc:
        logger.debug("Error procesando telemetría de sensor NASA FIRMS %s: %s", sensor_code, exc)
        return _quemas_vacias()


@st.cache_data(ttl=300, show_spinner=False)
def consultar_focos_calor_nasa_firms(
    map_key: str | None = None,
    bbox: tuple[float, float, float, float] = (-76.6, -10.3, -75.5, -9.0),
    dias: int = 3,
    sensores: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Consulta en paralelo los sensores NRT de NASA FIRMS (VIIRS SNPP/NOAA-20/NOAA-21, MODIS, Landsat).

    Nota: El endpoint NRT de NASA FIRMS admite un rango estricto de 1 a 5 días; valores fuera
    de ese rango se acotan automáticamente a [1, 5] para evitar respuestas HTTP 400.
    Si no hay clave válida o no existen anomalías térmicas reales en la ventana, devuelve
    un DataFrame vacío con esquema tipado (sin valores ficticios).
    """
    key = (map_key if map_key is not None else os.environ.get("NASA_API_KEY", "")).strip()
    if not key or len(key) < 10:
        return _quemas_vacias()

    dias_nrt = max(1, min(int(dias), 5))
    codigos = sensores or tuple(SENSORES_NASA_FIRMS_NRT.keys())
    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(len(codigos), 5)) as pool:
        futuros = {
            pool.submit(
                _fetch_sensor_firms,
                key,
                code,
                SENSORES_NASA_FIRMS_NRT.get(code, code),
                bbox,
                dias_nrt,
            ): code
            for code in codigos
        }
        for fut in as_completed(futuros):
            df_s = fut.result()
            if not df_s.empty:
                frames.append(df_s)

    if not frames:
        return _quemas_vacias()

    combinado = pd.concat(frames, ignore_index=True)
    combinado["_lat_r"] = combinado["latitude"].round(4)
    combinado["_lon_r"] = combinado["longitude"].round(4)
    combinado = (
        combinado.sort_values(
            by=["acq_date", "acq_time", "frp"], ascending=[False, False, False]
        )
        .drop_duplicates(subset=["acq_date", "acq_time", "_lat_r", "_lon_r", "origen_feed"])
        .drop(columns=["_lat_r", "_lon_r"])
        .reset_index(drop=True)
    )
    return combinado


def _extraer_lat_lon_geometria_arcgis(geom: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Extrae latitud y longitud de una geometría puntual o poligonal de ArcGIS REST (EPSG:4326)."""
    if not geom or not isinstance(geom, dict):
        return None, None
    if "y" in geom and "x" in geom and geom["y"] is not None and geom["x"] is not None:
        return float(geom["y"]), float(geom["x"])
    rings = geom.get("rings")
    if rings and isinstance(rings, list) and rings[0]:
        pts = np.asarray(rings[0], dtype=float)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            return float(np.mean(pts[:, 1])), float(np.mean(pts[:, 0]))
    return None, None


@st.cache_data(ttl=300, show_spinner=False)
def consultar_alertas_incendios_serfor_sami(
    bbox: tuple[float, float, float, float] = (-76.6, -10.3, -75.5, -9.0),
    dias: int = 5,
    limite: int = 250,
) -> pd.DataFrame:
    """Consulta en tiempo real el servicio oficial SERFOR UFMS (Incendios_SAMIv21) para Huánuco.

    No requiere API Key y recupera alertas oficiales de incendio forestal (Layer 0) y
    focos de calor registrados por la Unidad Funcional de Monitoreo Satelital de SERFOR (Layer 2)
    dentro de la ventana temporal solicitada.
    """
    west, south, east, north = bbox
    ahora_utc = datetime.now(timezone.utc)
    corte_utc = ahora_utc - timedelta(days=max(1, int(dias)))
    corte_ms = int(corte_utc.timestamp() * 1000)

    capas = (
        (0, "SERFOR SAMI — Alerta de Incendio Forestal", "SERFOR_SAMI_ALERTA"),
        (2, "SERFOR SAMI — Foco de Calor Satelital", "SERFOR_SAMI_FOCO"),
    )
    registros: list[dict[str, Any]] = []

    for layer_id, sensor_label, origen_tag in capas:
        url = f"{SERFOR_SAMI_BASE_URL}/{layer_id}/query"
        params = {
            "where": f"FECREG >= timestamp '{corte_utc.strftime('%Y-%m-%d')} 00:00:00'",
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "orderByFields": "FECREG DESC",
            "resultRecordCount": limite,
            "f": "json",
        }
        try:
            resp = requests.get(url, params=params, timeout=7)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            if "error" in payload:
                # Reintento sin filtro SQL timestamp si el servidor ArcGIS difiere en dialecto SQL
                params_fallback = dict(params)
                params_fallback["where"] = "1=1"
                resp = requests.get(url, params=params_fallback, timeout=7)
                if resp.status_code != 200:
                    continue
                payload = resp.json()

            for feat in payload.get("features", []):
                attrs = feat.get("attributes") or {}
                fec_ms = attrs.get("FECREG")
                if fec_ms is not None and int(fec_ms) < corte_ms:
                    continue
                lat, lon = _extraer_lat_lon_geometria_arcgis(feat.get("geometry"))
                if lat is None or lon is None:
                    continue
                if not (south <= lat <= north and west <= lon <= east):
                    continue

                if fec_ms is not None:
                    dt_utc = datetime.fromtimestamp(int(fec_ms) / 1000.0, tz=timezone.utc)
                    dt_pet = dt_utc.astimezone(TZ_PERU)
                    acq_date = dt_utc.strftime("%Y-%m-%d")
                    acq_time = dt_utc.strftime("%H%M")
                    f_utc = dt_utc.strftime("%Y-%m-%d %H:%M UTC")
                    f_pet = dt_pet.strftime("%Y-%m-%d %H:%M PET")
                else:
                    acq_date = ahora_utc.strftime("%Y-%m-%d")
                    acq_time = "0000"
                    f_utc = f"{acq_date} UTC"
                    f_pet = f"{acq_date} PET"

                sat_str = str(attrs.get("SATELITE") or "UFMS Satelital").strip()
                conf_str = str(attrs.get("CONFIANZA") or "oficial_serfor").strip()
                registros.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "bright_ti4": np.nan,
                        "bright_c": np.nan,
                        "frp": 0.0,
                        "acq_date": acq_date,
                        "acq_time": acq_time,
                        "fecha_hora_utc": f_utc,
                        "fecha_hora_pet": f_pet,
                        "satellite": sat_str,
                        "instrument": "SAMI-UFMS",
                        "confidence": conf_str,
                        "daynight": "",
                        "sensor_fuente": sensor_label,
                        "origen_feed": origen_tag,
                        "zona": _clasificar_zona_estudio(lat, lon),
                    }
                )
        except requests.RequestException as req_err:
            logger.warning("Fallo de conexión al consultar SERFOR SAMI capa %s: %s", layer_id, req_err)
            continue
        except Exception as exc:
            logger.debug("Error procesando registros de SERFOR SAMI capa %s: %s", layer_id, exc)
            continue

    if not registros:
        return _quemas_vacias()

    df_serfor = pd.DataFrame(registros)
    df_serfor["_lat_r"] = df_serfor["latitude"].round(4)
    df_serfor["_lon_r"] = df_serfor["longitude"].round(4)
    df_serfor = (
        df_serfor.sort_values(by=["acq_date", "acq_time"], ascending=[False, False])
        .drop_duplicates(subset=["acq_date", "_lat_r", "_lon_r", "origen_feed"])
        .drop(columns=["_lat_r", "_lon_r"])
        .reset_index(drop=True)
    )
    return df_serfor[list(COLUMNAS_QUEMAS_VIVO)]


def obtener_quemas_tiempo_real_integradas(
    map_key: str | None = None,
    bbox: tuple[float, float, float, float] = (-76.6, -10.3, -75.5, -9.0),
    dias: int = 3,
    incluir_serfor: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Integra los focos térmicos NRT de NASA FIRMS (multi-sensor) y alertas de incendio SERFOR SAMI.

    Returns:
        Tupla `(df_quemas, telemetria_quemas)` sin valores de relleno cuando no hay eventos activos.
    """
    ahora_utc = datetime.now(timezone.utc)
    ahora_pet = ahora_utc.astimezone(TZ_PERU)

    df_firms = consultar_focos_calor_nasa_firms(map_key=map_key, bbox=bbox, dias=dias)
    df_serfor = (
        consultar_alertas_incendios_serfor_sami(bbox=bbox, dias=dias)
        if incluir_serfor
        else _quemas_vacias()
    )

    partes = [d for d in (df_firms, df_serfor) if not d.empty]
    if partes:
        df_total = pd.concat(partes, ignore_index=True)
        # Priorizar registros NASA FIRMS con telemetría radiométrica (Kelvin y FRP) si coinciden en ~110 m y fecha
        df_total["_tiene_temp"] = df_total["bright_ti4"].notna().astype(int)
        df_total["_lat_c"] = df_total["latitude"].round(3)
        df_total["_lon_c"] = df_total["longitude"].round(3)
        df_total = (
            df_total.sort_values(
                by=["acq_date", "_tiene_temp", "frp"],
                ascending=[False, False, False],
            )
            .drop_duplicates(subset=["acq_date", "_lat_c", "_lon_c"])
            .drop(columns=["_tiene_temp", "_lat_c", "_lon_c"])
            .reset_index(drop=True)
        )
    else:
        df_total = _quemas_vacias()

    sensores_activos = (
        sorted(df_total["sensor_fuente"].dropna().unique().tolist())
        if not df_total.empty
        else []
    )
    telemetria = {
        "timestamp_utc": ahora_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_pet": ahora_pet.strftime("%H:%M:%S PET (%Y-%m-%d)"),
        "ventana_dias": int(dias),
        "focos_nasa_count": int(len(df_firms)),
        "alertas_serfor_count": int(len(df_serfor)),
        "total_quemas_activas": int(len(df_total)),
        "sensores_activos": sensores_activos,
        "nasa_key_configurada": bool(
            (map_key or os.environ.get("NASA_API_KEY", "")).strip()
        ),
    }
    return df_total, telemetria


def limpiar_cache_focos_tiempo_real() -> None:
    """Invalida la caché de focos térmicos NRT para forzar una consulta inmediata."""
    consultar_focos_calor_nasa_firms.clear()
    consultar_alertas_incendios_serfor_sami.clear()
