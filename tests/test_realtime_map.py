"""Pruebas unitarias e integración para el monitoreo en tiempo real (pérdida de cobertura y quemas)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.components.map_view import crear_mapa_monitoreo
from app.services.alerts_data import (
    GEOBOSQUES_ATD_WMS_URL,
    consultar_alertas_geobosques_api_vivo,
    cruzar_alertas_con_focos_termicos,
    limpiar_cache_alertas_tiempo_real,
)
from app.services.copernicus import (
    REGLA_CAMBIO_ESPECTRAL_S2,
    construir_overlay_espectral_sentinel2,
)
from app.services.nasa_firms import (
    COLUMNAS_QUEMAS_VIVO,
    consultar_alertas_incendios_serfor_sami,
    consultar_focos_calor_nasa_firms,
    limpiar_cache_focos_tiempo_real,
    obtener_quemas_tiempo_real_integradas,
)


def _escribir_parche_s2_utm(
    scene_dir: Path,
    b04: np.ndarray,
    b08: np.ndarray,
    b11: np.ndarray,
    x_min: float = 378930.0,
    y_max: float = 8927960.0,
) -> None:
    """Escribe un directorio de escena Sentinel-2 L2A sintético en EPSG:32718 (DN uint16, escala 10000)."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    h, w = b04.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint16",
        "crs": CRS.from_epsg(32718),
        "transform": from_origin(x_min, y_max, 10.0, 10.0),
    }
    for nombre, matriz in (("B04.tif", b04), ("B08.tif", b08), ("B11.tif", b11)):
        with rasterio.open(scene_dir / nombre, "w", **profile) as dst:
            dst.write(matriz.astype(np.uint16), 1)


def test_nasa_firms_sin_api_key_devuelve_vacio_sin_relleno() -> None:
    """Verifica que sin NASA_API_KEY se devuelva un DataFrame vacío tipado sin valores ficticios."""
    limpiar_cache_focos_tiempo_real()
    df = consultar_focos_calor_nasa_firms(map_key="", dias=5)
    assert df.empty
    assert tuple(df.columns) == COLUMNAS_QUEMAS_VIVO


def test_nasa_firms_acota_dias_nrt_y_normaliza_viirs_y_modis() -> None:
    """Verifica el clamping estricto de días NRT a [1, 5] y la normalización radiométrica de VIIRS y MODIS."""
    limpiar_cache_focos_tiempo_real()
    csv_viirs = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
        "-9.7118,-76.0942,338.15,0.4,0.4,2026-11-01,1830,N20,VIIRS,h,2.0NRT,295.0,18.4,D\n"
    )
    csv_modis = (
        "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
        "-9.5845,-76.0465,325.15,1.0,1.0,2026-11-01,1545,Terra,MODIS,82,6.1NRT,292.0,24.7,D\n"
    )
    urls_consultadas: list[str] = []

    def _mock_get(url: str, timeout: int = 7) -> MagicMock:
        urls_consultadas.append(url)
        resp = MagicMock()
        resp.status_code = 200
        if "VIIRS_NOAA20_NRT" in url:
            resp.text = csv_viirs
        elif "MODIS_NRT" in url:
            resp.text = csv_modis
        else:
            resp.text = "latitude,longitude\n"
        return resp

    with patch("app.services.nasa_firms.requests.get", side_effect=_mock_get):
        df = consultar_focos_calor_nasa_firms(
            map_key="1234567890abcdef1234567890abcdef",
            dias=12,  # Debe acotarse a 5 en NRT para evitar HTTP 400
            sensores=("VIIRS_NOAA20_NRT", "MODIS_NRT"),
        )

    assert len(df) == 2
    assert all(u.endswith("/5") for u in urls_consultadas)
    assert set(df["zona"]) == {"Bosque de Carpish", "Valle del Huallaga"}
    fila_viirs = df[df["instrument"] == "VIIRS"].iloc[0]
    assert np.isclose(fila_viirs["bright_ti4"], 338.15)
    assert np.isclose(fila_viirs["bright_c"], 65.0)
    assert "13:30 PET" in fila_viirs["fecha_hora_pet"]


def test_serfor_sami_normaliza_alertas_sin_inventar_temperatura() -> None:
    """Verifica que SERFOR SAMI mantenga NaN en temperatura si el feed oficial no entrega telemetría Kelvin."""
    limpiar_cache_focos_tiempo_real()
    ahora_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - 3600_000

    def _mock_get(url: str, params: dict | None = None, timeout: int = 7) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        if "/0/query" in url:
            resp.json.return_value = {
                "features": [
                    {
                        "attributes": {"FECREG": ahora_ms, "NOMDEP": "HUANUCO"},
                        "geometry": {"x": -76.0910, "y": -9.7125},
                    }
                ]
            }
        else:
            resp.json.return_value = {"features": []}
        return resp

    with patch("app.services.nasa_firms.requests.get", side_effect=_mock_get):
        df = consultar_alertas_incendios_serfor_sami(dias=3)

    assert len(df) == 1
    assert df.iloc[0]["origen_feed"] == "SERFOR_SAMI_ALERTA"
    assert df.iloc[0]["zona"] == "Bosque de Carpish"
    # Rigor de dominio: SERFOR SAMI no entrega Kelvin en esta capa; prohibido rellenar con 310 K
    assert pd.isna(df.iloc[0]["bright_ti4"])
    assert pd.isna(df.iloc[0]["bright_c"])


def test_obtener_quemas_integradas_deduplica_priorizando_telemetria_radiometrica() -> None:
    """Verifica la deduplicación espacial de quemas priorizando registros con temperatura física y FRP."""
    limpiar_cache_focos_tiempo_real()
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df_firms_mock = pd.DataFrame(
        [
            {
                "latitude": -9.58431,
                "longitude": -76.04632,
                "bright_ti4": 341.2,
                "bright_c": 68.1,
                "frp": 19.5,
                "acq_date": hoy,
                "acq_time": "1820",
                "fecha_hora_utc": f"{hoy} 18:20 UTC",
                "fecha_hora_pet": f"{hoy} 13:20 PET",
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "nominal",
                "daynight": "D",
                "sensor_fuente": "VIIRS NOAA-20 (375m)",
                "origen_feed": "NASA_FIRMS_VIIRS_NOAA20_NRT",
                "zona": "Valle del Huallaga",
            }
        ]
    )
    df_serfor_mock = pd.DataFrame(
        [
            {
                "latitude": -9.58429,
                "longitude": -76.04628,
                "bright_ti4": np.nan,
                "bright_c": np.nan,
                "frp": 0.0,
                "acq_date": hoy,
                "acq_time": "1800",
                "fecha_hora_utc": f"{hoy} 18:00 UTC",
                "fecha_hora_pet": f"{hoy} 13:00 PET",
                "satellite": "UFMS Satelital",
                "instrument": "SAMI-UFMS",
                "confidence": "oficial_serfor",
                "daynight": "",
                "sensor_fuente": "SERFOR SAMI — Foco de Calor Satelital",
                "origen_feed": "SERFOR_SAMI_FOCO",
                "zona": "Valle del Huallaga",
            }
        ]
    )

    with (
        patch("app.services.nasa_firms.consultar_focos_calor_nasa_firms", return_value=df_firms_mock),
        patch("app.services.nasa_firms.consultar_alertas_incendios_serfor_sami", return_value=df_serfor_mock),
    ):
        df_total, telem = obtener_quemas_tiempo_real_integradas(
            map_key="1234567890abcdef1234567890abcdef", dias=5
        )

    assert len(df_total) == 1
    assert np.isclose(df_total.iloc[0]["bright_ti4"], 341.2)
    assert telem["focos_nasa_count"] == 1
    assert telem["alertas_serfor_count"] == 1
    assert telem["total_quemas_activas"] == 1


def test_consultar_alertas_geobosques_api_vivo_y_cruce_espacial_quemas() -> None:
    """Verifica la consulta en vivo de la API Geobosques 2026 y el cruce espacial Haversine con focos térmicos."""
    limpiar_cache_alertas_tiempo_real()
    fec_ms = int(datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)

    resp_mock = MagicMock()
    resp_mock.status_code = 200
    resp_mock.json.return_value = {
        "features": [
            {
                "attributes": {"objectid": 101, "fe_alerta": fec_ms, "ubigeo": "100102"},
                "geometry": {"x": -76.0942, "y": -9.7118},
            },
            {
                "attributes": {"objectid": 102, "fe_alerta": fec_ms, "ubigeo": "100102"},
                "geometry": {"x": -76.0941, "y": -9.7119},
            },
            {
                "attributes": {"objectid": 201, "fe_alerta": fec_ms, "ubigeo": "100111"},
                "geometry": {"x": -76.2100, "y": -9.8500},
            },
        ]
    }

    with patch("app.services.alerts_data.requests.get", return_value=resp_mock):
        df_atd = consultar_alertas_geobosques_api_vivo(max_puntos=100)

    assert len(df_atd) == 2
    assert set(df_atd["origen_dato"]) == {"PNCB_GEOBOSQUES_API_VIVO"}

    # Foco térmico activo a < 200 m del clúster de Carpish (-9.7118, -76.0942)
    df_quemas = pd.DataFrame([{"latitude": -9.7120, "longitude": -76.0940}])
    df_cruzado = cruzar_alertas_con_focos_termicos(df_atd, df_quemas, radio_km=1.5)

    carpish_row = df_cruzado[df_cruzado["zona"] == "Bosque de Carpish"].iloc[0]
    lejano_row = df_cruzado[df_cruzado["zona"] != "Bosque de Carpish"].iloc[0]
    assert carpish_row["tipo_evento"] == "Pérdida de Cobertura + Quema Activa Cercana"
    assert "Foco térmico activo" in carpish_row["estado"]
    assert lejano_row["tipo_evento"] == "Deforestación Antrópica"


def test_construir_overlay_espectral_sentinel2_sin_cambio_reporta_cero_exacto(
    tmp_path: Path,
) -> None:
    """Verifica que un par multitemporal sin alteración espectral reporte estrictamente 0.00 ha."""
    shape = (64, 64)
    b04 = np.full(shape, 400, dtype=np.uint16)   # 0.04
    b08 = np.full(shape, 3200, dtype=np.uint16)  # 0.32 -> NDVI ~ 0.78
    b11 = np.full(shape, 1200, dtype=np.uint16)  # 0.12 -> NBR  ~ 0.45

    t0_dir = tmp_path / "2026-08-12_S2C_18LUQ_20260812_0_L2A"
    t1_dir = tmp_path / "2026-10-16_S2B_18LUQ_20261016_0_L2A"
    _escribir_parche_s2_utm(t0_dir, b04, b08, b11)
    _escribir_parche_s2_utm(t1_dir, b04, b08, b11)

    ov = construir_overlay_espectral_sentinel2(
        t0_dir, t1_dir, zona_titulo="Carpish Test", zona_slug="carpish"
    )
    assert ov["es_satelital_real"] is True
    assert ov["px_perdida_cobertura"] == 0
    assert ov["px_quema_espectral"] == 0
    assert ov["area_perdida_cobertura_ha"] == 0.0
    assert ov["area_quema_espectral_ha"] == 0.0
    assert int(ov["mask_rgba"][..., 3].max()) == 0


def test_construir_overlay_espectral_sentinel2_separa_perdida_cobertura_y_quema(
    tmp_path: Path,
) -> None:
    """Verifica que la LabelRule física distinga colapso del dosel (ΔNDVI) de cicatriz de quema (ΔNBR/SWIR)."""
    shape = (80, 80)
    b04_0 = np.full(shape, 400, dtype=np.uint16)
    b08_0 = np.full(shape, 3400, dtype=np.uint16)
    b11_0 = np.full(shape, 1200, dtype=np.uint16)

    b04_1 = b04_0.copy()
    b08_1 = b08_0.copy()
    b11_1 = b11_0.copy()

    # Parche 1 (10x10 = 100 px = 1.00 ha): Pérdida de cobertura sin incremento fuerte SWIR
    b04_1[10:20, 10:20] = 1800
    b08_1[10:20, 10:20] = 1700
    b11_1[10:20, 10:20] = 1350  # ΔB11 = +0.015 (< 0.04 umbral de quema)

    # Parche 2 (12x10 = 120 px = 1.20 ha): Cicatriz de quema (colapso NIR + alza fuerte SWIR B11)
    b04_1[45:57, 45:55] = 1600
    b08_1[45:57, 45:55] = 1100
    b11_1[45:57, 45:55] = 2600  # ΔB11 = +0.14 (>= 0.04) y ΔNBR << -0.18

    t0_dir = tmp_path / "2026-08-12_S2C_18LUQ_20260812_0_L2A"
    t1_dir = tmp_path / "2026-09-23_S2A_18LUQ_20260923_0_L2A"
    _escribir_parche_s2_utm(t0_dir, b04_0, b08_0, b11_0)
    _escribir_parche_s2_utm(t1_dir, b04_1, b08_1, b11_1)

    ov = construir_overlay_espectral_sentinel2(
        t0_dir,
        t1_dir,
        zona_titulo="Bosque Montano de Carpish",
        zona_slug="carpish",
        rule=REGLA_CAMBIO_ESPECTRAL_S2,
    )

    assert ov["px_perdida_cobertura"] == 100
    assert ov["px_quema_espectral"] == 120
    assert np.isclose(ov["area_perdida_cobertura_ha"], 1.00)
    assert np.isclose(ov["area_quema_espectral_ha"], 1.20)
    assert "bilinear" in ov["resampling_continuo"]
    assert "nearest" in ov["resampling_categorico"]
    assert "ΔNDVI" in ov["label_rule"] and "ΔNBR" in ov["label_rule"]

    # Verificar que el remuestreo categórico nearest no inventó colores mezclados
    mask_rgba = ov["mask_rgba"]
    activos = mask_rgba[mask_rgba[..., 3] > 0]
    colores_unicos = {tuple(int(x) for x in px) for px in activos}
    assert colores_unicos == {(220, 38, 38, 220), (255, 102, 0, 230)}


def test_crear_mapa_monitoreo_renderiza_capas_en_tiempo_real_y_respeta_dominio(
    tmp_path: Path,
) -> None:
    """Verifica que el mapa renderice capas NRT, overlays Sentinel-2 reales y respete las restricciones de dominio."""
    shape = (48, 48)
    b04_0 = np.full(shape, 400, dtype=np.uint16)
    b08_0 = np.full(shape, 3400, dtype=np.uint16)
    b11_0 = np.full(shape, 1200, dtype=np.uint16)
    b04_1 = b04_0.copy()
    b08_1 = b08_0.copy()
    b11_1 = b11_0.copy()
    b04_1[10:18, 10:18] = 1700
    b08_1[10:18, 10:18] = 1200
    b11_1[10:18, 10:18] = 2400

    t0_dir = tmp_path / "2026-10-01_S2C_18LUQ_20261001_0_L2A"
    t1_dir = tmp_path / "2026-10-16_S2B_18LUQ_20261016_0_L2A"
    _escribir_parche_s2_utm(t0_dir, b04_0, b08_0, b11_0)
    _escribir_parche_s2_utm(t1_dir, b04_1, b08_1, b11_1)
    ov = construir_overlay_espectral_sentinel2(t0_dir, t1_dir, zona_titulo="Huallaga", zona_slug="huallaga")

    df_alertas = pd.DataFrame(
        [
            {
                "id_alerta": "PNCB-LIVE-0001",
                "origen_dato": "PNCB_GEOBOSQUES_API_VIVO",
                "fecha": "2026-10-20",
                "zona": "Bosque de Carpish",
                "sector": "Sector Túnel de Carpish",
                "distrito": "Chinchao",
                "tipo_evento": "Pérdida de Cobertura + Quema Activa Cercana",
                "severidad": "Alta",
                "area_ha": 1.85,
                "ndvi_cambio": -0.34,
                "confianza_pct": 88.0,
                "lat": -9.7118,
                "lon": -76.0942,
                "estado": "Alerta Oficial En Vivo (API Geobosques 2026) · Foco térmico activo a 0.25 km",
                "poligono": [
                    [-9.7110, -76.0950],
                    [-9.7110, -76.0934],
                    [-9.7126, -76.0934],
                    [-9.7126, -76.0950],
                ],
            }
        ]
    )

    df_quemas = pd.DataFrame(
        [
            {
                "latitude": -9.7120,
                "longitude": -76.0940,
                "bright_ti4": 342.4,
                "bright_c": 69.3,
                "frp": 21.5,
                "acq_date": "2026-11-01",
                "acq_time": "1849",
                "fecha_hora_utc": "2026-11-01 18:49 UTC",
                "fecha_hora_pet": "2026-11-01 13:49 PET",
                "satellite": "N20",
                "instrument": "VIIRS",
                "confidence": "nominal",
                "daynight": "D",
                "sensor_fuente": "VIIRS NOAA-20 (375m)",
                "origen_feed": "NASA_FIRMS_VIIRS_NOAA20_NRT",
                "zona": "Bosque de Carpish",
            },
            {
                "latitude": -9.5850,
                "longitude": -76.0460,
                "bright_ti4": np.nan,
                "bright_c": np.nan,
                "frp": 0.0,
                "acq_date": "2026-11-01",
                "acq_time": "1900",
                "fecha_hora_utc": "2026-11-01 24:00 UTC",
                "fecha_hora_pet": "2026-11-01 19:00 PET",
                "satellite": "UFMS Satelital",
                "instrument": "SAMI-UFMS",
                "confidence": "oficial_serfor",
                "daynight": "",
                "sensor_fuente": "SERFOR SAMI — Alerta de Incendio Forestal",
                "origen_feed": "SERFOR_SAMI_ALERTA",
                "zona": "Valle del Huallaga",
            },
        ]
    )

    mapa = crear_mapa_monitoreo(
        df_filtrado=df_alertas,
        zona_seleccionada="Todas las zonas",
        mapa_base="Satélite real (Esri World Imagery)",
        mostrar_zonas=True,
        mostrar_poligonos_alerta=True,
        mostrar_marcadores=True,
        mostrar_nasa_fires=True,
        df_nasa_firms=df_quemas,
        mostrar_sentinel_copernicus=True,
        overlays_sentinel2_reales=[ov],
        mostrar_wms_geobosques=True,
        telemetria_tiempo_real={
            "sync_pet": "00:25:00 PET (2026-11-02)",
            "alertas_visibles": 1,
            "quemas_activas": 2,
        },
        vista_satelital_activa="t1",
    )

    html = mapa.get_root().render()
    html_legible = html.encode("utf-8").decode("unicode_escape", errors="ignore")
    assert GEOBOSQUES_ATD_WMS_URL in html
    assert "Cambio Espectral Real" in html and "NDVI" in html and "NBR" in html
    assert "LabelRule" in html
    assert "gf-fire-dot" in html
    assert "342.4 K" in html
    assert "No reportada en este feed vectorial" in html
    assert "310.0 K" not in html
    assert "EN VIVO" in html
    assert "no certifica tala ilegal" in html
    assert "Certeza IA" not in html and "Certeza IA" not in html_legible
