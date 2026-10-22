"""
Módulo de series temporales de alertas y escenarios para GeoForest-AI Huánuco.

Implementa la lógica determinista con Mulberry32 PRNG para proyectar
la evolución temporal de alertas de deforestación y quemas en el Bosque de Carpish
y el Valle del Huallaga.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd


def crear_prng(semilla_inicial: int = 42) -> Callable[[], float]:
    """Generador PRNG Mulberry32 determinista para reproducibilidad estricta.

    Produce exactamente la misma secuencia pseudoaleatoria flotante [0.0, 1.0).
    """
    s = semilla_inicial & 0xFFFFFFFF

    def aleatorio() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return float((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return aleatorio


SECTORES_CARPISH = [
    "Acomayo Alto",
    "Túnel de Carpish",
    "San Pedro de Carpish",
    "Chinchao",
    "Canchaparrosa",
]

SECTORES_HUALLAGA = [
    "Valle del Huallaga - Tomayquichua",
    "Amarilis Alto",
    "Margen Izquierda Huallaga",
    "Pillco Marca",
    "Churubamba",
]

DISTRITOS_POR_ZONA: dict[str, list[str]] = {
    "carpish": ["Chinchao", "Acomayo", "Churubamba"],
    "huallaga": ["Amarilis", "Pillco Marca", "Tomayquichua"],
}


@dataclass
class AlertaTerritorial:
    id: str
    origen: str
    fecha: str
    zona: str  # 'carpish' | 'huallaga'
    sector: str
    distrito: str
    tipo: str
    severidad: str  # 'Alta' | 'Media' | 'Baja'
    areaHa: float
    confianzaPct: float
    ndviCambio: float
    nbrCambio: float
    lat: float
    lon: float
    deltaLat: float
    deltaLon: float
    estado: str


AlertaSimulada = AlertaTerritorial


@dataclass
class PuntoSerieTemporal:
    fecha: str
    carpish: float
    huallaga: float


def generar_escenario_simulado(
    semilla: int = 42,
    pasos_semanales: int = 8,
    fecha_inicio: str = "2026-08-01",
    alertas_por_paso: int = 2,
) -> dict[str, Any]:
    """Genera alertas espaciales y series temporales de forma reproducible.

    Args:
        semilla: Semilla entera para el PRNG Mulberry32.
        pasos_semanales: Número de semanas/pasos a proyectar.
        fecha_inicio: Fecha ISO (YYYY-MM-DD) del paso 1.
        alertas_por_paso: Cantidad de nuevas alertas añadidas en cada paso semanal.

    Returns:
        Diccionario con:
          - 'alertas': lista de dicts de alertas territoriales.
          - 'serie': lista de dicts con puntos acumulados {fecha, carpish, huallaga}.
    """
    rand = crear_prng(semilla)
    alertas: list[dict[str, Any]] = []
    serie: list[dict[str, Any]] = []

    id_counter = 100
    fecha_actual = datetime.strptime(fecha_inicio, "%Y-%m-%d")

    acumulado_carpish = 0.0
    acumulado_huallaga = 0.0

    for _ in range(pasos_semanales):
        fecha_str = fecha_actual.strftime("%Y-%m-%d")

        area_paso_carpish = 0.0
        area_paso_huallaga = 0.0

        for _ in range(alertas_por_paso):
            id_counter += 1
            zona = "carpish" if rand() > 0.45 else "huallaga"
            sev_rand = rand()
            severidad = (
                "Alta"
                if sev_rand > 0.65
                else ("Media" if sev_rand > 0.3 else "Baja")
            )

            tipos_posibles = (
                ["Degradación de bosque nuboso", "Deforestación"]
                if zona == "carpish"
                else ["Quema agrícola", "Deforestación"]
            )
            tipo = tipos_posibles[math.floor(rand() * len(tipos_posibles))]

            sectores = SECTORES_CARPISH if zona == "carpish" else SECTORES_HUALLAGA
            sector = sectores[math.floor(rand() * len(sectores))]
            distritos = DISTRITOS_POR_ZONA[zona]
            distrito = distritos[math.floor(rand() * len(distritos))]

            lat = (
                -9.82 + (rand() - 0.5) * 0.12
                if zona == "carpish"
                else -9.95 + (rand() - 0.5) * 0.14
            )
            lon = (
                -76.08 + (rand() - 0.5) * 0.10
                if zona == "carpish"
                else -76.22 + (rand() - 0.5) * 0.12
            )

            max_area = 12.0 if severidad == "Alta" else 5.0
            area_ha = round(0.8 + rand() * max_area, 1)
            ndvi_cambio = round(-0.15 - rand() * 0.45, 2)
            nbr_cambio = round(-0.12 - rand() * 0.50, 2)
            confianza_pct = round(82.0 + rand() * 16.0, 1)

            estados = [
                "Alerta activa",
                "En verificación",
                "Monitoreo preventivo",
            ]
            estado = estados[math.floor(rand() * len(estados))]

            delta_lat = round(0.003 + rand() * 0.005, 4)
            delta_lon = round(0.003 + rand() * 0.005, 4)

            alerta = AlertaTerritorial(
                id=f"GF-ATD-{id_counter}",
                origen="GEOBOSQUES",
                fecha=fecha_str,
                zona=zona,
                sector=sector,
                distrito=distrito,
                tipo=tipo,
                severidad=severidad,
                areaHa=area_ha,
                confianzaPct=confianza_pct,
                ndviCambio=ndvi_cambio,
                nbrCambio=nbr_cambio,
                lat=lat,
                lon=lon,
                deltaLat=delta_lat,
                deltaLon=delta_lon,
                estado=estado,
            )
            alertas.append(asdict(alerta))

            if zona == "carpish":
                area_paso_carpish += area_ha
            else:
                area_paso_huallaga += area_ha

        acumulado_carpish += area_paso_carpish
        acumulado_huallaga += area_paso_huallaga

        serie.append(
            asdict(
                PuntoSerieTemporal(
                    fecha=fecha_str,
                    carpish=round(acumulado_carpish, 1),
                    huallaga=round(acumulado_huallaga, 1),
                )
            )
        )

        fecha_actual += timedelta(days=7)

    return {
        "alertas": alertas,
        "serie": serie,
    }


generar_escenario_territorial = generar_escenario_simulado


def convertir_alertas_a_dataframe(alertas: list[dict[str, Any]]) -> pd.DataFrame:
    """Adapta el formato de alertas al DataFrame estándar consumido por app/main.py."""
    if not alertas:
        return pd.DataFrame()

    mapeo_zonas = {
        "carpish": "Bosque de Carpish",
        "huallaga": "Valle del Huallaga",
    }

    filas = []
    for a in alertas:
        filas.append({
            "id_alerta": a["id"],
            "origen_dato": a.get("origen", "GEOBOSQUES"),
            "fecha": a["fecha"],
            "zona": mapeo_zonas.get(a["zona"], a["zona"]),
            "sector": a["sector"],
            "distrito": a["distrito"],
            "tipo_evento": a["tipo"],
            "severidad": a["severidad"],
            "area_ha": a["areaHa"],
            "confianza_pct": a["confianzaPct"],
            "ndvi_cambio": a["ndviCambio"],
            "nbr_cambio": a.get("nbrCambio", -0.2),
            "lat": a["lat"],
            "lon": a["lon"],
            "delta_lat": a["deltaLat"],
            "delta_lon": a["deltaLon"],
            "estado": a["estado"],
        })

    return pd.DataFrame(filas)
