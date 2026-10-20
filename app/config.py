"""Configuración y constantes globales de GeoForest-AI Huánuco."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

# Constantes geográficas y zonas de estudio (Huánuco, Perú)
ZONAS_ESTUDIO: dict[str, dict[str, Any]] = {
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

COLORES_SEVERIDAD: dict[str, str] = {
    "Alta": "#d73027",
    "Media": "#fc8d59",
    "Baja": "#fee08b",
}
