"""Componentes de interfaz y vistas de GeoForest-AI Huánuco."""

from app.components.map_view import crear_mapa_monitoreo
from app.components.tab_benchmark import render_tab_benchmark
from app.components.tab_inferencia import render_tab_laboratorio_inferencia
from app.components.tab_pitch import render_presentacion_unificada, render_tab_pitch
from app.components.tab_sig import render_consola_sig

__all__ = [
    "crear_mapa_monitoreo",
    "render_tab_benchmark",
    "render_tab_laboratorio_inferencia",
    "render_tab_pitch",
    "render_presentacion_unificada",
    "render_consola_sig",
]
