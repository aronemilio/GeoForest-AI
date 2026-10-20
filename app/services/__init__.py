"""Servicios de datos, APIs satelitales y procesamiento."""

from app.services.alerts_data import (
    construir_poligono_alerta,
    generar_geojson_alertas,
    generar_recorte_offline_huanuco,
    obtener_alertas_simuladas,
)
from app.services.expediente import generar_expediente_fema_serfor
from app.services.nasa_firms import consultar_focos_calor_nasa_firms
from app.services.onnx_demo import (
    generar_parche_demostracion,
    matriz_a_imagen_heatmap,
    matriz_a_imagen_mascara,
    matriz_a_imagen_ndvi,
    matriz_a_imagen_rgb,
    peso_modelo_onnx_mb,
)

__all__ = [
    "obtener_alertas_simuladas",
    "construir_poligono_alerta",
    "generar_geojson_alertas",
    "generar_recorte_offline_huanuco",
    "consultar_focos_calor_nasa_firms",
    "generar_expediente_fema_serfor",
    "peso_modelo_onnx_mb",
    "generar_parche_demostracion",
    "matriz_a_imagen_rgb",
    "matriz_a_imagen_ndvi",
    "matriz_a_imagen_heatmap",
    "matriz_a_imagen_mascara",
]
