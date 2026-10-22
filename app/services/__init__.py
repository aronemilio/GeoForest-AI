"""Servicios de datos, APIs satelitales y procesamiento."""

from app.services.alerts_data import (
    construir_poligono_alerta,
    generar_escenario_geobosques,
    generar_geojson_alertas,
    generar_recorte_offline_huanuco,
    obtener_alertas_reales_geobosques,
)
from app.services.copernicus import (
    CATALOGO_ESCENAS_REALES,
    cargar_escena_satelital_seleccionada,
    cargar_escena_sentinel2_real,
    consultar_catalogo_copernicus,
    descargar_banda_sentinel2,
    descargar_escena_sentinel2,
    descargar_par_sentinel2,
    sanitize_scene_id,
    validate_cog_url,
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
    "CATALOGO_ESCENAS_REALES",
    "cargar_escena_satelital_seleccionada",
    "cargar_escena_sentinel2_real",
    "consultar_catalogo_copernicus",
    "descargar_banda_sentinel2",
    "descargar_escena_sentinel2",
    "descargar_par_sentinel2",
    "sanitize_scene_id",
    "validate_cog_url",
    "obtener_alertas_reales_geobosques",
    "generar_escenario_geobosques",
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
