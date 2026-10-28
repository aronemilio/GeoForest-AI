"""
Módulo de Ingesta y Preparación del Dataset Especializado: GeoForest-AI
- Rasterización de polígonos validados manualmente por especialistas (Ground Truth).
- Reproyección cartográfica automática (WGS84 -> UTM Zona 18S EPSG:32718).
- Filtrado por Unidad Mínima Cartográfica (UMC >= 500 m²).
- Inclusión de pares multitemporales de estaciones opuestas (Estiaje vs. Lluvias).
- Carga y cálculo de máscara de cobertura boscosa continua para Loss Masking.
"""

from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence, Tuple, Dict, Any, List

try:
    import numpy as np
except ImportError:
    np = None

# Importaciones condicionales seguras para GIS y ML
try:
    import rasterio
    from rasterio import features as rio_features
    from rasterio.transform import Affine
    from rasterio.warp import transform_geom
except ImportError:
    rasterio = None
    rio_features = None
    Affine = None
    transform_geom = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

logger = logging.getLogger(__name__)

NORMALIZATION_DIVISOR = 10000.0
DEFAULT_TARGET_CRS = "EPSG:32718"
DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_MIN_AREA_M2 = 500.0  # Unidad Mínima Cartográfica: 5 píxeles de Sentinel-2 (10m x 10m)


def calculate_polygon_area_utm(coordinates: Sequence[Sequence[Tuple[float, float]]]) -> float:
    """
    Calcula el área métrica (en m²) de un polígono en coordenadas métricas planas (UTM)
    utilizando la fórmula del área de Gauss (Shoelace formula), considerando anillos exteriores e interiores.
    """
    if not coordinates:
        return 0.0

    def ring_area(ring: Sequence[Tuple[float, float]]) -> float:
        n = len(ring)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        return abs(area) / 2.0

    # Primer anillo es exterior; anillos subsecuentes son huecos interiores
    exterior_area = ring_area(coordinates[0])
    holes_area = sum(ring_area(h) for h in coordinates[1:]) if len(coordinates) > 1 else 0.0
    return max(0.0, exterior_area - holes_area)


class ExpertVectorMaskRasterizer:
    """
    Convierte capas vectoriales validadas manualmente (GeoJSON/SHP) en máscaras binarias ráster
    alineadas píxel a píxel con la resolución y transformación UTM de Sentinel-2 L2A (10 m/píxel).
    """

    def __init__(
        self,
        target_crs: str = DEFAULT_TARGET_CRS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
    ):
        self.target_crs = target_crs
        self.min_confidence = min_confidence
        self.min_area_m2 = min_area_m2

    def _is_geographic_coordinates(self, coords: Any) -> bool:
        """Determina heurísticamente si las coordenadas están en WGS84 grados (-180..180, -90..90)."""
        first = coords
        while isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            first = first[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            x, y = first[0], first[1]
            return (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)
        return False

    def _reproject_geometry_if_needed(self, geom: Dict[str, Any], src_crs: str | None = None) -> Dict[str, Any]:
        """Reproyecta la geometría GeoJSON hacia target_crs (EPSG:32718) si está en grados geográficos."""
        coords = geom.get("coordinates")
        if not coords:
            return geom

        is_geo = self._is_geographic_coordinates(coords)
        actual_src_crs = src_crs or ("EPSG:4326" if is_geo else self.target_crs)

        if actual_src_crs.upper() != self.target_crs.upper():
            if transform_geom is not None:
                try:
                    return transform_geom(actual_src_crs, self.target_crs, geom)
                except Exception as e:
                    logger.warning("Fallo en reproyección con transform_geom (%s). Se conserva geometría original.", e)
            else:
                logger.warning("rasterio.warp no disponible para reproyección. Se asume proyección directa.")
        return geom

    def rasterize_vector_layer(
        self,
        vector_path: str | Path,
        out_shape: Tuple[int, int],
        transform: Any,
        filter_attribute: str | None = "confianza_validacion",
        min_confidence: float | None = None,
        min_area_m2: float | None = None,
    ) -> np.ndarray:
        """
        Lee geometrías poligonales validadas, reproyecta a UTM si es necesario, filtra por UMC y confianza,
        y las rasteriza (1 = pérdida real validada, 0 = no evento).
        """
        conf_thresh = min_confidence if min_confidence is not None else self.min_confidence
        area_thresh = min_area_m2 if min_area_m2 is not None else self.min_area_m2

        path = Path(vector_path)
        if not path.is_file():
            return np.zeros(out_shape, dtype=np.uint8)

        raw_features = []
        if path.suffix.lower() in [".geojson", ".json"]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_features = data.get("features", [])
            except Exception as e:
                logger.error("Error al leer GeoJSON %s: %s", path, e)
                return np.zeros(out_shape, dtype=np.uint8)
        else:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs.to_string().upper() != self.target_crs.upper():
                    gdf = gdf.to_crs(self.target_crs)
                
                shapes = []
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    if not geom or geom.is_empty:
                        continue
                    if filter_attribute and filter_attribute in row:
                        val = row[filter_attribute]
                        if val is not None and float(val) < conf_thresh:
                            continue
                    if geom.area < area_thresh:
                        continue
                    shapes.append((geom, 1))

                if shapes and rio_features is not None:
                    return rio_features.rasterize(
                        shapes=shapes,
                        out_shape=out_shape,
                        transform=transform,
                        fill=0,
                        default_value=1,
                        dtype=np.uint8,
                        all_touched=False,
                    )
                return np.zeros(out_shape, dtype=np.uint8)
            except Exception as e:
                logger.debug("Fallback geopandas no disponible o falló: %s", e)

        # Procesamiento GeoJSON con reproyección y filtrado UMC
        valid_shapes = []
        for feat in raw_features:
            props = feat.get("properties", {})
            if filter_attribute and props.get(filter_attribute) is not None:
                try:
                    conf = float(props.get(filter_attribute, 1.0))
                    if conf < conf_thresh:
                        continue
                except (ValueError, TypeError):
                    continue

            geom = feat.get("geometry")
            if not geom or not geom.get("coordinates"):
                continue

            # Reproyectar a UTM 18S si viene en WGS84
            geom_utm = self._reproject_geometry_if_needed(geom)
            geom_type = geom_utm.get("type", "")
            coords = geom_utm.get("coordinates", [])

            # Descomponer MultiPolygons para validar UMC individual por polígono
            if geom_type == "Polygon":
                area = calculate_polygon_area_utm(coords)
                if area >= area_thresh:
                    valid_shapes.append((geom_utm, 1))
            elif geom_type == "MultiPolygon":
                for poly_coords in coords:
                    area = calculate_polygon_area_utm(poly_coords)
                    if area >= area_thresh:
                        single_poly = {"type": "Polygon", "coordinates": poly_coords}
                        valid_shapes.append((single_poly, 1))

        if not valid_shapes or rio_features is None:
            return np.zeros(out_shape, dtype=np.uint8)

        # all_touched=False: Solo píxeles cuyo centroide cae en el polígono son marcados con 1
        mask = rio_features.rasterize(
            shapes=valid_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
            all_touched=False,
        )
        return mask


class PhenologicalPairRegistry:
    """
    Registro y gestión de pares multitemporales de estaciones opuestas:
    - Estiaje (Mayo - Octubre) vs. Lluvias (Noviembre - Abril).
    - Inyecta pares de control fenológico con cambio objetivo Y = 0 garantizando
      que no existan eventos reales de deforestación confirmados en dicha ventana.
    """

    @staticmethod
    def identify_season(date_str: str) -> str:
        """
        Clasifica una fecha en Huánuco como 'estiaje' o 'lluvias'.
        En los Andes centrales y selva alta de Huánuco:
        - Estiaje: Mayo a Octubre (meses 5 al 10).
        - Lluvias: Noviembre a Abril (meses 11, 12, 1, 2, 3, 4).
        """
        try:
            parts = date_str.split("-")
            month = int(parts[1]) if len(parts) > 1 else 1
            if 5 <= month <= 10:
                return "estiaje"
            return "lluvias"
        except Exception:
            return "desconocido"

    @staticmethod
    def is_opposite_season_pair(date_t0: str, date_t1: str) -> bool:
        s0 = PhenologicalPairRegistry.identify_season(date_t0)
        s1 = PhenologicalPairRegistry.identify_season(date_t1)
        return s0 in ["estiaje", "lluvias"] and s1 in ["estiaje", "lluvias"] and s0 != s1


class ExpertForestDataset(Dataset):
    """
    Dataset especializado para GeoForest-AI:
    1. Pares Sentinel-2 L2A reales de 6 canales (T0 y T1: B04, B08, B11).
    2. Máscaras vectoriales validadas manualmente por especialistas (Ground Truth).
    3. Máscara ráster de cobertura boscosa continua para Loss Masking.
    4. Control fenológico verificado contra falsos positivos estacionales.
    """

    def __init__(
        self,
        chips_data: List[Dict[str, Any]],
        forest_cover_mask_dir: Path | str | None = None,
        is_train: bool = True,
        patch_size: int = 256,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_area_m2: float = DEFAULT_MIN_AREA_M2,
    ):
        self.chips_data = chips_data
        self.forest_mask_dir = Path(forest_cover_mask_dir) if forest_cover_mask_dir else None
        self.is_train = is_train
        self.patch_size = patch_size
        self.rasterizer = ExpertVectorMaskRasterizer(
            min_confidence=min_confidence, min_area_m2=min_area_m2
        )

    def __len__(self) -> int:
        return len(self.chips_data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        item = self.chips_data[idx]
        image_6ch = item["image_6ch"]  # np.ndarray (6, H, W)
        _, h, w = image_6ch.shape
        transform = item.get("transform", Affine.identity() if Affine is not None else None)

        # 1. Obtener máscara validada por especialista
        is_phenological_neg = item.get("is_phenological_negative", False)
        vec_path = item.get("vector_mask_path")

        target_mask = np.zeros((1, h, w), dtype=np.float32)

        if vec_path and os.path.exists(vec_path):
            raw_mask = self.rasterizer.rasterize_vector_layer(
                vec_path, out_shape=(h, w), transform=transform
            )
            # Si hay un vector verificado con eventos, tiene precedencia absoluta sobre el flag fenológico
            if raw_mask.sum() > 0:
                target_mask = np.expand_dims(raw_mask.astype(np.float32), axis=0)
            elif is_phenological_neg:
                target_mask = np.zeros((1, h, w), dtype=np.float32)
        elif is_phenological_neg:
            target_mask = np.zeros((1, h, w), dtype=np.float32)

        # 2. Obtener máscara de cobertura boscosa continua
        forest_path = item.get("forest_cover_path")
        if forest_path and os.path.exists(forest_path) and rasterio is not None:
            try:
                with rasterio.open(forest_path) as src:
                    f_mask = src.read(1, out_shape=(h, w))
                    forest_mask = (f_mask > 0).astype(np.float32)
                    forest_mask = np.expand_dims(forest_mask, axis=0)
            except Exception as e:
                logger.warning("Fallo al leer máscara de bosque %s: %s. Aplicando fallback NDVI.", forest_path, e)
                forest_mask = self._compute_ndvi_forest_mask(image_6ch)
        else:
            forest_mask = self._compute_ndvi_forest_mask(image_6ch)

        if torch is not None:
            return (
                torch.from_numpy(image_6ch).float(),
                torch.from_numpy(target_mask).float(),
                torch.from_numpy(forest_mask).float(),
            )
        return image_6ch, target_mask, forest_mask

    @staticmethod
    def _compute_ndvi_forest_mask(image_6ch: np.ndarray) -> np.ndarray:
        """Calcula máscara de bosque continuo en T0 (NDVI T0 >= 0.60 y B08 >= 0.20) como fallback."""
        b4_t0, b8_t0 = image_6ch[0], image_6ch[1]
        ndvi_t0 = (b8_t0 - b4_t0) / (b8_t0 + b4_t0 + 1e-6)
        inferred_forest = ((ndvi_t0 >= 0.60) & (b8_t0 >= 0.20)).astype(np.float32)
        return np.expand_dims(inferred_forest, axis=0)
