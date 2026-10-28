#!/usr/bin/env python3
"""
Pruebas Unitarias del Pipeline de Anotación y Validación Técnica: GeoForest-AI
"""

import unittest
import json
import tempfile
from pathlib import Path

try:
    from src.validate_annotations import (
        calculate_polygon_area_utm,
        estimate_polygon_area_degrees_to_m2,
        validate_feature,
        validate_geojson_file,
    )
    from src.expert_dataset import (
        PhenologicalPairRegistry,
        ExpertVectorMaskRasterizer,
    )
except ImportError:
    from geoforest_validate_annotations import (
        calculate_polygon_area_utm,
        estimate_polygon_area_degrees_to_m2,
        validate_feature,
        validate_geojson_file,
    )
    from geoforest_expert_dataset import (
        PhenologicalPairRegistry,
        ExpertVectorMaskRasterizer,
    )


class TestAnnotationPipeline(unittest.TestCase):

    def test_utm_polygon_area_square(self):
        # Cuadrado de 100m x 100m en UTM = 10,000 m² (1 hectárea)
        coords = [
            [[300000.0, 8900000.0], [300100.0, 8900000.0], [300100.0, 8900100.0], [300000.0, 8900100.0], [300000.0, 8900000.0]]
        ]
        area = calculate_polygon_area_utm(coords)
        self.assertAlmostEqual(area, 10000.0, places=1)

    def test_utm_polygon_area_sub_umc(self):
        # Parche pequeño de 10m x 20m = 200 m² (< 500 m² UMC)
        coords = [
            [[300000.0, 8900000.0], [300010.0, 8900000.0], [300010.0, 8900020.0], [300000.0, 8900020.0], [300000.0, 8900000.0]]
        ]
        area = calculate_polygon_area_utm(coords)
        self.assertAlmostEqual(area, 200.0, places=1)
        self.assertLess(area, 500.0)

    def test_phenological_season_identification(self):
        # Meses de estiaje en Huánuco: Mayo a Octubre (5 a 10)
        self.assertEqual(PhenologicalPairRegistry.identify_season("2023-07-15"), "estiaje")
        self.assertEqual(PhenologicalPairRegistry.identify_season("2023-09-01"), "estiaje")
        # Meses de lluvias: Noviembre a Abril (11, 12, 1, 2, 3, 4)
        self.assertEqual(PhenologicalPairRegistry.identify_season("2023-01-20"), "lluvias")
        self.assertEqual(PhenologicalPairRegistry.identify_season("2023-12-10"), "lluvias")

        # Par opuesto estacional
        self.assertTrue(PhenologicalPairRegistry.is_opposite_season_pair("2023-07-15", "2024-01-20"))
        # Par intra-estacional (ambos estiaje)
        self.assertFalse(PhenologicalPairRegistry.is_opposite_season_pair("2023-06-10", "2023-08-25"))

    def test_feature_validation_valid(self):
        # Polígono válido de 50m x 50m = 2,500 m² en UTM
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[300000.0, 8900000.0], [300050.0, 8900000.0], [300050.0, 8900050.0], [300000.0, 8900050.0], [300000.0, 8900000.0]]
                ]
            },
            "properties": {
                "evento": "deforestacion",
                "confianza_validacion": 0.95,
                "fecha_t0": "2023-06-01",
                "fecha_t1": "2023-09-15",
                "agente_causante": "expansion_agricola",
                "validador": "ING_SOTO"
            }
        }
        is_valid, errors, diag = validate_feature(feature, 0, min_confidence=0.8, min_area_m2=500.0)
        self.assertTrue(is_valid, f"Errores encontrados: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertEqual(diag["status"], "VALIDO")

    def test_feature_validation_sub_umc_rejected(self):
        # Polígono sub-UMC de 10m x 20m = 200 m²
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[300000.0, 8900000.0], [300010.0, 8900000.0], [300010.0, 8900020.0], [300000.0, 8900020.0], [300000.0, 8900000.0]]
                ]
            },
            "properties": {
                "evento": "deforestacion",
                "confianza_validacion": 0.95,
                "fecha_t0": "2023-06-01",
                "fecha_t1": "2023-09-15"
            }
        }
        is_valid, errors, diag = validate_feature(feature, 0, min_confidence=0.8, min_area_m2=500.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("Unidad Mínima Cartográfica" in err for err in errors))

    def test_feature_validation_low_confidence_rejected(self):
        # Polígono con confianza 0.60 (< 0.80)
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[300000.0, 8900000.0], [300100.0, 8900000.0], [300100.0, 8900100.0], [300000.0, 8900100.0], [300000.0, 8900000.0]]
                ]
            },
            "properties": {
                "evento": "deforestacion",
                "confianza_validacion": 0.60,
                "fecha_t0": "2023-06-01",
                "fecha_t1": "2023-09-15"
            }
        }
        is_valid, errors, diag = validate_feature(feature, 0, min_confidence=0.8, min_area_m2=500.0)
        self.assertFalse(is_valid)
        self.assertTrue(any("menor al umbral" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
