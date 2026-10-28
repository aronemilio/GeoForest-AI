#!/usr/bin/env python3
"""
Herramienta de Control de Calidad y Validación de Anotaciones Vectoriales: GeoForest-AI
Verifica:
1. Validez estructural y topológica de geometrías (Polygon / MultiPolygon).
2. Esquema de atributos obligatorios y umbrales de confianza (>= 0.80).
3. Cumplimiento de la Unidad Mínima Cartográfica (UMC >= 500 m²).
4. Detección y diagnóstico de Sistema de Referencia de Coordenadas (CRS).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

MANDATORY_ATTRIBUTES = {
    "evento": (str, ["deforestacion", "quema_agricola", "tala_selectiva", "degradacion"]),
    "confianza_validacion": (float, (0.0, 1.0)),
    "fecha_t0": (str, None),
    "fecha_t1": (str, None),
}

DEFAULT_MIN_CONFIDENCE = 0.80
DEFAULT_MIN_AREA_M2 = 500.0


def calculate_polygon_area_utm(coordinates: List[List[List[float]]]) -> float:
    """Calcula el área en m² de un polígono en coordenadas métricas UTM (Shoelace formula)."""
    if not coordinates:
        return 0.0

    def ring_area(ring: List[List[float]]) -> float:
        n = len(ring)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        return abs(area) / 2.0

    ext_area = ring_area(coordinates[0])
    holes = sum(ring_area(h) for h in coordinates[1:]) if len(coordinates) > 1 else 0.0
    return max(0.0, ext_area - holes)


def estimate_polygon_area_degrees_to_m2(coordinates: List[List[List[float]]], ref_lat: float = -9.93) -> float:
    """
    Convierte aproximadamente coordenadas WGS84 en grados a metros cuadrados para Huánuco (~lat -9.93°).
    1 grado latitud ≈ 110,574 m.
    1 grado longitud ≈ 111,320 * cos(lat) ≈ 109,655 m.
    """
    if not coordinates:
        return 0.0
    m_per_deg_lat = 110574.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    def project_ring(ring: List[List[float]]) -> List[List[float]]:
        return [[pt[0] * m_per_deg_lon, pt[1] * m_per_deg_lat] for pt in ring]

    projected_coords = [project_ring(ring) for ring in coordinates]
    return calculate_polygon_area_utm(projected_coords)


def validate_feature(
    feature: Dict[str, Any],
    feat_index: int,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Valida una entidad vectorial individual y retorna (es_valido, errores, diagnostico)."""
    errors: List[str] = []
    diag: Dict[str, Any] = {"index": feat_index, "status": "OK"}

    geom = feature.get("geometry")
    if not geom:
        errors.append("Geometría ausente o nula.")
        return False, errors, diag

    g_type = geom.get("type")
    if g_type not in ["Polygon", "MultiPolygon"]:
        errors.append(f"Tipo de geometría no admitido: {g_type}. Se requiere Polygon o MultiPolygon.")
        return False, errors, diag

    coords = geom.get("coordinates", [])
    if not coords:
        errors.append("Lista de coordenadas vacía.")
        return False, errors, diag

    # Detectar si las coordenadas son geográficas (grados) o proyectadas (UTM)
    sample_pt = coords[0][0] if g_type == "Polygon" else coords[0][0][0]
    is_geo = (-180.0 <= sample_pt[0] <= 180.0) and (-90.0 <= sample_pt[1] <= 90.0)
    diag["is_geographic"] = is_geo

    # Cálculo de área métrica
    if g_type == "Polygon":
        area = estimate_polygon_area_degrees_to_m2(coords) if is_geo else calculate_polygon_area_utm(coords)
    else:
        area = sum(
            (estimate_polygon_area_degrees_to_m2(poly) if is_geo else calculate_polygon_area_utm(poly))
            for poly in coords
        )
    diag["area_m2"] = area

    if area < min_area_m2:
        errors.append(f"Área {area:.1f} m² inferior a la Unidad Mínima Cartográfica ({min_area_m2:.1f} m²).")

    # Validación de atributos
    props = feature.get("properties") or {}
    diag["evento"] = props.get("evento", "no_definido")

    conf = props.get("confianza_validacion")
    if conf is None:
        errors.append("Falta atributo obligatorio 'confianza_validacion'.")
    else:
        try:
            conf_val = float(conf)
            diag["confianza"] = conf_val
            if conf_val < min_confidence:
                errors.append(f"Confianza {conf_val:.2f} menor al umbral de entrenamiento ({min_confidence:.2f}).")
        except (ValueError, TypeError):
            errors.append(f"Valor no numérico en 'confianza_validacion': {conf}")

    for req_attr, (expected_type, allowed_vals) in MANDATORY_ATTRIBUTES.items():
        if req_attr == "confianza_validacion":
            continue
        val = props.get(req_attr)
        if val is None:
            errors.append(f"Falta atributo obligatorio: '{req_attr}'.")
        elif allowed_vals and isinstance(allowed_vals, list) and val.lower() not in allowed_vals:
            errors.append(f"Valor '{val}' no permitido en '{req_attr}'. Permitidos: {allowed_vals}")

    is_valid = len(errors) == 0
    diag["status"] = "VALIDO" if is_valid else "OBSERVADO"
    return is_valid, errors, diag


def validate_geojson_file(
    file_path: Path, min_confidence: float = DEFAULT_MIN_CONFIDENCE, min_area_m2: float = DEFAULT_MIN_AREA_M2
) -> Dict[str, Any]:
    print(f"\n-> Analizando archivo: {file_path.name}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERROR FATAL] No se pudo leer el archivo JSON: {e}")
        return {"error": str(e), "total": 0, "valid": 0}

    features = data.get("features", [])
    total = len(features)
    valid_count = 0
    sub_umc_count = 0
    low_conf_count = 0
    attr_err_count = 0

    for i, feat in enumerate(features):
        valid, errors, diag = validate_feature(
            feat, i, min_confidence=min_confidence, min_area_m2=min_area_m2
        )
        if valid:
            valid_count += 1
        else:
            for err in errors:
                if "Unidad Mínima Cartográfica" in err:
                    sub_umc_count += 1
                elif "menor al umbral de entrenamiento" in err:
                    low_conf_count += 1
                else:
                    attr_err_count += 1

    print(f"  Entidades analizadas : {total}")
    print(f"  Entidades válidas    : {valid_count} ({ (valid_count/total*100) if total else 0:.1f}%)")
    print(f"  Descartes Sub-UMC    : {sub_umc_count}")
    print(f"  Baja confianza (<{min_confidence}): {low_conf_count}")
    print(f"  Observaciones esquema: {attr_err_count}")

    return {
        "file": file_path.name,
        "total": total,
        "valid": valid_count,
        "sub_umc": sub_umc_count,
        "low_conf": low_conf_count,
        "attr_errors": attr_err_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Validador de Anotaciones Vectoriales GeoForest-AI")
    parser.add_argument("--path", type=str, default=".", help="Ruta a archivo .geojson o directorio con anotaciones.")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, help="Umbral mín. de confianza.")
    parser.add_argument("--min-area-m2", type=float, default=DEFAULT_MIN_AREA_M2, help="UMC mínima en m².")
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Ruta no encontrada: {target_path}")
        sys.exit(1)

    files_to_check = [target_path] if target_path.is_file() else list(target_path.glob("**/*.geojson"))

    if not files_to_check:
        print(f"No se encontraron archivos .geojson en {target_path}")
        sys.exit(0)

    print("==========================================================")
    print("      VALIDADOR TÉCNICO DE ANOTACIONES: GEOFOREST-AI     ")
    print("==========================================================")
    print(f"UMC Mínima: {args.min_area_m2} m² | Confianza Mínima: {args.min_confidence}")

    results = []
    for f in files_to_check:
        results.append(validate_geojson_file(f, min_confidence=args.min_confidence, min_area_m2=args.min_area_m2))

    print("\n==========================================================")
    print("                 RESUMEN DE AUDITORÍA                     ")
    print("==========================================================")
    total_all = sum(r.get("total", 0) for r in results)
    valid_all = sum(r.get("valid", 0) for r in results)
    print(f"Total entidades evaluadas: {total_all}")
    print(f"Total entidades aptas para entrenamiento: {valid_all}")


if __name__ == "__main__":
    main()
