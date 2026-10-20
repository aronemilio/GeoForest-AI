"""Generador de expediente técnico-legal FEMA / SERFOR."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import pandas as pd


def generar_expediente_fema_serfor(
    df_alertas: pd.DataFrame,
    zona: str,
) -> str:
    """
    Genera el expediente formal técnico-legal para fiscalía ambiental (FEMA) y SERFOR.

    Estructurado bajo los lineamientos de la Ley Forestal y de Fauna Silvestre N° 29763
    y el Código Penal Peruano (Art. 310 y concordantes sobre delitos contra los bosques).
    """
    ahora_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    exp_id = f"EXP-FEMA-HCO-{datetime.now().strftime('%Y%m')}-{len(df_alertas):03d}"
    area_total = df_alertas["area_ha"].sum() if not df_alertas.empty else 0.0
    confianza_prom = df_alertas["confianza_pct"].mean() if not df_alertas.empty else 0.0

    lineas = [
        "=" * 80,
        "REPÚBLICA DEL PERÚ — MINISTERIO PÚBLICO — FISCALÍA DE LA NACIÓN",
        "FISCALÍA ESPECIALIZADA EN MATERIA AMBIENTAL (FEMA) — DISTRITO FISCAL DE HUÁNUCO",
        "SERVICIO NACIONAL FORESTAL Y DE FAUNA SILVESTRE (SERFOR) — ATFFS HUÁNUCO",
        "=" * 80,
        f"INFORME TÉCNICO PERICIAL DE ALERTA TEMPRANA N°: {exp_id}",
        f"FECHA Y HORA DE EMISIÓN: {ahora_utc}",
        f"ÁMBITO TERRITORIAL: {zona} (Huánuco, Perú)",
        f"SISTEMA EMISOR: GeoForest-AI Huánuco v0.1.0 (Visor Táctico & U-Net ONNX)",
        "-" * 80,
        "1. MARCO NORMATIVO Y FUNDAMENTO TÉCNICO",
        "   - Ley N° 29763: Ley Forestal y de Fauna Silvestre y sus Reglamentos de Gestión Forestal.",
        "   - Código Penal, Título XIII: Delitos Ambientales, Cap. II (Delitos contra los Bosques,",
        "     Art. 310, 310-A, 310-B y 310-C sobre desbosque y quema no autorizada).",
        "   - Protocolo de Cooperación Interinstitucional FEMA - SERFOR para Alerta Temprana Satelital.",
        "-" * 80,
        "2. METODOLOGÍA Y EVIDENCIA TÉCNICA SATELITAL",
        "   - Constelación: Sentinel-2 MSI L2A (Agencia Espacial Europea / Copernicus).",
        "   - Sistema de Referencia Espacial: WGS 84 / UTM Zona 18S (EPSG:32718).",
        "   - Bandas de Análisis: B4 (Rojo 10m), B8 (NIR 10m), B11 (SWIR remuestreada a 10m vía bilineal).",
        "   - Algoritmo de Segmentación: Red Convolucional U-Net ejecutada con motor ONNX Runtime en CPU.",
        "   - Índices de Diagnóstico Espectral: Diferencial de Vegetación (ΔNDVI) y Calcinación (ΔNBR).",
        "-" * 80,
        "3. RESUMEN EJECUTIVO DE HALLAZGOS",
        f"   - Total de eventos identificados en el ámbito: {len(df_alertas)} alertas",
        f"   - Superficie total acumulada estimada: {area_total:.2f} hectáreas",
        f"   - Certeza algorítmica media del modelo: {confianza_prom:.1f}% de confianza",
        f"   - Recomendación: Constatación fiscal in situ con apoyo de la Unidad de Policía de Medio Ambiente.",
        "-" * 80,
        "4. DETALLE INDIVIDUAL DE EVENTOS DETECTADOS PARA FISCALIZACIÓN",
    ]

    if df_alertas.empty:
        lineas.append("   (No se registran eventos activos con los criterios de filtrado seleccionados)")
    else:
        for idx, (_, fila) in enumerate(df_alertas.iterrows(), start=1):
            lineas.extend([
                f"\n   [EVENTO N° {idx:02d}] Código: {fila['id_alerta']} | Severidad: {fila['severidad']}",
                f"   - Tipo de Evento: {fila['tipo_evento']}",
                f"   - Ubicación Jurisdiccional: Dpto. Huánuco, Distrito {fila['distrito']}, Sector {fila['sector']}",
                f"   - Coordenadas WGS84: Latitud {fila['lat']:.5f}°, Longitud {fila['lon']:.5f}°",
                f"   - Coordenadas Proyectadas Referenciales: EPSG:32718 (UTM Zona 18S)",
                f"   - Superficie Estimada: {fila['area_ha']:.2f} ha (Resolución 10m/píxel)",
                f"   - Nivel de Confianza U-Net: {fila['confianza_pct']:.1f}%",
                f"   - Anomalía Espectral ΔNDVI: {fila['ndvi_cambio']:.3f}",
                f"   - Fecha de Observación Satelital: {fila['fecha']}",
                f"   - Estado Procesal: {fila['estado']}",
            ])

    base_text = "\n".join(lineas)
    hash_sha256 = hashlib.sha256(base_text.encode("utf-8")).hexdigest()

    lineas.extend([
        "\n" + "-" * 80,
        "5. CADENA DE CUSTODIA DIGITAL E INTEGRIDAD CRIPTOGRÁFICA",
        f"   - Hash criptográfico SHA-256 de integridad: {hash_sha256}",
        "   - Certificación: Generado automáticamente por GeoForest-AI Huánuco.",
        "   - Nota técnica: Todo registro marcado como [SIMULADO] corresponde al banco demostrativo",
        "     para verificación de flujo procesal antes del enlace de producción.",
        "=" * 80,
    ])

    return "\n".join(lineas)
