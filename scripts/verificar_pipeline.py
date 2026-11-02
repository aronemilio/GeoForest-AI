"""Verificación previa del pipeline de GeoForest-AI Huánuco.

Comprueba entorno, artefactos de modelo y datos reales antes de una demo o de
citar métricas certificadas. No descarga nada ni modifica archivos.

Uso::

    .venv/bin/python scripts/verificar_pipeline.py
    .venv/bin/python scripts/verificar_pipeline.py --json

Código de salida: ``0`` si todos los artefactos críticos están presentes,
``1`` si falta alguno o si el manifiesto no coincide con los archivos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELO_CHECKPOINT = PROJECT_ROOT / "models" / "unet_best.pth"
MODELO_ONNX = PROJECT_ROOT / "models" / "unet_geoforest.onnx"
MANIFIESTO = PROJECT_ROOT / "models" / "manifest.json"
CALIBRACION = PROJECT_ROOT / "models" / "calibration.json"
ONNX_SERVIBLES = (
    PROJECT_ROOT / "models" / "unet_sentinel2_huanuco.onnx",
    PROJECT_ROOT / "models" / "unet_sentinel2_multitemporal.onnx",
)

PAR_SENTINEL2 = (
    PROJECT_ROOT / "data" / "raw" / "sentinel2" / "2024-07-18_S2B_18LUQ_20240718_0_L2A",
    PROJECT_ROOT / "data" / "raw" / "sentinel2" / "2024-09-11_S2A_18LUQ_20240911_0_L2A",
)
BENCHMARK_SCENES = PROJECT_ROOT / "data" / "processed" / "benchmark_scenes.json"
VAL_INDEX = PROJECT_ROOT / "data" / "processed" / "val_index.json"

REQUISITOS_EJECUCION = ("numpy", "rasterio", "onnxruntime")


@dataclass
class Resultado:
    """Estado agregado de la verificación."""

    criticos_faltantes: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    ok: list[str] = field(default_factory=list)


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def _rel(ruta: Path) -> str:
    try:
        return str(ruta.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(ruta)


def verificar_entorno(resultado: Resultado) -> None:
    """Comprueba intérprete activo y dependencias mínimas de ejecución."""
    ejecutable = Path(sys.executable)
    if "venv" in ejecutable.parts and ".venv" not in ejecutable.parts:
        resultado.criticos_faltantes.append(
            f"Intérprete {ejecutable} pertenece a venv/ (vacío). "
            "Activar con `source .venv/bin/activate`."
        )
    else:
        resultado.ok.append(f"Intérprete: {ejecutable}")

    faltantes = []
    for modulo in REQUISITOS_EJECUCION:
        try:
            __import__(modulo)
        except ImportError:
            faltantes.append(modulo)
    if faltantes:
        resultado.criticos_faltantes.append(
            "Módulos no importables: " + ", ".join(faltantes)
        )
    else:
        resultado.ok.append("Dependencias de ejecución importables: " + ", ".join(REQUISITOS_EJECUCION))


def _verificar_calibracion(resultado: Resultado) -> None:
    """Lee `models/calibration.json` y avisa si `T` agudiza en lugar de suavizar."""
    if not CALIBRACION.is_file():
        resultado.advertencias.append(f"Falta {_rel(CALIBRACION)}: se usará T = 1.0.")
        return

    try:
        calibracion = json.loads(CALIBRACION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        resultado.advertencias.append(f"{_rel(CALIBRACION)} ilegible: {exc}")
        return

    try:
        temperatura = float(calibracion.get("temperature", 1.0))
    except (TypeError, ValueError):
        resultado.advertencias.append(f"{_rel(CALIBRACION)}: 'temperature' no numérico.")
        return

    if temperatura < 1.0:
        resultado.advertencias.append(
            f"T = {temperatura:.4f} (< 1): el escalado agudiza las probabilidades; "
            "no describirlo como suavizado ni como calibración validada."
        )
    else:
        resultado.ok.append(f"T = {temperatura:.4f}")


def _verificar_manifiesto(resultado: Resultado) -> None:
    """Compara los SHA-256 de `models/manifest.json` con los archivos en disco."""
    if not MANIFIESTO.is_file():
        resultado.advertencias.append(f"Falta {_rel(MANIFIESTO)}: no se pueden verificar hashes.")
        return

    try:
        manifiesto = json.loads(MANIFIESTO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        resultado.advertencias.append(f"{_rel(MANIFIESTO)} ilegible: {exc}")
        return

    for clave, esperado in manifiesto.items():
        ruta = PROJECT_ROOT / clave
        if isinstance(esperado, dict):
            esperado = esperado.get("sha256", "")
        if not ruta.is_file():
            resultado.criticos_faltantes.append(
                f"El manifiesto referencia {clave}, que no existe en disco."
            )
            continue
        real = _sha256(ruta)
        if real != esperado:
            resultado.criticos_faltantes.append(
                f"Hash de {clave} no coincide con el manifiesto "
                f"(disco {real[:8]}… ≠ manifiesto {str(esperado)[:8]}…)."
            )
        else:
            resultado.ok.append(f"Hash verificado: {clave}")


def verificar_modelos(resultado: Resultado) -> None:
    """Comprueba checkpoint, ONNX canónico, grafos servibles, calibración y manifiesto."""
    for ruta in (MODELO_CHECKPOINT, MODELO_ONNX):
        if ruta.is_file():
            resultado.ok.append(f"{_rel(ruta)} presente ({ruta.stat().st_size / 1_000_000:.1f} MB)")
        else:
            resultado.criticos_faltantes.append(
                f"Falta {_rel(ruta)}. Regenerar con `python scripts/run_phase1_and_2.py` "
                "y `python scripts/export_onnx.py`."
            )

    for ruta in ONNX_SERVIBLES:
        if ruta.is_file():
            resultado.ok.append(f"{_rel(ruta)} presente ({_sha256(ruta)[:8]}…)")
        else:
            resultado.advertencias.append(
                f"Falta {_rel(ruta)}: la app no podrá cargar el modelo en la consola de inferencia."
            )

    # Comprobaciones independientes: un artefacto ausente no debe ocultar los demás.
    _verificar_calibracion(resultado)
    _verificar_manifiesto(resultado)


def verificar_inferencia(resultado: Resultado) -> None:
    """Ejecuta un smoke test del ONNX servible y detecta salidas casi constantes.

    Un modelo entrenado debe variar sus logits con la entrada. Un rango
    ``max - min < 0.05`` en un parche sintético 256x256 indica un grafo muerto
    (pesos sin entrenar o export incorrecto): cualquier umbral produciría
    detecciones sin significado.

    Limitación conocida: si el grafo ya incluye la sigmoide, la salida está en
    ``[0, 1]`` y un modelo muy seguro puede dar un rango pequeño incluso
    estando entrenado; en ese caso el aviso invita a revisarlo manualmente.
    La entrada es ruido (fuera de distribución), así que solo se marca como
    crítico un rango *prácticamente* nulo.
    """
    candidatos = [ruta for ruta in ONNX_SERVIBLES if ruta.is_file()]
    if not candidatos:
        resultado.advertencias.append("Sin ONNX servibles: no se pudo ejecutar el smoke test.")
        return

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        resultado.advertencias.append(f"Smoke test de inferencia omitido: {exc}")
        return

    for modelo in candidatos:
        try:
            session = ort.InferenceSession(str(modelo), providers=["CPUExecutionProvider"])
            forma_entrada = session.get_inputs()[0].shape
            canales = forma_entrada[1] if isinstance(forma_entrada[1], int) else 6
            rng = np.random.default_rng(0)
            entrada = rng.random((1, canales, 256, 256), dtype=np.float32)
            salida = session.run(None, {session.get_inputs()[0].name: entrada})[0]
            rango = float(salida.max() - salida.min())
            desviacion = float(salida.std())
        except Exception as exc:  # noqa: BLE001 - el smoke test no debe romper la verificación
            resultado.advertencias.append(f"No se pudo ejecutar {_rel(modelo)}: {exc}")
            continue

        if rango < 0.05:
            pista = (
                " Si la salida está en [0, 1] revisa si el grafo ya incluye Sigmoid antes de descartarlo."
                if 0.0 <= float(salida.min()) and float(salida.max()) <= 1.0
                else ""
            )
            resultado.criticos_faltantes.append(
                f"{_rel(modelo)} produce una salida casi constante "
                f"(rango {rango:.4f}, σ {desviacion:.5f}) sobre entrada aleatoria: "
                "el modelo servido no está entrenado o el export está roto. "
                "No presentar sus máscaras como detecciones." + pista
            )
        else:
            resultado.ok.append(
                f"Smoke test de inferencia: {_rel(modelo)} varía con la entrada "
                f"(rango {rango:.3f}, σ {desviacion:.3f})"
            )


def verificar_datos(resultado: Resultado) -> None:
    """Comprueba pares Sentinel-2 reales e índices del pipeline."""
    for ruta in PAR_SENTINEL2:
        bandas = [ruta / f"{banda}.tif" for banda in ("B04", "B08", "B11")]
        faltantes = [_rel(b) for b in bandas if not b.is_file()]
        if faltantes:
            resultado.criticos_faltantes.append(
                f"Par Sentinel-2 incompleto en {_rel(ruta)}; faltan: {', '.join(faltantes)}. "
                "La app usará la escena sintética de relleno y no debe presentarse como real."
            )
        else:
            resultado.ok.append(f"Par Sentinel-2 real: {_rel(ruta)}")

    for ruta, regenerar in (
        (BENCHMARK_SCENES, "python scripts/run_phase1_and_2.py"),
        (VAL_INDEX, "python scripts/run_phase1_and_2.py"),
    ):
        if ruta.is_file():
            resultado.ok.append(f"{_rel(ruta)} presente")
        else:
            resultado.criticos_faltantes.append(
                f"Falta {_rel(ruta)}. Regenerar con `{regenerar}`."
            )


def ejecutar_verificacion() -> tuple[Resultado, dict[str, Any]]:
    """Ejecuta todas las comprobaciones y devuelve el resultado agregado."""
    resultado = Resultado()
    verificar_entorno(resultado)
    verificar_modelos(resultado)
    verificar_inferencia(resultado)
    verificar_datos(resultado)

    resumen = {
        "ok": resultado.ok,
        "advertencias": resultado.advertencias,
        "criticos_faltantes": resultado.criticos_faltantes,
        "pipeline_operativo": not resultado.criticos_faltantes,
    }
    return resultado, resumen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verificación previa del pipeline GeoForest-AI.")
    parser.add_argument("--json", action="store_true", help="Emitir el resultado como JSON.")
    args = parser.parse_args(argv)

    resultado, resumen = ejecutar_verificacion()

    if args.json:
        print(json.dumps(resumen, indent=2, ensure_ascii=False))
    else:
        print("=== GeoForest-AI · verificación del pipeline ===")
        for linea in resultado.ok:
            print(f"[OK]      {linea}")
        for linea in resultado.advertencias:
            print(f"[AVISO]   {linea}")
        for linea in resultado.criticos_faltantes:
            print(f"[FALTA]   {linea}")
        print(
            "\nResultado: "
            + ("pipeline operativo" if resumen["pipeline_operativo"] else "artefactos críticos ausentes")
        )

    return 0 if resumen["pipeline_operativo"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
