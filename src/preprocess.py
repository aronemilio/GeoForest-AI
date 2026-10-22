"""
Base GIS de GeoForest-AI Huánuco.

Este módulo implementa la capa geoespacial de **preprocesamiento** del sistema:
apertura y validación de rásteres GeoTIFF, preparación de las bandas Sentinel-2
(B4, B8, B11) sobre una cuadrícula común, validación de vectores de referencia
(Geobosques / MINAM), rasterización de máscaras de etiquetas a partir de reglas
explícitas y escritura de resultados con sus metadatos.

Alcance
-------
- NO descarga datos satelitales ni referenciales.
- NO entrena modelos ni genera predicciones.
- NO inventa etiquetas de deforestación: toda máscara requiere una
  :class:`LabelRule` provista explícitamente por quien la invoca.
- NO reproyecta ni remuestrea en silencio: cualquier cambio de CRS o de
  cuadrícula exige parámetros explícitos y queda registrado en los metadatos.

Entradas esperadas
------------------
- Rásteres GeoTIFF georreferenciados (idealmente Sentinel-2 L2A). Las bandas
  pueden venir en un único archivo multibanda o en archivos separados.
- Vectores de referencia en cualquier formato soportado por GeoPandas
  (GeoPackage, GeoJSON, Shapefile, etc.).

Salidas
-------
- :class:`PreparedBands`: bandas alineadas como ``float32`` con ``NaN`` en
  píxeles inválidos, su :class:`GridSpec` y una máscara de validez combinada.
- :class:`LabelMask`: máscara categórica de etiquetas alineada a una cuadrícula.
- Archivos GeoTIFF reproyectados/guardados y un ``*.json`` de metadatos de
  procedencia (CRS, transform, resolución, NoData, método de remuestreo...).

Supuestos
---------
- Ráster: se asume que el orden de bandas dentro de un archivo multibanda no
  es fiable. Para seleccionar bandas por nombre use las descripciones de banda
  o pase el índice de banda de forma explícita en :class:`RasterSource`.
- Continuo vs. categórico: las bandas espectrales son datos *continuos*
  (remuestreo bilineal por defecto); las máscaras/etiquetas son *categóricas*
  (remuestreo ``nearest``).
- Resolución nativa: B4 y B8 son nativas a 10 m y B11 a 20 m. Alinear B11 a la
  cuadrícula de 10 m **requiere remuestreo explícito**.
- El significado de las clases de una máscara proviene exclusivamente de la
  ``LabelRule`` suministrada; este módulo no asume que un cambio de cobertura
  forestal equivalga a tala ilegal.

Errores esperados
-----------------
Todas las excepciones derivan de :class:`PreprocessError`:
``InvalidRasterError``, ``MissingCRSError``, ``CRSMismatchError``,
``BandError``, ``ResolutionMismatchError``, ``GridMismatchError``,
``InvalidVectorError`` y ``LabelRuleError``.

Limitaciones
------------
- No gestiona máscaras de nubes/SLC ni corrección atmosférica.
- El remuestreo se delega en Rasterio/GDAL; no se aplica una ponderación de
  área (``sum``/``average``) salvo que se solicite explícitamente.
- La rasterización usa ``rasterio.features.rasterize`` (una sola etiqueta por
  píxel mediante la última geometría que cubre el centro del píxel, o cualquiera
  que lo toque si ``all_touched=True``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # mensaje de error claro si falta la dependencia geoespacial
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import Affine
    from rasterio.warp import reproject as _reproject
except ImportError as exc:  # pragma: no cover - depende del entorno
    raise ImportError(
        "GeoForest-AI requiere 'rasterio' para el módulo GIS. "
        "Instálelo en el entorno virtual del proyecto con: pip install rasterio"
    ) from exc

try:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
except ImportError as exc:  # pragma: no cover - depende del entorno
    raise ImportError(
        "GeoForest-AI requiere 'geopandas' para trabajar con vectores de "
        "referencia. Instálelo con: pip install geopandas"
    ) from exc


__all__ = [
    # excepciones
    "PreprocessError",
    "InvalidRasterError",
    "MissingCRSError",
    "CRSMismatchError",
    "BandError",
    "ResolutionMismatchError",
    "GridMismatchError",
    "InvalidVectorError",
    "LabelRuleError",
    # constantes
    "SENTINEL2_BANDS",
    "S2_BAND_METADATA",
    # ráster
    "RasterProfile",
    "GridSpec",
    "RasterSource",
    "open_geotiff",
    "validate_geotiff",
    "read_raster_band",
    # remuestreo
    "DataKind",
    "ResamplingPolicy",
    "resolve_resampling",
    # bandas
    "PreparedBands",
    "prepare_s2_bands",
    "stack_bands",
    "compute_ndvi",
    "compute_nbr",
    # vectores
    "read_reference_vector",
    "prepare_reference_vector",
    "validate_reference_vector",
    # máscaras
    "LabelRule",
    "LabelMask",
    "rasterize_reference",
    # persistencia
    "save_raster",
    "save_prepared_bands",
    "save_label_mask",
    "save_metadata",
    "load_metadata",
]

# ---------------------------------------------------------------------------
# Utilidad de comparación de CRS
# ---------------------------------------------------------------------------
def _crs_equal(a: Any, b: Any) -> bool:
    """Compara dos CRS de forma robusta ante None o diferencias de clase."""
    if a is None or b is None:
        return a is b
    try:
        ra = CRS.from_user_input(a)
        rb = CRS.from_user_input(b)
    except Exception:
        return str(a) == str(b)
    if hasattr(ra, "equals"):
        return bool(ra.equals(rb))
    return bool(ra == rb)


# ---------------------------------------------------------------------------
# Constantes: bandas Sentinel-2 relevantes
# ---------------------------------------------------------------------------
SENTINEL2_BANDS: tuple[str, ...] = ("B4", "B8", "B11")

S2_BAND_METADATA: Mapping[str, Mapping[str, Any]] = {
    "B4": {"nombre": "Rojo", "longitud_onda_nm": 665, "resolucion_nativa_m": 10},
    "B8": {
        "nombre": "Infrarrojo cercano (NIR)",
        "longitud_onda_nm": 842,
        "resolucion_nativa_m": 10,
    },
    "B11": {
        "nombre": "Infrarrojo de onda corta (SWIR1)",
        "longitud_onda_nm": 1610,
        "resolucion_nativa_m": 20,
    },
}

MODULE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------
class PreprocessError(Exception):
    """Excepción base de todos los errores de preprocesamiento geoespacial."""


class InvalidRasterError(PreprocessError):
    """El archivo no es un ráster legible o carece de datos válidos."""


class MissingCRSError(PreprocessError):
    """El recurso no declara un sistema de referencia de coordenadas (CRS)."""


class CRSMismatchError(PreprocessError):
    """El CRS del recurso no coincide con el esperado."""


class BandError(PreprocessError):
    """Banda ausente, fuera de rango o sin información descriptiva."""


class ResolutionMismatchError(PreprocessError):
    """La resolución del ráster no coincide con la esperada."""


class GridMismatchError(PreprocessError):
    """Dos o más rásteres no comparten cuadrícula y falta una referencia."""


class InvalidVectorError(PreprocessError):
    """El vector de referencia es inválido o no cumple los requisitos."""


class LabelRuleError(PreprocessError):
    """La regla de etiquetado es incompleta, ambigua o inconsistente."""


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _as_path(path: str | Path | Any) -> Path | str:
    """Convierte a ``Path`` o ruta virtual ``/vsi`` y valida su existencia sin abrir el archivo."""
    if not isinstance(path, (str, Path)) and hasattr(path, "name"):
        name_str = str(path.name)
        if name_str.startswith("/vsi"):
            return name_str
        path = name_str
    if isinstance(path, str) and path.startswith("/vsi"):
        return path
    p = Path(path)
    if not p.exists():
        raise InvalidRasterError(f"No existe el archivo: {p}")
    if not p.is_file():
        raise InvalidRasterError(f"La ruta no es un archivo: {p}")
    return p


def _is_identity_transform(transform: Affine) -> bool:
    """``True`` si el transform no aporta georreferenciación (identidad 1:1)."""
    ident = Affine.identity()
    return tuple(transform)[:6] == tuple(ident)[:6]


def _utc_now_iso() -> str:
    """Marca de tiempo UTC en ISO 8601 (sin microsegundos)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Perfil de ráster
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RasterProfile:
    """
    Metadatos de un ráster GeoTIFF, ya validado como legible.

    Attributes:
        path: Ruta del archivo.
        crs: CRS declarado, o ``None`` si el archivo no está georreferenciado.
        transform: Transform afín píxel -> coordenadas.
        width: Número de columnas.
        height: Número de filas.
        count: Número de bandas.
        dtypes: Tipo de dato de cada banda.
        nodata: Valor NoData global (primera banda) o ``None``.
        nodata_per_band: Valor NoData por banda.
        descriptions: Descripción/nombre de cada banda (puede contener ``None``).
        bounds: ``(left, bottom, right, top)`` en el CRS del ráster.
    """

    path: Path
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    nodata: float | None
    nodata_per_band: tuple[float | None, ...]
    descriptions: tuple[str | None, ...]
    bounds: tuple[float, float, float, float]

    @property
    def shape(self) -> tuple[int, int]:
        """Forma ``(alto, ancho)`` de una banda."""
        return (self.height, self.width)

    @property
    def res(self) -> tuple[float, float]:
        """Resolución ``(x, y)`` en unidades del CRS."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def is_georeferenced(self) -> bool:
        """``True`` si hay CRS y el transform no es la identidad."""
        return self.crs is not None and not _is_identity_transform(self.transform)

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable (apta para JSON) del perfil."""
        return {
            "path": str(self.path),
            "crs": self.crs.to_string() if self.crs is not None else None,
            "transform": tuple(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtypes": list(self.dtypes),
            "nodata": self.nodata,
            "nodata_per_band": list(self.nodata_per_band),
            "descriptions": list(self.descriptions),
            "bounds": list(self.bounds),
            "res": list(self.res),
            "is_georeferenced": self.is_georeferenced,
        }


def open_geotiff(path: str | Path) -> RasterProfile:
    """
    Abre un GeoTIFF y devuelve su perfil validado como legible.

    Args:
        path: Ruta al archivo GeoTIFF (u otro formato ráster de GDAL).

    Returns:
        El :class:`RasterProfile` con CRS, dimensiones, resolución, NoData y
        bandas disponibles.

    Raises:
        InvalidRasterError: si el archivo no existe, no puede abrirse o no
            contiene al menos una banda con dimensiones positivas.

    Notas:
        Este paso es *permisivo*: un archivo sin CRS se abre y se reporta con
        ``crs=None``. Use :func:`validate_geotiff` para exigir georreferenciación.
    """
    p = _as_path(path)
    try:
        with rasterio.open(p) as ds:
            if ds.width <= 0 or ds.height <= 0 or ds.count < 1:
                raise InvalidRasterError(
                    f"Ráster sin datos válidos: {p} "
                    f"(ancho={ds.width}, alto={ds.height}, bandas={ds.count})"
                )
            descriptions = tuple(getattr(ds, "descriptions", (None,) * ds.count))
            nodatavals = tuple(ds.nodatavals)
            return RasterProfile(
                path=p,
                crs=ds.crs,
                transform=ds.transform,
                width=ds.width,
                height=ds.height,
                count=ds.count,
                dtypes=tuple(ds.dtypes),
                nodata=nodatavals[0] if nodatavals else None,
                nodata_per_band=nodatavals,
                descriptions=descriptions,
                bounds=tuple(ds.bounds),
            )
    except RasterioIOError as exc:
        raise InvalidRasterError(f"No se pudo abrir el ráster {p}: {exc}") from exc


def validate_geotiff(
    path: str | Path,
    *,
    require_crs: bool = True,
    require_georeferenced: bool = True,
    expected_crs: str | CRS | None = None,
    expected_bands: Sequence[str] | None = None,
    expected_band_count: int | None = None,
    expected_resolution: float | tuple[float, float] | None = None,
    resolution_tolerance: float = 1e-6,
) -> RasterProfile:
    """
    Abre un GeoTIFF y verifica CRS, dimensiones, resolución y bandas.

    Args:
        path: Ruta al GeoTIFF.
        require_crs: Exige que el archivo declare un CRS.
        require_georeferenced: Exige además un transform no identidad.
        expected_crs: CRS esperado (EPSG, WKT o cualquier valor aceptado por
            ``rasterio.crs.CRS.from_user_input``).
        expected_bands: Nombres de banda que deben existir según las
            descripciones del archivo (p. ej. ``("B4", "B8", "B11")``).
        expected_band_count: Número de bandas esperado.
        expected_resolution: Resolución esperada. Un escalar se aplica a X e Y;
            una tupla ``(x, y)`` se compara componente a componente.
        resolution_tolerance: Tolerancia absoluta al comparar resoluciones.

    Returns:
        El :class:`RasterProfile` validado.

    Raises:
        InvalidRasterError: archivo ilegible o sin datos.
        MissingCRSError: falta el CRS cuando es obligatorio.
        CRSMismatchError: el CRS no coincide con ``expected_crs``.
        BandError: bandas ausentes, recuento incorrecto o sin descripciones
            cuando se piden bandas por nombre.
        ResolutionMismatchError: la resolución no coincide.
    """
    profile = open_geotiff(path)

    if require_crs and profile.crs is None:
        raise MissingCRSError(f"El ráster {profile.path} no declara CRS.")
    if require_georeferenced and not profile.is_georeferenced:
        raise InvalidRasterError(
            f"El ráster {profile.path} no está georreferenciado "
            "(transform identidad o CRS ausente)."
        )

    if expected_crs is not None and profile.crs is not None:
        esperado = CRS.from_user_input(expected_crs)
        if not _crs_equal(profile.crs, esperado):
            raise CRSMismatchError(
                f"CRS inesperado en {profile.path}: "
                f"encontrado={profile.crs.to_string()}, esperado={esperado.to_string()}"
            )

    if expected_band_count is not None and profile.count != expected_band_count:
        raise BandError(
            f"Recuento de bandas inesperado en {profile.path}: "
            f"encontrado={profile.count}, esperado={expected_band_count}"
        )

    if expected_bands is not None:
        if all(d is None for d in profile.descriptions):
            raise BandError(
                f"El ráster {profile.path} no incluye descripciones de banda; "
                "no es posible verificar bandas por nombre. "
                "Seleccione bandas por índice explícito (RasterSource)."
            )
        disponibles = {
            str(d).strip().upper() for d in profile.descriptions if d is not None
        }
        faltantes = [
            b for b in expected_bands if str(b).strip().upper() not in disponibles
        ]
        if faltantes:
            raise BandError(
                f"Faltan bandas {faltantes} en {profile.path}. "
                f"Disponibles: {sorted(disponibles)}"
            )

    if expected_resolution is not None:
        if isinstance(expected_resolution, (int, float)):
            esperado_res = (float(expected_resolution), float(expected_resolution))
        else:
            esperado_res = (float(expected_resolution[0]), float(expected_resolution[1]))
        actual = profile.res
        if (
            abs(actual[0] - esperado_res[0]) > resolution_tolerance
            or abs(actual[1] - esperado_res[1]) > resolution_tolerance
        ):
            raise ResolutionMismatchError(
                f"Resolución inesperada en {profile.path}: "
                f"encontrada={actual}, esperada={esperado_res}"
            )

    return profile


# ---------------------------------------------------------------------------
# Cuadrícula de referencia
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridSpec:
    """
    Definición de una cuadrícula geoespacial (CRS + transform + dimensiones).

    Es el contrato de alineación del módulo: dos rásteres son compatibles si
    comparten CRS, dimensiones y transform.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int
    nodata: float | None = None

    @classmethod
    def from_profile(cls, profile: RasterProfile) -> "GridSpec":
        """
        Construye la cuadrícula a partir de un :class:`RasterProfile`.

        Raises:
            MissingCRSError: si el perfil no declara CRS.
        """
        if profile.crs is None:
            raise MissingCRSError(
                f"El ráster {profile.path} no tiene CRS; no puede usarse "
                "como cuadrícula de referencia."
            )
        return cls(
            crs=profile.crs,
            transform=profile.transform,
            width=profile.width,
            height=profile.height,
            nodata=profile.nodata,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Forma ``(alto, ancho)``."""
        return (self.height, self.width)

    @property
    def res(self) -> tuple[float, float]:
        """Resolución ``(x, y)``."""
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` en coordenadas del CRS."""
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return (left, bottom, right, top)

    def matches(self, other: "GridSpec", *, atol: float = 1e-6) -> bool:
        """
        Compara dos cuadrículas con tolerancia numérica.

        Args:
            other: Otra cuadrícula.
            atol: Tolerancia absoluta en las componentes del transform.

        Returns:
            ``True`` si comparten CRS, dimensiones y transform (dentro de ``atol``).
        """
        if self.shape != other.shape:
            return False
        if not _crs_equal(self.crs, other.crs):
            return False
        return bool(np.allclose(tuple(self.transform)[:6], tuple(other.transform)[:6], atol=atol, rtol=0))

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable de la cuadrícula."""
        return {
            "crs": self.crs.to_string(),
            "transform": list(tuple(self.transform)[:6]),
            "width": self.width,
            "height": self.height,
            "shape": list(self.shape),
            "res": list(self.res),
            "bounds": list(self.bounds),
            "nodata": self.nodata,
        }


def _crs_equal(a: Any, b: Any) -> bool:
    """
    Compara dos CRS de forma robusta ante ``None``.

    Normaliza ambos lados con ``rasterio.crs.CRS.from_user_input`` para poder
    mezclar objetos de Rasterio (``rasterio.crs.CRS``) y de GeoPandas/PyProj
    (``pyproj.CRS``) sin falsos negativos.
    """
    if a is None or b is None:
        return a is b
    try:
        ra = CRS.from_user_input(a)
        rb = CRS.from_user_input(b)
    except Exception:  # noqa: BLE001 - comparación de respaldo por texto
        return str(a) == str(b)
    if hasattr(ra, "equals"):
        return bool(ra.equals(rb))
    return bool(ra == rb)


def _coerce_grid(reference: "GridSpec | RasterProfile | str | Path | None") -> GridSpec | None:
    """Acepta GridSpec, RasterProfile, ruta a ráster o ``None`` -> GridSpec."""
    if reference is None:
        return None
    if isinstance(reference, GridSpec):
        return reference
    if isinstance(reference, RasterProfile):
        return GridSpec.from_profile(reference)
    if isinstance(reference, (str, Path)):
        return GridSpec.from_profile(open_geotiff(reference))
    raise TypeError(
        "referencia de cuadrícula no soportada: "
        f"{type(reference).__name__} (use GridSpec, RasterProfile o una ruta)"
    )


# ---------------------------------------------------------------------------
# Fuente de banda y remuestreo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RasterSource:
    """
    Referencia a una banda concreta dentro de un archivo ráster.

    Args:
        path: Ruta del GeoTIFF (monobanda o multibanda).
        band: Índice de banda **1-based** dentro del archivo.
        scale_factor: Si se indica, aplica ``valor * scale_factor + offset``
            tras leer (p. ej. ``1/10000`` para reflectancia Sentinel-2 L2A).
            ``None`` (por defecto) no escala nada.
        offset: Término aditivo del escalado.
        nodata: Valor NoData adicional a enmascarar (además del declarado en
            el archivo).
    """

    path: Path | str
    band: int = 1
    scale_factor: float | None = None
    offset: float = 0.0
    nodata: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable de la fuente."""
        return {
            "path": str(self.path),
            "band": self.band,
            "scale_factor": self.scale_factor,
            "offset": self.offset,
            "nodata": self.nodata,
        }


def _coerce_source(value: Any) -> RasterSource:
    """
    Normaliza una fuente de banda.

    Acepta ``RasterSource``, una ruta, una tupla ``(ruta, banda)`` o un mapeo
    con las claves ``path``/``band``/``scale_factor``/``offset``/``nodata``.
    """
    if isinstance(value, RasterSource):
        return value
    if not isinstance(value, (str, Path)) and hasattr(value, "name"):
        name_s = str(value.name)
        p = name_s if name_s.startswith("/vsi") else Path(name_s)
        return RasterSource(path=p, band=1)
    if isinstance(value, str) and value.startswith("/vsi"):
        return RasterSource(path=value, band=1)
    if isinstance(value, (str, Path)):
        return RasterSource(path=Path(value), band=1)
    if isinstance(value, Mapping):
        path_val = value["path"]
        if not isinstance(path_val, (str, Path)) and hasattr(path_val, "name"):
            name_s = str(path_val.name)
            p_val = name_s if name_s.startswith("/vsi") else Path(name_s)
        elif isinstance(path_val, str) and path_val.startswith("/vsi"):
            p_val = path_val
        else:
            p_val = Path(path_val)
        return RasterSource(
            path=p_val,
            band=int(value.get("band", 1)),
            scale_factor=value.get("scale_factor"),
            offset=float(value.get("offset", 0.0)),
            nodata=value.get("nodata"),
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        p_val = value[0]
        if not isinstance(p_val, (str, Path)) and hasattr(p_val, "name"):
            name_s = str(p_val.name)
            p_val = name_s if name_s.startswith("/vsi") else Path(name_s)
        elif isinstance(p_val, str) and p_val.startswith("/vsi"):
            pass
        else:
            p_val = Path(p_val)
        return RasterSource(path=p_val, band=int(value[1]))
    raise TypeError(f"fuente de banda no soportada: {type(value).__name__}")


class DataKind(str, Enum):
    """Naturaleza de los datos, determina el método de remuestreo adecuado."""

    CONTINUO = "continuo"
    CATEGORICO = "categorico"


@dataclass(frozen=True)
class ResamplingPolicy:
    """
    Política de remuestreo por naturaleza de dato.

    Evita elegir un método inadecuado de forma implícita: los datos continuos
    (reflectancia) y los categóricos (etiquetas) requieren métodos distintos.
    """

    continuous: Resampling = Resampling.bilinear
    categorical: Resampling = Resampling.nearest

    @classmethod
    def spectral(cls) -> "ResamplingPolicy":
        """Política recomendada para bandas espectrales continuas."""
        return cls(continuous=Resampling.bilinear, categorical=Resampling.nearest)

    def for_kind(self, kind: DataKind) -> Resampling:
        """Devuelve el método de remuestreo para la naturaleza indicada."""
        return self.continuous if kind is DataKind.CONTINUO else self.categorical

    def to_dict(self) -> dict[str, str]:
        """Representación serializable de la política."""
        return {
            "continuous": self.continuous.name,
            "categorical": self.categorical.name,
        }


def resolve_resampling(
    resampling: "Resampling | ResamplingPolicy | None",
    kind: DataKind,
) -> Resampling | None:
    """
    Resuelve el método de remuestreo efectivo para una naturaleza de dato.

    Args:
        resampling: ``None``, un :class:`rasterio.enums.Resampling` concreto o
            una :class:`ResamplingPolicy`.
        kind: Naturaleza del dato (continuo o categórico).

    Returns:
        El método de ``rasterio.enums.Resampling`` o ``None`` si no se pidió
        remuestreo.
    """
    if resampling is None:
        return None
    if isinstance(resampling, ResamplingPolicy):
        return resampling.for_kind(kind)
    if isinstance(resampling, Resampling):
        return resampling
    raise TypeError(
        "resampling debe ser None, Resampling o ResamplingPolicy; "
        f"recibido {type(resampling).__name__}"
    )


# ---------------------------------------------------------------------------
# Lectura de bandas y preparación Sentinel-2
# ---------------------------------------------------------------------------
def _read_source(source: RasterSource) -> tuple[np.ndarray, GridSpec, np.ndarray]:
    """
    Lee una banda como ``float32`` (``NaN`` en inválidos) + cuadrícula + validez.

    Raises:
        InvalidRasterError, MissingCRSError, BandError.
    """
    profile = open_geotiff(source.path)
    if source.band < 1 or source.band > profile.count:
        raise BandError(
            f"Banda {source.band} fuera de rango en {source.path} "
            f"(1..{profile.count})."
        )
    grid = GridSpec.from_profile(profile)
    with rasterio.open(source.path) as ds:
        band = ds.read(source.band, masked=True).astype("float32")
    if np.ma.isMaskedArray(band):
        data = np.ma.filled(band, np.nan)
    else:  # pragma: no cover - rasterio siempre devuelve MaskedArray con masked=True
        data = np.asarray(band, dtype="float32")

    if source.nodata is not None:
        data = np.where(data == source.nodata, np.nan, data)
    if source.scale_factor is not None:
        data = data * float(source.scale_factor) + float(source.offset)

    valid = ~np.isnan(data)
    return data, grid, valid


def read_raster_band(
    path: str | Path,
    band: int = 1,
    *,
    scale_factor: float | None = None,
    offset: float = 0.0,
    nodata: float | None = None,
    require_crs: bool = True,
) -> tuple[np.ndarray, GridSpec, np.ndarray]:
    """
    Lee una banda de un GeoTIFF y valida su georreferenciación.

    Args:
        path: Ruta al GeoTIFF.
        band: Índice de banda 1-based.
        scale_factor: Escalado opcional ``valor * scale_factor + offset``.
        offset: Término aditivo del escalado.
        nodata: NoData adicional a enmascarar.
        require_crs: Exige CRS en el archivo.

    Returns:
        ``(datos, grid, validez)`` donde ``datos`` es ``float32`` con ``NaN`` en
        píxeles inválidos, ``grid`` es la :class:`GridSpec` y ``validez`` es un
        arreglo booleano ``(alto, ancho)``.

    Raises:
        InvalidRasterError, MissingCRSError, BandError.
    """
    validate_geotiff(path, require_crs=require_crs, require_georeferenced=require_crs)
    source = RasterSource(
        path=Path(path),
        band=band,
        scale_factor=scale_factor,
        offset=offset,
        nodata=nodata,
    )
    return _read_source(source)


def _reproject_to_grid(
    array: np.ndarray,
    src_grid: GridSpec,
    dst_grid: GridSpec,
    resampling: Resampling,
) -> np.ndarray:
    """
    Reproyecta/remuestrea una banda continua a ``dst_grid`` (acción explícita).

    Los píxeles inválidos se propagan como ``NaN``.
    """
    destination = np.full(dst_grid.shape, np.nan, dtype="float32")
    _reproject(
        source=array,
        destination=destination,
        src_transform=src_grid.transform,
        src_crs=src_grid.crs,
        src_nodata=np.nan,
        dst_transform=dst_grid.transform,
        dst_crs=dst_grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return destination


def stack_bands(
    prepared: "PreparedBands",
    order: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Apila las bandas de un :class:`PreparedBands` en un arreglo ``(n, alto, ancho)``.

    Args:
        prepared: Bandas preparadas.
        order: Orden de bandas; por defecto el orden de inserción.

    Returns:
        Arreglo ``float32`` de forma ``(n_bandas, alto, ancho)``.
    """
    names = list(order) if order is not None else list(prepared.arrays)
    return np.stack([prepared.arrays[n] for n in names], axis=0)


@dataclass
class PreparedBands:
    """
    Bandas alineadas sobre una cuadrícula común.

    Attributes:
        arrays: Mapeo nombre de banda -> arreglo ``float32`` ``(alto, ancho)``
            con ``NaN`` en píxeles inválidos.
        grid: Cuadrícula común de todas las bandas.
        valid_mask: Máscara booleana combinada (válida solo si todas las bandas
            lo son en ese píxel).
        sources: Fuente de cada banda.
        resampling_applied: Método de remuestreo aplicado por banda (``"none"``
            si no fue necesario).
        band_metadata: Descripción opcional por banda (p. ej. longitud de onda).
    """

    arrays: dict[str, np.ndarray]
    grid: GridSpec
    valid_mask: np.ndarray
    sources: dict[str, RasterSource]
    resampling_applied: dict[str, str]
    band_metadata: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def band_names(self) -> tuple[str, ...]:
        """Nombres de banda en el orden de inserción."""
        return tuple(self.arrays.keys())

    def stack(self, order: Sequence[str] | None = None) -> np.ndarray:
        """Alias de :func:`stack_bands` para esta instancia."""
        return stack_bands(self, order)

    def to_metadata(self) -> dict[str, Any]:
        """Metadatos de procedencia de las bandas preparadas."""
        return {
            "modulo": "src.preprocess",
            "version": MODULE_VERSION,
            "generado_utc": _utc_now_iso(),
            "bandas": list(self.arrays.keys()),
            "band_metadata": {k: dict(v) for k, v in self.band_metadata.items()},
            "grid": self.grid.to_dict(),
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "resampling_aplicado": dict(self.resampling_applied),
            "pixeles_validos": int(np.count_nonzero(self.valid_mask)),
            "pixeles_totales": int(self.valid_mask.size),
        }


def prepare_s2_bands(
    sources: Mapping[str, Any],
    *,
    reference: GridSpec | RasterProfile | str | Path | None = None,
    resampling: Resampling | ResamplingPolicy | None = None,
    required_bands: Sequence[str] = SENTINEL2_BANDS,
    enforce_band_grid_match: bool = False,
) -> PreparedBands:
    """
    Prepara las bandas Sentinel-2 (por defecto B4, B8, B11) en una cuadrícula común.

    Comportamiento de alineación (nunca silencioso):

    - Si ``reference`` es ``None``, **todas** las bandas deben compartir ya la
      misma cuadrícula; si no, se lanza :class:`GridMismatchError` indicando que
      se proporcione ``reference`` y un ``resampling`` explícito.
    - Si ``reference`` se indica, cada banda cuya cuadrícula difiera se
      remuestrea con el método resuelto vía :func:`resolve_resampling`
      (continuo por defecto). Si difiere y ``resampling`` es ``None`` se lanza
      :class:`GridMismatchError` en lugar de remuestrear por defecto.

    Args:
        sources: Mapeo ``nombre_banda -> fuente``. La fuente puede ser una
            ruta, ``(ruta, banda)``, un mapeo o un :class:`RasterSource`.
        reference: Cuadrícula de destino. Si es ``None`` se infiere de las
            bandas (que deben coincidir entre sí).
        resampling: Método explícito (:class:`Resampling` o
            :class:`ResamplingPolicy`) usado solo cuando hace falta alinear.
        required_bands: Bandas obligatorias (por defecto ``B4, B8, B11``).
        enforce_band_grid_match: Si es ``True``, exige que todas las bandas
            compartan cuadrícula con ``reference`` y prohíbe remuestrear.

    Returns:
        :class:`PreparedBands` con las bandas alineadas.

    Raises:
        BandError: si falta alguna banda de ``required_bands``, si una fuente
            apunta a un archivo sin CRS o si el índice de banda es inválido.
        GridMismatchError: si las cuadrículas difieren sin ``reference``, o si
            difieren respecto de ``reference`` sin ``resampling``.
        InvalidRasterError, MissingCRSError.
    """
    if not sources:
        raise BandError("No se proporcionaron fuentes de bandas Sentinel-2.")

    normalized = {str(name): _coerce_source(value) for name, value in sources.items()}
    faltantes = [b for b in required_bands if b not in normalized]
    if faltantes:
        raise BandError(
            f"Faltan bandas requeridas {faltantes}. "
            f"Disponibles: {sorted(normalized)}"
        )

    read: dict[str, tuple[np.ndarray, GridSpec, np.ndarray]] = {}
    for name, src in normalized.items():
        read[name] = _read_source(src)

    target = _coerce_grid(reference)
    if target is None:
        grids = [g for _, g, _ in read.values()]
        base = grids[0]
        for name, (_, g, _) in read.items():
            if not base.matches(g):
                raise GridMismatchError(
                    f"La banda {name} no comparte cuadrícula con el resto y no "
                    "se indicó 'reference'. Proporcione una cuadrícula de "
                    "referencia y un método de remuestreo explícito para "
                    "alinearlas (no se remuestrea en silencio)."
                )
        target = base

    arrays: dict[str, np.ndarray] = {}
    resampling_applied: dict[str, str] = {}
    valid_mask: np.ndarray | None = None

    for name, (data, grid, valid) in read.items():
        if grid.matches(target):
            arrays[name] = data
            resampling_applied[name] = "none"
            band_valid = valid
        else:
            method = resolve_resampling(resampling, DataKind.CONTINUO)
            if enforce_band_grid_match or method is None:
                raise GridMismatchError(
                    f"La banda {name} está en una cuadrícula distinta a la de "
                    "referencia. Indique un método de remuestreo explícito "
                    "(por ejemplo ResamplingPolicy.spectral()) o use "
                    "enforce_band_grid_match=False. No se remuestrea en silencio."
                )
            arrays[name] = _reproject_to_grid(data, grid, target, method)
            resampling_applied[name] = method.name
            band_valid = ~np.isnan(arrays[name])

        valid_mask = band_valid if valid_mask is None else (valid_mask & band_valid)

    assert valid_mask is not None  # garantizado porque sources no está vacío

    band_metadata = {
        name: dict(S2_BAND_METADATA.get(name, {})) for name in arrays
    }

    return PreparedBands(
        arrays=arrays,
        grid=target,
        valid_mask=valid_mask,
        sources=normalized,
        resampling_applied=resampling_applied,
        band_metadata=band_metadata,
    )


def compute_ndvi(prepared: PreparedBands, *, b_red: str = "B4", b_nir: str = "B8") -> np.ndarray:
    """
    Calcula el NDVI = (NIR - Rojo) / (NIR + Rojo).

    Args:
        prepared: Bandas preparadas que incluyan las bandas requeridas.
        b_red: Nombre de la banda roja.
        b_nir: Nombre de la banda NIR.

    Returns:
        Arreglo ``float32`` con NDVI en ``[-1, 1]`` y ``NaN`` donde el
        denominador es nulo o la banda no es válida.

    Raises:
        BandError: si alguna banda no está presente.
    """
    return _normalized_difference(prepared, b_nir, b_red, "NDVI")


def compute_nbr(prepared: PreparedBands, *, b_nir: str = "B8", b_swir: str = "B11") -> np.ndarray:
    """
    Calcula el NBR = (NIR - SWIR1) / (NIR + SWIR1).

    Nota: el NBR es sensible a cambios de humedad y vegetación, pero **un
    descenso del NBR no implica por sí mismo deforestación ni tala ilegal**;
    requiere interpretación con series temporales y validación externa.

    Args:
        prepared: Bandas preparadas que incluyan las bandas requeridas.
        b_nir: Nombre de la banda NIR.
        b_swir: Nombre de la banda SWIR1.

    Returns:
        Arreglo ``float32`` con NBR en ``[-1, 1]`` y ``NaN`` en píxeles inválidos.

    Raises:
        BandError: si alguna banda no está presente.
    """
    return _normalized_difference(prepared, b_nir, b_swir, "NBR")


def _normalized_difference(
    prepared: PreparedBands, name_a: str, name_b: str, etiqueta: str
) -> np.ndarray:
    """Índice normalizado ``(A - B) / (A + B)`` con manejo de división por cero."""
    if name_a not in prepared.arrays or name_b not in prepared.arrays:
        raise BandError(
            f"{etiqueta} requiere las bandas '{name_a}' y '{name_b}'; "
            f"disponibles: {sorted(prepared.arrays)}"
        )
    a = prepared.arrays[name_a].astype("float64")
    b = prepared.arrays[name_b].astype("float64")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0, np.nan, (a - b) / denom)
    result = np.where(prepared.valid_mask, result, np.nan)
    return result.astype("float32")


# ---------------------------------------------------------------------------
# Vectores de referencia (Geobosques / MINAM)
# ---------------------------------------------------------------------------
def validate_reference_vector(
    gdf: "gpd.GeoDataFrame",
    *,
    required_columns: Sequence[str] | None = None,
    min_features: int = 1,
) -> dict[str, Any]:
    """
    Valida un GeoDataFrame de referencia y devuelve un resumen.

    Args:
        gdf: Vector de referencia.
        required_columns: Columnas (p. ej. el campo de clase) que deben existir.
        min_features: Número mínimo de entidades.

    Returns:
        Resumen con ``n_features``, ``crs``, ``geom_types``, ``bounds`` y
        ``n_geometrias_invalidas``.

    Raises:
        InvalidVectorError: si no es un GeoDataFrame, está vacío, tiene menos
            entidades de las exigidas, contiene geometrías nulas o no incluye
            las columnas requeridas.
        MissingCRSError: si el vector no declara CRS.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise InvalidVectorError(
            f"Se esperaba un GeoDataFrame; recibido {type(gdf).__name__}"
        )
    if len(gdf) < min_features:
        raise InvalidVectorError(
            f"El vector tiene {len(gdf)} entidades; se requieren al menos {min_features}."
        )
    if gdf.crs is None:
        raise MissingCRSError(
            "El vector de referencia no declara CRS; no puede alinearse a una cuadrícula."
        )
    if gdf.geometry.isna().any():
        raise InvalidVectorError("El vector contiene geometrías nulas (None).")

    if required_columns is not None:
        faltantes = [c for c in required_columns if c not in gdf.columns]
        if faltantes:
            raise InvalidVectorError(
                f"Faltan columnas requeridas {faltantes} en el vector. "
                f"Disponibles: {list(gdf.columns)}"
            )

    invalidas = int((~gdf.geometry.is_valid).sum())
    geom_types = sorted({str(t) for t in gdf.geometry.geom_type.unique()})
    return {
        "n_features": int(len(gdf)),
        "crs": gdf.crs.to_string() if gdf.crs is not None else None,
        "geom_types": geom_types,
        "bounds": [float(v) for v in gdf.total_bounds],
        "n_geometrias_invalidas": invalidas,
    }


def read_reference_vector(
    path: str | Path,
    *,
    layer: str | None = None,
    required_columns: Sequence[str] | None = None,
) -> "gpd.GeoDataFrame":
    """
    Lee y valida un vector de referencia desde disco.

    Args:
        path: Ruta al vector (GeoPackage, GeoJSON, Shapefile, ...).
        layer: Capa a leer en formatos multícapa (p. ej. GeoPackage).
        required_columns: Columnas obligatorias.

    Returns:
        El GeoDataFrame validado (sin reproyectar).

    Raises:
        InvalidVectorError: archivo ilegible o no válido.
        MissingCRSError: vector sin CRS.
    """
    p = Path(path)
    if not p.exists():
        raise InvalidVectorError(f"No existe el vector de referencia: {p}")
    try:
        gdf = gpd.read_file(p, layer=layer) if layer else gpd.read_file(p)
    except Exception as exc:  # noqa: BLE001 - normalizamos cualquier fallo de I/O
        raise InvalidVectorError(f"No se pudo leer el vector {p}: {exc}") from exc
    validate_reference_vector(gdf, required_columns=required_columns)
    return gdf


def prepare_reference_vector(
    path: str | Path,
    *,
    layer: str | None = None,
    required_columns: Sequence[str] | None = None,
    target_crs: str | CRS | None = None,
    fix_invalid: bool = True,
) -> "gpd.GeoDataFrame":
    """
    Lee, repara y (opcionalmente) reproyecta un vector de referencia.

    Args:
        path: Ruta al vector.
        layer: Capa a leer.
        required_columns: Columnas obligatorias.
        target_crs: CRS de destino. Si se indica y difiere del actual se aplica
            ``to_crs`` (transformación **explícita**, registrada por el llamador).
        fix_invalid: Si ``True``, sustituye geometrías inválidas por su versión
            ``make_valid``. Documentado porque altera la geometría original.

    Returns:
        GeoDataFrame listo para rasterizar.

    Raises:
        InvalidVectorError, MissingCRSError, CRSMismatchError.
    """
    gdf = read_reference_vector(path, layer=layer, required_columns=required_columns)

    if fix_invalid:
        if not gdf.geometry.is_valid.all():
            n_invalidas = int((~gdf.geometry.is_valid).sum())
            logger.warning(
                "Reparando %d geometrías inválidas de %s con make_valid().",
                n_invalidas,
                path,
            )
            gdf = gdf.copy()
            gdf.geometry = gdf.geometry.make_valid()

    if target_crs is not None:
        destino = CRS.from_user_input(target_crs)
        if gdf.crs is None:  # pragma: no cover - ya validado arriba
            raise MissingCRSError("El vector no declara CRS; no puede reproyectarse.")
        if not _crs_equal(gdf.crs, destino):
            gdf = gdf.to_crs(destino)

    validate_reference_vector(gdf, required_columns=required_columns)
    return gdf


# ---------------------------------------------------------------------------
# Rasterización de máscaras con reglas explícitas
# ---------------------------------------------------------------------------
@dataclass
class LabelRule:
    """
    Regla explícita de etiquetado vector -> máscara categórica.

    El significado de cada código de clase lo define quien construye la regla;
    este módulo **no** asume ninguna semántica de deforestación.

    Args:
        field: Columna del vector que contiene la clase de origen.
        mapping: Mapeo ``valor_de_campo -> código_de_clase`` (enteros).
        background_value: Valor para píxeles no cubiertos por ninguna geometría.
        nodata: Valor NoData que se declarará si la máscara se guarda como
            GeoTIFF (``None`` para no declararlo).
        allow_unmapped: Si ``False`` (por defecto), cualquier valor del campo
            ausente en ``mapping`` provoca ``LabelRuleError``. Si ``True``, esos
            valores se marcan con ``unmapped_value``.
        unmapped_value: Código usado cuando ``allow_unmapped=True``.
    """

    field: str
    mapping: Mapping[Any, int]
    background_value: int = 0
    nodata: int | None = None
    allow_unmapped: bool = False
    unmapped_value: int = -1

    def __post_init__(self) -> None:
        self.mapping = {k: int(v) for k, v in self.mapping.items()}
        if not self.mapping:
            raise LabelRuleError("La regla de etiquetado no define ningún mapeo.")
        if not isinstance(self.field, str) or not self.field:
            raise LabelRuleError("La regla debe indicar un campo ('field') válido.")

    @property
    def label_values(self) -> tuple[int, ...]:
        """Códigos de clase únicos y ordenados definidos en el mapeo."""
        return tuple(sorted(set(self.mapping.values())))

    def to_dict(self) -> dict[str, Any]:
        """Representación serializable de la regla."""
        return {
            "field": self.field,
            "mapping": {str(k): int(v) for k, v in self.mapping.items()},
            "background_value": self.background_value,
            "nodata": self.nodata,
            "allow_unmapped": self.allow_unmapped,
            "unmapped_value": self.unmapped_value,
            "label_values": list(self.label_values),
            "nota": "El significado de cada código es externo al módulo.",
        }


@dataclass(frozen=True)
class LabelMask:
    """
    Máscara categórica rasterizada alineada a una cuadrícula.

    Attributes:
        array: Arreglo entero ``(alto, ancho)`` con los códigos de clase.
        grid: Cuadrícula de alineación (CRS + transform + dimensiones).
        rule: Regla de etiquetado aplicada.
    """

    array: np.ndarray
    grid: GridSpec
    rule: LabelRule

    @property
    def class_values(self) -> tuple[int, ...]:
        """Códigos presentes en la máscara, ordenados."""
        return tuple(sorted(int(v) for v in np.unique(self.array)))

    def counts(self) -> dict[int, int]:
        """Conteo de píxeles por código de clase."""
        valores, cuentas = np.unique(self.array, return_counts=True)
        return {int(v): int(c) for v, c in zip(valores, cuentas)}

    def to_metadata(self) -> dict[str, Any]:
        """Metadatos de procedencia de la máscara."""
        return {
            "modulo": "src.preprocess",
            "version": MODULE_VERSION,
            "generado_utc": _utc_now_iso(),
            "tipo": "mascara_etiquetas",
            "grid": self.grid.to_dict(),
            "rule": self.rule.to_dict(),
            "class_values_presentes": list(self.class_values),
            "conteo_pixeles": self.counts(),
            "shape": list(self.array.shape),
        }


def rasterize_reference(
    gdf: "gpd.GeoDataFrame",
    grid: GridSpec,
    rule: LabelRule,
    *,
    all_touched: bool = False,
    dtype: str = "int32",
) -> LabelMask:
    """
    Rasteriza un vector de referencia a una máscara alineada a ``grid``.

    Requiere **datos vectoriales y una regla de etiquetado explícita**; no
    infiere clases ni asume tipos de cambio forestal.

    Args:
        gdf: Vector de referencia con el campo indicado por ``rule.field``.
        grid: Cuadrícula de destino; la máscara queda exactamente alineada
            (mismo CRS, transform y dimensiones).
        rule: Regla explícita ``valor -> código``.
        all_touched: Si ``True``, incluye todo píxel tocado por una geometría;
            si ``False``, solo píxeles cuyo centro cae dentro.
        dtype: Tipo entero de la máscara (p. ej. ``"uint8"``, ``"int32"``).

    Returns:
        :class:`LabelMask` alineada con ``grid``.

    Raises:
        LabelRuleError: campo ausente, valores sin mapear (sin ``allow_unmapped``)
            o geometrías no rasterizables.
        InvalidVectorError, MissingCRSError.
    """
    if grid.crs is None:  # pragma: no cover - GridSpec siempre lleva CRS
        raise MissingCRSError("La cuadrícula de destino no tiene CRS.")
    validate_reference_vector(gdf, required_columns=[rule.field])

    if not _crs_equal(gdf.crs, grid.crs):
        raise CRSMismatchError(
            "El vector y la cuadrícula están en CRS distintos "
            f"({gdf.crs.to_string()} vs {grid.crs.to_string()}); reproyecte el "
            "vector explícitamente con prepare_reference_vector(target_crs=...)."
        )

    valores_presentes = set(gdf[rule.field].dropna().unique().tolist())
    sin_mapear = sorted(valores_presentes - set(rule.mapping.keys()), key=str)
    if sin_mapear and not rule.allow_unmapped:
        raise LabelRuleError(
            f"Valores de '{rule.field}' sin mapear en la regla: {sin_mapear}. "
            "Añádalos a mapping o active allow_unmapped=True."
        )

    shapes: list[tuple[BaseGeometry, int]] = []
    for valor, geom in zip(gdf[rule.field].tolist(), gdf.geometry.tolist()):
        if geom is None or geom.is_empty:
            continue
        if valor in rule.mapping:
            shapes.append((geom, rule.mapping[valor]))
        elif rule.allow_unmapped:
            shapes.append((geom, rule.unmapped_value))

    if not shapes:
        raise LabelRuleError(
            "No hay geometrías válidas con valores mapeados para rasterizar."
        )

    arr = _rasterize(
        shapes=shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=rule.background_value,
        all_touched=all_touched,
        dtype=dtype,
    )
    return LabelMask(array=arr, grid=grid, rule=rule)


# ---------------------------------------------------------------------------
# Persistencia: rásteres, máscaras y metadatos
# ---------------------------------------------------------------------------
def save_raster(
    array: np.ndarray,
    path: str | Path,
    grid: GridSpec,
    *,
    dtype: str | None = None,
    nodata: float | int | None = None,
    descriptions: Sequence[str] | None = None,
    compress: str = "deflate",
    tiled: bool = True,
) -> Path:
    """
    Guarda un arreglo como GeoTIFF georreferenciado con la cuadrícula indicada.

    Args:
        array: Arreglo 2D ``(alto, ancho)`` o 3D ``(bandas, alto, ancho)``.
        path: Ruta de salida ``.tif``.
        grid: Cuadrícula (CRS, transform, dimensiones).
        dtype: Tipo de dato de salida; por defecto el del arreglo.
        nodata: Valor NoData a declarar.
        descriptions: Nombres de banda (solo para arreglos 3D).
        compress: Compresión GDAL.
        tiled: Escritura en teselas (recomendado para rásteres grandes).

    Returns:
        La ruta escrita.

    Raises:
        InvalidRasterError: forma incompatible con la cuadrícula o CRS ausente.
    """
    if array.ndim == 2:
        count = 1
        height, width = array.shape
        data = array[np.newaxis, :, :]
    elif array.ndim == 3:
        count, height, width = array.shape
        data = array
    else:
        raise InvalidRasterError(
            f"El arreglo debe ser 2D o 3D; recibido {array.ndim} dimensiones."
        )

    if (height, width) != grid.shape:
        raise InvalidRasterError(
            f"La forma del arreglo {(height, width)} no coincide con la "
            f"cuadrícula {grid.shape}."
        )
    if grid.crs is None:  # pragma: no cover - GridSpec siempre lleva CRS
        raise MissingCRSError("No se puede guardar un ráster sin CRS.")

    if descriptions is not None and len(descriptions) != count:
        raise InvalidRasterError(
            f"Se esperaban {count} descripciones de banda; recibidas {len(descriptions)}."
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    perfil = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "crs": grid.crs,
        "transform": grid.transform,
        "dtype": dtype or data.dtype.name,
        "compress": compress,
        "tiled": tiled and width > 16 and height > 16,
    }
    if nodata is not None:
        perfil["nodata"] = nodata

    with rasterio.open(out, "w", **perfil) as dst:
        if descriptions is not None:
            dst.descriptions = tuple(descriptions)
        dst.write(data.astype(perfil["dtype"]))
    logger.info("Ráster guardado en %s", out)
    return out


def save_prepared_bands(
    prepared: PreparedBands,
    output_dir: str | Path,
    *,
    filename: str = "s2_b4_b8_b11.tif",
    metadata_filename: str | None = "s2_b4_b8_b11.metadata.json",
    order: Sequence[str] | None = None,
    nodata: float | None = np.nan,
    compress: str = "deflate",
) -> dict[str, Any]:
    """
    Guarda las bandas preparadas en un GeoTIFF multibanda + metadatos JSON.

    Args:
        prepared: Bandas preparadas.
        output_dir: Directorio de salida (se crea si no existe).
        filename: Nombre del GeoTIFF multibanda.
        metadata_filename: Nombre del JSON de metadatos (``None`` para omitirlo).
        order: Orden de apilado de bandas.
        nodata: Valor NoData a declarar (``NaN`` por defecto, coherente con la
            representación de píxeles inválidos).
        compress: Compresión GDAL.

    Returns:
        Diccionario con ``raster`` (ruta) y ``metadata`` (ruta o ``None``).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(order) if order is not None else list(prepared.arrays)
    stack = stack_bands(prepared, names)
    raster_path = save_raster(
        stack,
        out_dir / filename,
        prepared.grid,
        nodata=nodata,
        descriptions=names,
        compress=compress,
    )

    metadata = prepared.to_metadata()
    metadata["archivo_raster"] = str(raster_path)
    metadata["orden_bandas"] = names
    meta_path: Path | None = None
    if metadata_filename is not None:
        meta_path = save_metadata(metadata, out_dir / metadata_filename)

    return {"raster": raster_path, "metadata": meta_path}


def save_label_mask(
    mask: LabelMask,
    path: str | Path,
    *,
    dtype: str | None = None,
    metadata_path: str | Path | None = None,
    compress: str = "deflate",
) -> dict[str, Any]:
    """
    Guarda una máscara de etiquetas alineada a su cuadrícula + metadatos.

    Args:
        mask: Máscara a guardar.
        path: Ruta del GeoTIFF de salida.
        dtype: Tipo entero de salida; por defecto el de la máscara.
        metadata_path: Ruta del JSON de metadatos (``None`` para derivarla del
            nombre del GeoTIFF cambiando la extensión a ``.metadata.json``).
        compress: Compresión GDAL.

    Returns:
        Diccionario con ``raster`` y ``metadata``.

    Raises:
        InvalidRasterError: si ``dtype`` no es entero.
    """
    dt = dtype or mask.array.dtype.name
    if not np.issubdtype(np.dtype(dt), np.integer):
        raise InvalidRasterError(
            f"Una máscara categórica debe ser entera; dtype recibido: {dt}"
        )

    raster_path = save_raster(
        mask.array.astype(dt),
        path,
        mask.grid,
        nodata=mask.rule.nodata,
        compress=compress,
    )

    if metadata_path is None:
        p = Path(path)
        metadata_path = p.with_suffix(".metadata.json")
    metadata = mask.to_metadata()
    metadata["archivo_raster"] = str(raster_path)
    metadata["dtype"] = dt
    meta_path = save_metadata(metadata, metadata_path)

    return {"raster": raster_path, "metadata": meta_path}


def save_metadata(metadata: Mapping[str, Any], path: str | Path) -> Path:
    """
    Escribe metadatos geoespaciales en un archivo JSON (UTF-8).

    Args:
        metadata: Diccionario serializable.
        path: Ruta de salida ``.json``.

    Returns:
        La ruta escrita.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(dict(metadata), fh, ensure_ascii=False, indent=2, default=str)
    return out


def load_metadata(path: str | Path) -> dict[str, Any]:
    """
    Lee un archivo de metadatos JSON.

    Args:
        path: Ruta del JSON.

    Returns:
        El diccionario de metadatos.

    Raises:
        InvalidRasterError: si el archivo no existe o no es JSON válido.
    """
    p = Path(path)
    if not p.exists():
        raise InvalidRasterError(f"No existen los metadatos: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise InvalidRasterError(f"Metadatos JSON inválidos en {p}: {exc}") from exc


# ---------------------------------------------------------------------------
# Contrato de preparación multitemporal y rasterización MINAM
# ---------------------------------------------------------------------------
def _parse_time_sources(sources: Any) -> dict[str, Any]:
    """Normaliza las fuentes de bandas (B4, B8, B11) para una fecha temporal."""
    if isinstance(sources, Mapping):
        res = {}
        for b in ("B4", "B8", "B11"):
            aliases = [b, b.lower()]
            if b in ("B4", "B8"):
                aliases.extend([f"B0{b[1]}", f"b0{b[1]}"])
            found = False
            for alias in aliases:
                if alias in sources:
                    res[b] = sources[alias]
                    found = True
                    break
            if not found:
                raise BandError(f"Falta la banda requerida '{b}' en las fuentes proporcionadas.")
        return res
    if isinstance(sources, (list, tuple)):
        if len(sources) == 3:
            return {"B4": sources[0], "B8": sources[1], "B11": sources[2]}
        elif len(sources) == 1:
            sources = sources[0]
        else:
            raise BandError(
                f"Se esperaban 3 bandas (B4, B8, B11) o 1 archivo multibanda, "
                f"se recibieron {len(sources)} elementos."
            )
    return {
        "B4": (sources, 1),
        "B8": (sources, 2),
        "B11": (sources, 3),
    }


def preparar_tensor_multitemporal(
    t0_paths: Mapping[str, Any] | Sequence[Any] | str | Path | Any,
    t1_paths: Mapping[str, Any] | Sequence[Any] | str | Path | Any,
    *,
    target_crs: str | CRS = "EPSG:32718",
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Carga, valida y normaliza bandas multitemporales Sentinel-2 para el pipeline de inferencia.

    Genera una tupla ``(tensor_numpy, profile)`` con dimensiones exactas ``(6, H, W)``
    escaladas a ``[0.0, 1.0]``, con B11 remuestreada a 10 m vía interpolación bilineal
    y validando consistencia en coordenadas proyectadas EPSG:32718 (UTM Zona 18S para Huánuco).

    Canales de salida del tensor (6, H, W):
        - Canal 0: T0 B4 (Rojo, 10 m)
        - Canal 1: T0 B8 (NIR, 10 m)
        - Canal 2: T0 B11 (SWIR, remuestreada a 10 m vía bilineal)
        - Canal 3: T1 B4 (Rojo, 10 m)
        - Canal 4: T1 B8 (NIR, 10 m)
        - Canal 5: T1 B11 (SWIR, remuestreada a 10 m vía bilineal)

    Args:
        t0_paths: Fuentes de bandas para T0 (antes). Dict {'B4', 'B8', 'B11'},
            secuencia [B4, B8, B11], o archivo multibanda.
        t1_paths: Fuentes de bandas para T1 (después). Dict {'B4', 'B8', 'B11'},
            secuencia [B4, B8, B11], o archivo multibanda.
        target_crs: CRS objetivo esperado (por defecto "EPSG:32718").
        normalize: Si es True, escala la reflectancia a [0.0, 1.0].

    Returns:
        tuple[np.ndarray, dict[str, Any]]:
            - tensor_numpy: Arreglo (6, H, W) en float32 con valores en [0.0, 1.0].
            - profile: Perfil de Rasterio con CRS, transform, dimensiones y metadatos.

    Raises:
        MissingCRSError: Si algún archivo carece de CRS.
        CRSMismatchError: Si el CRS de entrada no coincide con target_crs proyectado.
        BandError: Si faltan bandas requeridas o el número de bandas es insuficiente.
        GridMismatchError: Si las cuadrículas no se pueden alinear.
    """
    s0 = _parse_time_sources(t0_paths)
    s1 = _parse_time_sources(t1_paths)

    expected_crs = target_crs if isinstance(target_crs, CRS) else CRS.from_user_input(target_crs)
    if not expected_crs.is_projected:
        raise CRSMismatchError(
            f"target_crs debe ser un CRS proyectado (como EPSG:32718 UTM Zona 18S), "
            f"se recibió: {expected_crs}."
        )

    # Validar consistencia de CRS en todas las fuentes
    for label, s_dict in (("T0", s0), ("T1", s1)):
        for name, src in s_dict.items():
            coerced = _coerce_source(src)
            prof = open_geotiff(coerced.path)
            if prof.crs is None:
                raise MissingCRSError(
                    f"La fuente para {label} {name} ({coerced.path}) no contiene CRS georreferenciado."
                )
            if not _crs_equal(prof.crs, expected_crs):
                raise CRSMismatchError(
                    f"CRS inconsistente en {label} {name}: se encontró {prof.crs}, "
                    f"se requiere coordenadas proyectadas {expected_crs} (UTM Zona 18S)."
                )

    # Referencia espacial a 10m obtenida de B4 de T0
    src_b4_t0 = _coerce_source(s0["B4"])
    prof_b4_t0 = open_geotiff(src_b4_t0.path)
    grid_10m = GridSpec.from_profile(prof_b4_t0)

    # Preparar bandas de T0 y T1 alineadas a grid_10m con remuestreo bilineal explícito
    prep_t0 = prepare_s2_bands(
        s0,
        reference=grid_10m,
        resampling=Resampling.bilinear,
    )
    prep_t1 = prepare_s2_bands(
        s1,
        reference=grid_10m,
        resampling=Resampling.bilinear,
    )

    t0_arrays = [prep_t0.arrays["B4"], prep_t0.arrays["B8"], prep_t0.arrays["B11"]]
    t1_arrays = [prep_t1.arrays["B4"], prep_t1.arrays["B8"], prep_t1.arrays["B11"]]

    raw_tensor = np.stack(t0_arrays + t1_arrays, axis=0).astype(np.float32)
    raw_tensor = np.nan_to_num(raw_tensor, nan=0.0, posinf=1.0, neginf=0.0)

    if normalize:
        max_val = float(np.max(raw_tensor))
        if max_val > 1.0:
            raw_tensor = raw_tensor / 10000.0
        raw_tensor = np.clip(raw_tensor, 0.0, 1.0).astype(np.float32)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": int(grid_10m.width),
        "height": int(grid_10m.height),
        "count": 6,
        "crs": expected_crs,
        "transform": grid_10m.transform,
        "descriptions": (
            "T0_B4",
            "T0_B8",
            "T0_B11",
            "T1_B4",
            "T1_B8",
            "T1_B11",
        ),
    }

    return raw_tensor, profile


def rasterizar_capa_geobosques_minam(
    gdf: "gpd.GeoDataFrame",
    grid: GridSpec,
    rule: LabelRule | None = None,
    *,
    campo_evento: str = "deforest",
    all_touched: bool = False,
    dtype: str = "uint8",
) -> LabelMask:
    """
    Rasteriza capas vectoriales de Geobosques (MINAM) mediante vecino más cercano (nearest neighbor).

    Garantiza una asignación categórica estricta sin interpolación continua entre clases de
    cobertura forestal o pérdida de bosque.

    Args:
        gdf: Vector GeoDataFrame de referencia de Geobosques MINAM.
        grid: Cuadrícula de destino GridSpec con CRS proyectado y transform definido.
        rule: Regla LabelRule explícita. Si es None, se genera automáticamente una regla
            basada en ``campo_evento``.
        campo_evento: Nombre del atributo a clasificar si ``rule`` es None.
        all_touched: Si True, incluye cualquier píxel tocado; si False (por defecto),
            emplea la regla de centro de píxel por vecino más cercano.
        dtype: Tipo entero ('uint8', 'int32').

    Returns:
        LabelMask rasterizada con vecino más cercano.
    """
    if rule is None:
        if campo_evento not in gdf.columns:
            posibles = ["perdida", "cambio", "bosque", "gridcode", "dn", "tipo", "id"]
            col_encontrada = next((c for c in posibles if c in gdf.columns), None)
            if col_encontrada is not None:
                campo_usar = col_encontrada
            else:
                campo_usar = next((c for c in gdf.columns if c != "geometry"), campo_evento)
        else:
            campo_usar = campo_evento

        valores_unicos = set(gdf[campo_usar].dropna().unique().tolist())
        mapping = {}
        for i, val in enumerate(sorted(valores_unicos, key=str), start=1):
            mapping[val] = i

        rule = LabelRule(
            field=campo_usar,
            mapping=mapping if mapping else {1: 1},
            background_value=0,
            allow_unmapped=True,
            unmapped_value=0,
        )

    return rasterize_reference(gdf, grid, rule, all_touched=all_touched, dtype=dtype)

