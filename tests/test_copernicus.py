"""Pruebas unitarias para el servicio de descarga y catálogo de Copernicus / Sentinel-2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from app.services.copernicus import (
    HUANUCO_BOUNDS_UTM,
    _construir_info_escena_desde_id,
    consultar_catalogo_copernicus,
    descargar_banda_sentinel2,
    descargar_escena_sentinel2,
    descargar_par_sentinel2,
    main,
    sanitize_scene_id,
    validate_cog_url,
    validate_output_directory,
)
from src.dataset import discover_sentinel2_scenes, load_huanuco_sentinel2_scene


def _crear_banda_mock_geotiff(path: Path, shape: tuple[int, int] = (256, 256), value: int = 1000) -> None:
    """Crea un GeoTIFF válido sintético para pruebas unitarias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(370580.0, 8951940.0, 10.0, 10.0)
    data = np.full(shape, value, dtype=np.uint16)
    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 1,
        "dtype": "uint16",
        "crs": CRS.from_epsg(32718),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


# ---------------------------------------------------------------------------
# Seguridad y Validación de Entradas
# ---------------------------------------------------------------------------


def test_sanitize_scene_id_valido() -> None:
    assert sanitize_scene_id("S2A_18LUQ_20240911_0_L2A") == "S2A_18LUQ_20240911_0_L2A"
    assert sanitize_scene_id("2024-07-18_S2B_18LUQ_20240718_0_L2A") == "2024-07-18_S2B_18LUQ_20240718_0_L2A"


@pytest.mark.parametrize(
    "invalido",
    [
        "",
        "   ",
        "../../etc/passwd",
        "escena/con/slash",
        "escena\\con\\backslash",
        "escena\x00nulo",
        "escena con espacios",
        "a" * 130,  # Supera límite de longitud
        "escena; rm -rf /",
    ],
)
def test_sanitize_scene_id_rechaza_inyecciones(invalido: str) -> None:
    with pytest.raises(ValueError):
        sanitize_scene_id(invalido)


def test_validate_cog_url_valida() -> None:
    url_aws = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/L/UQ/2024/9/S2A/B04.tif"
    assert validate_cog_url(url_aws) == url_aws

    url_copernicus = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json"
    assert validate_cog_url(url_copernicus) == url_copernicus


@pytest.mark.parametrize(
    "url_insegura",
    [
        "http://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",  # No HTTPS
        "file:///etc/passwd",
        "ftp://sentinel-cogs.s3.amazonaws.com/B04.tif",
        "https://evil-server.com/malware.tif",
        "",
        "   ",
    ],
)
def test_validate_cog_url_rechaza_inseguras(url_insegura: str) -> None:
    with pytest.raises(ValueError):
        validate_cog_url(url_insegura)


def test_validate_output_directory_permite_dentro_del_proyecto(tmp_path: Path) -> None:
    base = tmp_path / "proyecto"
    data_dir = base / "data" / "raw" / "sentinel2"
    valido = validate_output_directory(data_dir, base_dir=base)
    assert valido == data_dir.resolve()


def test_validate_output_directory_bloquea_escape(tmp_path: Path) -> None:
    base = tmp_path / "proyecto"
    fuera = Path("/var/log/fuera_del_proyecto")
    with pytest.raises(ValueError, match="fuera del directorio permitido"):
        validate_output_directory(fuera, base_dir=base)


# ---------------------------------------------------------------------------
# Reconstrucción de Metadatos y Consulta STAC
# ---------------------------------------------------------------------------


def test_construir_info_escena_desde_id() -> None:
    info = _construir_info_escena_desde_id("S2A_18LUQ_20240911_0_L2A")
    assert info["id"] == "S2A_18LUQ_20240911_0_L2A"
    assert info["fecha"] == "2024-09-11"
    assert info["tile"] == "18LUQ"
    assert "B04.tif" in info["url_b04"]
    assert "B08.tif" in info["url_b08"]
    assert "B11.tif" in info["url_b11"]
    assert "sentinel-cogs.s3.us-west-2.amazonaws.com" in info["url_b04"]


def test_consultar_catalogo_copernicus_mocked() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "features": [
            {
                "id": "S2A_18LUQ_20240911_0_L2A",
                "properties": {
                    "datetime": "2024-09-11T15:20:10Z",
                    "platform": "Sentinel-2A",
                    "grid:code": "MGRS-18LUQ",
                    "eo:cloud_cover": 9.25,
                },
                "assets": {
                    "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"},
                    "nir": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"},
                    "swir16": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B11.tif"},
                },
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        resultados = consultar_catalogo_copernicus()
        assert len(resultados) == 1
        item = resultados[0]
        assert item["id"] == "S2A_18LUQ_20240911_0_L2A"
        assert item["fecha"] == "2024-09-11"
        assert item["nubes_pct"] == 9.25
        assert item["url_b04"] == "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"


def test_consultar_catalogo_copernicus_falla_red() -> None:
    with patch("requests.post", side_effect=Exception("Timeout de conexión")):
        res = consultar_catalogo_copernicus()
        assert res == []


# ---------------------------------------------------------------------------
# Descarga y Estructura Exacta para src/dataset.py
# ---------------------------------------------------------------------------


def test_descargar_banda_sentinel2_mocked(tmp_path: Path) -> None:
    out_file = tmp_path / "B04.tif"
    url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif"

    # Simular lectura de COG remoto con rasterio.open
    mock_src = MagicMock()
    mock_src.transform = from_origin(300000.0, 9000000.0, 10.0, 10.0)
    mock_src.read.return_value = np.full((256, 256), 650, dtype=np.uint16)
    mock_src.profile = {
        "driver": "GTiff",
        "height": 10980,
        "width": 10980,
        "crs": CRS.from_epsg(32718),
        "transform": mock_src.transform,
        "count": 1,
        "dtype": "uint16",
    }

    real_open = rasterio.open

    def mock_rasterio_open(path_or_url, mode="r", **kwargs):
        if isinstance(path_or_url, str) and path_or_url.startswith("https://"):
            return MagicMock(__enter__=MagicMock(return_value=mock_src), __exit__=MagicMock())
        return real_open(path_or_url, mode, **kwargs)

    with patch("rasterio.open", side_effect=mock_rasterio_open):
        descargado = descargar_banda_sentinel2(
            url,
            out_file,
            crop_bounds=HUANUCO_BOUNDS_UTM,
            target_shape=(256, 256),
        )

        assert descargado.is_file()
        with rasterio.open(descargado) as ds:
            assert ds.shape == (256, 256)
            assert ds.dtypes[0] == "uint16"


def test_descargar_escena_sentinel2_estructura_exacta(tmp_path: Path) -> None:
    escena_dict = {
        "id": "S2A_18LUQ_20240911_0_L2A",
        "fecha": "2024-09-11",
        "url_b04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
        "url_b08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif",
        "url_b11": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B11.tif",
    }

    def mock_dl_band(url, out_path, **kwargs):
        _crear_banda_mock_geotiff(Path(out_path))
        return Path(out_path)

    with patch("app.services.copernicus.descargar_banda_sentinel2", side_effect=mock_dl_band):
        directorio_escena = descargar_escena_sentinel2(
            escena_dict,
            output_dir=tmp_path,
        )

        # Verificar formato exacto YYYY-MM-DD_S2...
        assert directorio_escena.name == "2024-09-11_S2A_18LUQ_20240911_0_L2A"
        assert (directorio_escena / "B04.tif").is_file()
        assert (directorio_escena / "B08.tif").is_file()
        assert (directorio_escena / "B11.tif").is_file()


def test_descargar_par_sentinel2_y_compatibilidad_con_dataset(tmp_path: Path) -> None:
    e_t0 = {
        "id": "S2B_18LUQ_20240718_0_L2A",
        "fecha": "2024-07-18",
        "url_b04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
        "url_b08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif",
        "url_b11": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B11.tif",
    }
    e_t1 = {
        "id": "S2A_18LUQ_20240911_0_L2A",
        "fecha": "2024-09-11",
        "url_b04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
        "url_b08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif",
        "url_b11": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B11.tif",
    }

    def mock_dl_band(url, out_path, **kwargs):
        _crear_banda_mock_geotiff(Path(out_path))
        return Path(out_path)

    with patch("app.services.copernicus.descargar_banda_sentinel2", side_effect=mock_dl_band):
        p_t0, p_t1 = descargar_par_sentinel2(e_t0, e_t1, output_dir=tmp_path)

        assert p_t0.name == "2024-07-18_S2B_18LUQ_20240718_0_L2A"
        assert p_t1.name == "2024-09-11_S2A_18LUQ_20240911_0_L2A"

        # Verificar que discover_sentinel2_scenes de src/dataset lo encuentra
        descubiertas = discover_sentinel2_scenes(tmp_path)
        assert len(descubiertas) == 2
        assert descubiertas[0] == p_t0.name
        assert descubiertas[1] == p_t1.name

        # Verificar que load_huanuco_sentinel2_scene carga el par sin errores
        img, mask = load_huanuco_sentinel2_scene(sentinel2_root=tmp_path, geobosques_shp=tmp_path / "no_shp.shp")
        assert img.shape == (6, 256, 256)
        assert mask.shape == (256, 256)


def test_copernicus_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "Gestor de catálogo y descarga" in captured.out
