"""
Pruebas automatizadas de la base GIS (``src/preprocess.py``).

Todas las pruebas usan **datos sintéticos** creados en ``tmp_path``: nunca
dependen de imágenes Sentinel-2 ni de vectores reales de Geobosques.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import geopandas as gpd
from shapely.geometry import Polygon, box

from src import preprocess as pp


# ---------------------------------------------------------------------------
# Utilidades y fixtures
# ---------------------------------------------------------------------------
CRS_UTM = CRS.from_epsg(32718)  # UTM 18S (Huánuco)
CRS_WGS84 = CRS.from_epsg(4326)

# Cuadrícula de 10 m, 8x8 píxeles, esquina superior izquierda conocida.
TRANSFORM_10M = from_origin(400000.0, 8900000.0, 10.0, 10.0)
WIDTH = HEIGHT = 8


def _write_tif(
    path,
    array,
    *,
    transform=TRANSFORM_10M,
    crs=CRS_UTM,
    nodata=None,
    descriptions=None,
    dtype=None,
):
    """Escribe un GeoTIFF sintético (2D o 3D) con georreferenciación explícita."""
    array = np.asarray(array)
    data = array[np.newaxis, :, :] if array.ndim == 2 else array
    count, height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "crs": crs,
        "transform": transform,
        "dtype": dtype or array.dtype.name,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        if descriptions is not None:
            dst.descriptions = tuple(descriptions)
        dst.write(data.astype(profile["dtype"]))
    return path


@pytest.fixture
def grid():
    """Cuadrícula de referencia 8x8 a 10 m en EPSG:32718."""
    return pp.GridSpec(crs=CRS_UTM, transform=TRANSFORM_10M, width=WIDTH, height=HEIGHT)


@pytest.fixture
def s2_triplet(tmp_path):
    """Tres GeoTIFF monobanda B4/B8/B11 en la *misma* cuadrícula de 10 m."""
    rng = np.random.default_rng(42)
    paths = {}
    for name, dtype in (("B4", "uint16"), ("B8", "uint16"), ("B11", "uint16")):
        arr = rng.integers(100, 5000, size=(HEIGHT, WIDTH)).astype(dtype)
        paths[name] = _write_tif(tmp_path / f"{name}.tif", arr)
    return paths


# ---------------------------------------------------------------------------
# Apertura y validación de rásteres
# ---------------------------------------------------------------------------
def test_open_geotiff_reports_metadata(tmp_path):
    arr = np.arange(HEIGHT * WIDTH, dtype="uint16").reshape(HEIGHT, WIDTH)
    path = _write_tif(tmp_path / "band.tif", arr, nodata=0, descriptions=["B8"])

    profile = pp.open_geotiff(path)

    assert profile.crs is not None and profile.crs.to_epsg() == 32718
    assert profile.width == WIDTH and profile.height == HEIGHT
    assert profile.count == 1
    assert profile.res == (10.0, 10.0)
    assert profile.nodata == 0
    assert profile.descriptions == ("B8",)
    assert profile.is_georeferenced is True
    assert profile.shape == (HEIGHT, WIDTH)
    # El perfil debe ser serializable a JSON.
    assert profile.to_dict()["res"] == [10.0, 10.0]


def test_open_geotiff_missing_file_raises():
    with pytest.raises(pp.InvalidRasterError):
        pp.open_geotiff("/tmp/opencode/no_existe_geoforest.tif")


def test_validate_geotiff_requires_crs(tmp_path):
    arr = np.ones((HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(tmp_path / "sin_crs.tif", arr, crs=None)

    # open_geotiff es permisivo...
    assert pp.open_geotiff(path).crs is None
    # ...pero validate_geotiff exige CRS por defecto.
    with pytest.raises(pp.MissingCRSError):
        pp.validate_geotiff(path)


def test_validate_geotiff_crs_mismatch(tmp_path):
    arr = np.ones((HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(tmp_path / "band.tif", arr)

    with pytest.raises(pp.CRSMismatchError):
        pp.validate_geotiff(path, expected_crs="EPSG:4326")

    assert pp.validate_geotiff(path, expected_crs="EPSG:32718").crs.to_epsg() == 32718


def test_validate_geotiff_bands(tmp_path):
    arr = np.ones((3, HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(
        tmp_path / "stack.tif", arr, descriptions=["B4", "B8", "B11"]
    )

    perf = pp.validate_geotiff(
        path, expected_bands=("B4", "B8", "B11"), expected_band_count=3
    )
    assert perf.count == 3

    with pytest.raises(pp.BandError):
        pp.validate_geotiff(path, expected_bands=("B4", "B8", "B12"))
    with pytest.raises(pp.BandError):
        pp.validate_geotiff(path, expected_band_count=5)


def test_validate_geotiff_bands_without_descriptions_raises(tmp_path):
    arr = np.ones((3, HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(tmp_path / "stack.tif", arr)  # sin descripciones

    with pytest.raises(pp.BandError):
        pp.validate_geotiff(path, expected_bands=("B4", "B8", "B11"))


def test_validate_geotiff_resolution(tmp_path):
    arr = np.ones((HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(tmp_path / "band.tif", arr)

    assert pp.validate_geotiff(path, expected_resolution=10.0)
    with pytest.raises(pp.ResolutionMismatchError):
        pp.validate_geotiff(path, expected_resolution=(20.0, 20.0))


def test_gridspec_from_profile_without_crs_raises(tmp_path):
    arr = np.ones((HEIGHT, WIDTH), dtype="uint16")
    path = _write_tif(tmp_path / "sin_crs.tif", arr, crs=None)
    with pytest.raises(pp.MissingCRSError):
        pp.GridSpec.from_profile(pp.open_geotiff(path))


def test_gridspec_matches(grid):
    otra = pp.GridSpec(crs=CRS_UTM, transform=TRANSFORM_10M, width=WIDTH, height=HEIGHT)
    assert grid.matches(otra)

    distinto_crs = pp.GridSpec(
        crs=CRS_WGS84, transform=TRANSFORM_10M, width=WIDTH, height=HEIGHT
    )
    assert not grid.matches(distinto_crs)

    otra_forma = pp.GridSpec(
        crs=CRS_UTM, transform=TRANSFORM_10M, width=WIDTH + 1, height=HEIGHT
    )
    assert not grid.matches(otra_forma)


# ---------------------------------------------------------------------------
# Remuestreo: continuo vs categórico
# ---------------------------------------------------------------------------
def test_resolve_resampling_policy():
    policy = pp.ResamplingPolicy.spectral()
    assert pp.resolve_resampling(policy, pp.DataKind.CONTINUO) is not None
    assert (
        pp.resolve_resampling(policy, pp.DataKind.CATEGORICO)
        is not None
    )
    # Continuo usa bilineal y categórico nearest por defecto.
    assert pp.resolve_resampling(policy, pp.DataKind.CONTINUO) != pp.resolve_resampling(
        policy, pp.DataKind.CATEGORICO
    )
    assert pp.resolve_resampling(None, pp.DataKind.CONTINUO) is None

    with pytest.raises(TypeError):
        pp.resolve_resampling("bilinear", pp.DataKind.CONTINUO)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Preparación de bandas Sentinel-2
# ---------------------------------------------------------------------------
def test_prepare_s2_bands_same_grid(s2_triplet):
    prepared = pp.prepare_s2_bands(s2_triplet)

    assert set(prepared.band_names) == {"B4", "B8", "B11"}
    for arr in prepared.arrays.values():
        assert arr.shape == (HEIGHT, WIDTH)
        assert arr.dtype == np.float32
    assert prepared.grid.shape == (HEIGHT, WIDTH)
    assert prepared.valid_mask.shape == (HEIGHT, WIDTH)
    # Sin remuestreo: todo 'none' y sin píxeles inválidos.
    assert set(prepared.resampling_applied.values()) == {"none"}
    assert prepared.valid_mask.all()
    assert prepared.stack().shape == (3, HEIGHT, WIDTH)


def test_prepare_s2_bands_multiband_by_index(tmp_path):
    stack = np.stack(
        [
            np.full((HEIGHT, WIDTH), 10, dtype="uint16"),
            np.full((HEIGHT, WIDTH), 20, dtype="uint16"),
            np.full((HEIGHT, WIDTH), 30, dtype="uint16"),
        ]
    )
    path = _write_tif(
        tmp_path / "stack.tif", stack, descriptions=["B4", "B8", "B11"]
    )
    prepared = pp.prepare_s2_bands(
        {"B4": (path, 1), "B8": (path, 2), "B11": (path, 3)}
    )
    assert prepared.arrays["B4"].mean() == pytest.approx(10.0)
    assert prepared.arrays["B11"].mean() == pytest.approx(30.0)
    assert prepared.sources["B8"].band == 2


def test_prepare_s2_bands_missing_required(s2_triplet):
    parcial = {"B4": s2_triplet["B4"], "B8": s2_triplet["B8"]}
    with pytest.raises(pp.BandError):
        pp.prepare_s2_bands(parcial)


def test_prepare_s2_bands_nodata_propagates(tmp_path):
    b4 = np.full((HEIGHT, WIDTH), 100, dtype="uint16")
    b8 = np.full((HEIGHT, WIDTH), 200, dtype="uint16")
    b11 = np.full((HEIGHT, WIDTH), 300, dtype="uint16")
    b11[0, 0] = 0  # píxel NoData
    paths = {
        "B4": _write_tif(tmp_path / "B4.tif", b4),
        "B8": _write_tif(tmp_path / "B8.tif", b8),
        "B11": _write_tif(tmp_path / "B11.tif", b11, nodata=0),
    }
    prepared = pp.prepare_s2_bands(paths)

    assert np.isnan(prepared.arrays["B11"][0, 0])
    assert not prepared.valid_mask[0, 0]
    assert prepared.valid_mask.sum() == HEIGHT * WIDTH - 1


def test_prepare_s2_bands_scale_factor(tmp_path):
    b4 = np.full((HEIGHT, WIDTH), 2000, dtype="uint16")
    b8 = np.full((HEIGHT, WIDTH), 4000, dtype="uint16")
    b11 = np.full((HEIGHT, WIDTH), 3000, dtype="uint16")
    paths = {
        "B4": pp.RasterSource(tmp_path / "B4.tif", 1, scale_factor=1 / 10000),
        "B8": pp.RasterSource(tmp_path / "B8.tif", 1, scale_factor=1 / 10000),
        "B11": pp.RasterSource(tmp_path / "B11.tif", 1, scale_factor=1 / 10000),
    }
    for name, src in paths.items():
        arr = {"B4": b4, "B8": b8, "B11": b11}[name]
        _write_tif(src.path, arr)

    prepared = pp.prepare_s2_bands(paths)
    assert prepared.arrays["B8"].mean() == pytest.approx(0.4)


def test_prepare_s2_bands_grid_mismatch_without_reference(tmp_path):
    # B4/B8 a 10 m (8x8) y B11 a 20 m (4x4) sobre la misma extensión.
    b4 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b8 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b11 = np.ones((HEIGHT // 2, WIDTH // 2), dtype="uint16")
    paths = {
        "B4": _write_tif(tmp_path / "B4.tif", b4),
        "B8": _write_tif(tmp_path / "B8.tif", b8),
        "B11": _write_tif(
            tmp_path / "B11.tif", b11, transform=from_origin(400000, 8900000, 20, 20)
        ),
    }
    with pytest.raises(pp.GridMismatchError):
        pp.prepare_s2_bands(paths)


def test_prepare_s2_bands_grid_mismatch_requires_explicit_resampling(tmp_path):
    b4 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b8 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b11 = np.ones((HEIGHT // 2, WIDTH // 2), dtype="uint16")
    b4_path = _write_tif(tmp_path / "B4.tif", b4)
    b8_path = _write_tif(tmp_path / "B8.tif", b8)
    b11_path = _write_tif(
        tmp_path / "B11.tif", b11, transform=from_origin(400000, 8900000, 20, 20)
    )
    paths = {"B4": b4_path, "B8": b8_path, "B11": b11_path}

    # Con referencia pero sin método de remuestreo -> no se remuestrea en silencio.
    with pytest.raises(pp.GridMismatchError):
        pp.prepare_s2_bands(paths, reference=b8_path)

    # Con referencia y método explícito -> remuestreo aplicado.
    prepared = pp.prepare_s2_bands(
        paths, reference=b8_path, resampling=pp.ResamplingPolicy.spectral()
    )
    assert prepared.arrays["B11"].shape == (HEIGHT, WIDTH)
    assert prepared.resampling_applied["B11"] != "none"
    assert prepared.resampling_applied["B4"] == "none"
    assert np.nanmean(prepared.arrays["B11"]) == pytest.approx(1.0, abs=1e-3)


def test_prepare_s2_bands_enforce_grid_match(tmp_path):
    b4 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b8 = np.ones((HEIGHT, WIDTH), dtype="uint16")
    b11 = np.ones((HEIGHT // 2, WIDTH // 2), dtype="uint16")
    b8_path = _write_tif(tmp_path / "B8.tif", b8)
    paths = {
        "B4": _write_tif(tmp_path / "B4.tif", b4),
        "B8": b8_path,
        "B11": _write_tif(
            tmp_path / "B11.tif", b11, transform=from_origin(400000, 8900000, 20, 20)
        ),
    }
    with pytest.raises(pp.GridMismatchError):
        pp.prepare_s2_bands(
            paths,
            reference=b8_path,
            resampling=pp.ResamplingPolicy.spectral(),
            enforce_band_grid_match=True,
        )


def test_stack_bands_order(s2_triplet):
    prepared = pp.prepare_s2_bands(s2_triplet)
    stack = pp.stack_bands(prepared, order=["B11", "B4", "B8"])
    assert np.allclose(stack[0], prepared.arrays["B11"])
    assert np.allclose(stack[1], prepared.arrays["B4"])


# ---------------------------------------------------------------------------
# Índices espectrales
# ---------------------------------------------------------------------------
def _prepared_from_arrays(b4, b8, b11, grid):
    arrays = {
        "B4": np.asarray(b4, dtype="float32"),
        "B8": np.asarray(b8, dtype="float32"),
        "B11": np.asarray(b11, dtype="float32"),
    }
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    sources = {k: pp.RasterSource(path=Path("sintetico.tif"), band=1) for k in arrays}
    return pp.PreparedBands(
        arrays=arrays,
        grid=grid,
        valid_mask=valid,
        sources=sources,
        resampling_applied={k: "none" for k in arrays},
    )


def test_compute_indexes(grid):
    b4 = np.full((HEIGHT, WIDTH), 1000, dtype="float32")
    b8 = np.full((HEIGHT, WIDTH), 3000, dtype="float32")
    b11 = np.full((HEIGHT, WIDTH), 2000, dtype="float32")
    prepared = _prepared_from_arrays(b4, b8, b11, grid)

    ndvi = pp.compute_ndvi(prepared)
    nbr = pp.compute_nbr(prepared)
    assert ndvi[0, 0] == pytest.approx((3000 - 1000) / (3000 + 1000))
    assert nbr[0, 0] == pytest.approx((3000 - 2000) / (3000 + 2000))


def test_compute_ndvi_missing_band_raises(grid):
    prepared = _prepared_from_arrays(
        np.ones((HEIGHT, WIDTH)), np.ones((HEIGHT, WIDTH)), np.ones((HEIGHT, WIDTH)), grid
    )
    del prepared.arrays["B4"]
    with pytest.raises(pp.BandError):
        pp.compute_ndvi(prepared)


# ---------------------------------------------------------------------------
# Vectores de referencia
# ---------------------------------------------------------------------------
def _reference_gdf(crs=CRS_UTM):
    polys = [
        box(400000, 8899950, 400030, 8900000),
        box(400050, 8899950, 400080, 8900000),
    ]
    return gpd.GeoDataFrame(
        {"clase": ["bosque", "perdida"], "anio": [2019, 2021]},
        geometry=polys,
        crs=crs,
    )


def test_validate_reference_vector_summary():
    resumen = pp.validate_reference_vector(_reference_gdf())
    assert resumen["n_features"] == 2
    assert resumen["crs"].upper().startswith("EPSG:32718")
    assert resumen["n_geometrias_invalidas"] == 0
    assert "Polygon" in resumen["geom_types"]


def test_validate_reference_vector_reports_invalid_geometry():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])  # auto-intersección
    gdf = gpd.GeoDataFrame({"clase": ["x"]}, geometry=[bowtie], crs=CRS_UTM)
    resumen = pp.validate_reference_vector(gdf)
    assert resumen["n_geometrias_invalidas"] == 1


def test_validate_reference_vector_errors():
    with pytest.raises(pp.InvalidVectorError):
        pp.validate_reference_vector([1, 2, 3])  # type: ignore[arg-type]

    vacio = gpd.GeoDataFrame({"clase": []}, geometry=[], crs=CRS_UTM)
    with pytest.raises(pp.InvalidVectorError):
        pp.validate_reference_vector(vacio)

    sin_crs = gpd.GeoDataFrame({"clase": ["a"]}, geometry=[box(0, 0, 1, 1)])
    with pytest.raises(pp.MissingCRSError):
        pp.validate_reference_vector(sin_crs)

    con_nulo = gpd.GeoDataFrame({"clase": ["a"]}, geometry=[None], crs=CRS_UTM)
    with pytest.raises(pp.InvalidVectorError):
        pp.validate_reference_vector(con_nulo)

    with pytest.raises(pp.InvalidVectorError):
        pp.validate_reference_vector(_reference_gdf(), required_columns=["tipo"])


def test_read_and_prepare_reference_vector(tmp_path):
    path = tmp_path / "geobosques.gpkg"
    _reference_gdf().to_file(path, driver="GPKG")

    gdf = pp.read_reference_vector(path, required_columns=["clase"])
    assert len(gdf) == 2

    reparado = pp.prepare_reference_vector(path, required_columns=["clase"])
    assert reparado.crs.to_epsg() == 32718

    # Reproyección explícita a WGS84.
    wgs = pp.prepare_reference_vector(path, target_crs="EPSG:4326")
    assert wgs.crs.to_epsg() == 4326

    with pytest.raises(pp.InvalidVectorError):
        pp.read_reference_vector(tmp_path / "no_existe.gpkg")


def test_prepare_reference_vector_fixes_invalid_geometry(tmp_path):
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    gdf = gpd.GeoDataFrame({"clase": ["x"]}, geometry=[bowtie], crs=CRS_UTM)
    path = tmp_path / "invalido.geojson"
    gdf.to_file(path, driver="GeoJSON")

    crudo = pp.read_reference_vector(path)
    assert int((~crudo.geometry.is_valid).sum()) == 1

    reparado = pp.prepare_reference_vector(path, fix_invalid=True)
    assert reparado.geometry.is_valid.all()


# ---------------------------------------------------------------------------
# Rasterización de máscaras
# ---------------------------------------------------------------------------
def test_rasterize_reference_alignment(grid):
    rule = pp.LabelRule(
        field="clase",
        mapping={"bosque": 1, "perdida": 2},
        background_value=0,
    )
    mask = pp.rasterize_reference(_reference_gdf(), grid, rule)

    assert mask.array.shape == grid.shape
    assert mask.grid.transform == grid.transform
    assert mask.grid.crs == grid.crs
    assert set(mask.class_values) == {0, 1, 2}
    assert mask.counts()[1] > 0 and mask.counts()[2] > 0
    assert mask.counts()[0] > 0  # fondo


def test_rasterize_rule_is_explicit_and_required(grid):
    rule = pp.LabelRule(field="clase", mapping={"bosque": 1, "perdida": 2})
    mask = pp.rasterize_reference(_reference_gdf(), grid, rule)
    # El fondo por defecto es 0 y no representa ninguna clase temática.
    assert 0 in mask.class_values


def test_label_rule_empty_mapping_raises():
    with pytest.raises(pp.LabelRuleError):
        pp.LabelRule(field="clase", mapping={})


def test_rasterize_unmapped_value_raises(grid):
    gdf = _reference_gdf()
    gdf.loc[1, "clase"] = "otro"
    rule = pp.LabelRule(field="clase", mapping={"bosque": 1, "perdida": 2})
    with pytest.raises(pp.LabelRuleError):
        pp.rasterize_reference(gdf, grid, rule)


def test_rasterize_allow_unmapped(grid):
    gdf = _reference_gdf()
    gdf.loc[1, "clase"] = "otro"
    rule = pp.LabelRule(
        field="clase",
        mapping={"bosque": 1, "perdida": 2},
        allow_unmapped=True,
        unmapped_value=9,
    )
    mask = pp.rasterize_reference(gdf, grid, rule)
    assert 9 in mask.class_values


def test_rasterize_missing_field_raises(grid):
    rule = pp.LabelRule(field="tipo_evento", mapping={"bosque": 1})
    with pytest.raises(pp.InvalidVectorError):
        pp.rasterize_reference(_reference_gdf(), grid, rule)


def test_rasterize_crs_mismatch_raises(grid):
    gdf = _reference_gdf(crs=CRS_WGS84)
    rule = pp.LabelRule(field="clase", mapping={"bosque": 1, "perdida": 2})
    with pytest.raises(pp.CRSMismatchError):
        pp.rasterize_reference(gdf, grid, rule)


def test_rasterize_all_touched_changes_coverage(grid):
    rule = pp.LabelRule(field="clase", mapping={"bosque": 1, "perdida": 2})
    gdf = _reference_gdf()
    preciso = pp.rasterize_reference(gdf, grid, rule, all_touched=False)
    tocando = pp.rasterize_reference(gdf, grid, rule, all_touched=True)
    n_preciso = sum(v for k, v in preciso.counts().items() if k != 0)
    n_tocando = sum(v for k, v in tocando.counts().items() if k != 0)
    assert n_tocando >= n_preciso


# ---------------------------------------------------------------------------
# Persistencia y metadatos
# ---------------------------------------------------------------------------
def test_save_raster_roundtrip(tmp_path, grid):
    arr = np.arange(HEIGHT * WIDTH, dtype="float32").reshape(HEIGHT, WIDTH)
    out = pp.save_raster(
        arr, tmp_path / "sub" / "salida.tif", grid, nodata=np.nan
    )

    with rasterio.open(out) as ds:
        assert ds.crs.to_epsg() == 32718
        assert ds.transform == grid.transform
        assert ds.width == WIDTH and ds.height == HEIGHT
        assert np.isnan(ds.nodata)
        assert np.allclose(ds.read(1, masked=True).filled(np.nan), arr)


def test_save_raster_shape_mismatch_raises(tmp_path, grid):
    arr = np.ones((3, 3), dtype="float32")
    with pytest.raises(pp.InvalidRasterError):
        pp.save_raster(arr, tmp_path / "x.tif", grid)


def test_save_raster_multiband_descriptions(tmp_path, grid):
    arr = np.ones((3, HEIGHT, WIDTH), dtype="float32")
    out = pp.save_raster(
        arr, tmp_path / "stack.tif", grid, descriptions=["B4", "B8", "B11"]
    )
    with rasterio.open(out) as ds:
        assert ds.count == 3
        assert tuple(ds.descriptions) == ("B4", "B8", "B11")


def test_save_prepared_bands_and_metadata(tmp_path, s2_triplet):
    prepared = pp.prepare_s2_bands(s2_triplet)
    resultado = pp.save_prepared_bands(prepared, tmp_path / "processed")

    assert resultado["raster"].exists()
    assert resultado["metadata"] is not None and resultado["metadata"].exists()

    with rasterio.open(resultado["raster"]) as ds:
        assert ds.count == 3
        assert tuple(ds.descriptions) == ("B4", "B8", "B11")
        assert ds.crs.to_epsg() == 32718

    meta = pp.load_metadata(resultado["metadata"])
    assert meta["grid"]["crs"].upper().startswith("EPSG:32718")
    assert meta["resampling_aplicado"]["B4"] == "none"
    assert meta["bandas"] == ["B4", "B8", "B11"]


def test_save_label_mask_and_metadata(tmp_path, grid):
    rule = pp.LabelRule(
        field="clase",
        mapping={"bosque": 1, "perdida": 2},
        background_value=0,
    )
    mask = pp.rasterize_reference(_reference_gdf(), grid, rule)
    resultado = pp.save_label_mask(mask, tmp_path / "mask" / "mascara.tif")

    with rasterio.open(resultado["raster"]) as ds:
        assert np.issubdtype(np.dtype(ds.dtypes[0]), np.integer)
        assert ds.crs.to_epsg() == 32718
        assert ds.transform == grid.transform

    meta = pp.load_metadata(resultado["metadata"])
    assert meta["tipo"] == "mascara_etiquetas"
    assert meta["rule"]["mapping"] == {"bosque": 1, "perdida": 2}


def test_save_label_mask_rejects_float_dtype(tmp_path, grid):
    rule = pp.LabelRule(field="clase", mapping={"bosque": 1, "perdida": 2})
    mask = pp.rasterize_reference(_reference_gdf(), grid, rule)
    with pytest.raises(pp.InvalidRasterError):
        pp.save_label_mask(mask, tmp_path / "x.tif", dtype="float32")


def test_metadata_roundtrip(tmp_path):
    datos = {"grid": {"crs": "EPSG:32718"}, "nota": "prueba"}
    path = pp.save_metadata(datos, tmp_path / "meta.json")
    assert pp.load_metadata(path) == datos


def test_load_metadata_missing_file_raises(tmp_path):
    with pytest.raises(pp.InvalidRasterError):
        pp.load_metadata(tmp_path / "no_existe.json")
