"""Tests for HurrywaveQuadtreeGrid, Elevation, and Mask components."""

import numpy as np
import pytest
import shapely
import shapely.ops
import xarray as xr
import xugrid as xu
from pyproj import CRS

from hydromt_hurrywave.components.quadtree.quadtree import HurrywaveQuadtreeGrid

from constants import DX, DY, EPSG, MMAX, NMAX, X0, Y0


class TestQuadtreeGrid:
    """Grid creation and properties."""

    def test_create_grid(self, model_with_grid):
        grid = model_with_grid.quadtree_grid
        assert grid.data is not None
        assert grid.data.sizes["mesh2d_nFaces"] > 0

    def test_grid_crs(self, model_with_grid):
        crs = model_with_grid.quadtree_grid.crs
        assert crs is not None
        assert crs == CRS.from_epsg(EPSG)

    def test_grid_exterior(self, model_with_grid):
        ext = model_with_grid.quadtree_grid.exterior
        assert ext is not None
        assert len(ext) > 0

    def test_grid_face_coordinates(self, model_with_grid):
        xc, yc = model_with_grid.quadtree_grid.face_coordinates
        n = model_with_grid.quadtree_grid.data.sizes["mesh2d_nFaces"]
        assert len(xc) == n
        assert len(yc) == n


class TestQuadtreeCreateIndexTiles:
    """Index-tile generation for the quadtree grid."""

    def test_create_index_tiles_png_and_html(self, model_with_mask, tmp_path):
        # Small grid (10 x 10 @ 1 km) — need enough zoom to resolve pixels.
        zoom_range = [12, 13]
        root = tmp_path / "tiles_png"
        model_with_mask.quadtree_grid.create_index_tiles(
            root=root, zoom_range=zoom_range
        )
        png_files = list((root / "indices").rglob("*.png"))
        assert len(png_files) > 0
        # <indices>/<zoom>/<x>/<y>.png
        zoom_levels = {int(p.parts[-3]) for p in png_files}
        assert zoom_levels.issubset(set(range(zoom_range[0], zoom_range[1] + 1)))
        # HTML viewer is written by default alongside PNG tiles
        html_file = root / "indices" / "index.html"
        assert html_file.is_file()
        assert "{z}/{x}/{y}.png" in html_file.read_text()

    def test_create_index_tiles_bin(self, model_with_mask, tmp_path):
        zoom_range = [12, 13]
        root = tmp_path / "tiles_bin"
        model_with_mask.quadtree_grid.create_index_tiles(
            root=root, zoom_range=zoom_range, fmt="bin"
        )
        dat_files = list((root / "indices").rglob("*.dat"))
        assert len(dat_files) > 0
        # Each .dat tile should be 256*256 int32 = 262144 bytes
        assert dat_files[0].stat().st_size == 256 * 256 * 4
        # No HTML viewer for bin format
        assert not (root / "indices" / "index.html").exists()


class TestQuadtreeGridIO:
    """Grid read/write round-trip."""

    def test_write_read_roundtrip(self, model_with_grid, tmp_path):
        mod = model_with_grid
        root = tmp_path / "grid_rt"
        root.mkdir()
        mod.root.set(str(root), mode="w")
        mod.quadtree_grid.write()
        mod.config.write()  # config must be written AFTER grid so qtrfile key is set

        from hydromt_hurrywave import HurrywaveModel

        mod2 = HurrywaveModel(root=str(root), mode="r")
        mod2.config.read()
        mod2.quadtree_grid.read()

        assert (
            mod2.quadtree_grid.data.sizes["mesh2d_nFaces"]
            == mod.quadtree_grid.data.sizes["mesh2d_nFaces"]
        )
        assert mod2.quadtree_grid.crs == mod.quadtree_grid.crs


class TestQuadtreeElevation:
    """Elevation creation and properties."""

    def test_uniform_elevation(self, model_with_grid):
        z = model_with_grid.quadtree_grid.data["z"]
        assert z is not None
        assert np.allclose(z.values, -10.0)

    def test_create_uniform_overwrites(self, model_with_grid):
        mod = model_with_grid
        mod.quadtree_elevation.create_uniform(zb=-20.0)
        z = mod.quadtree_grid.data["z"]
        assert np.allclose(z.values, -20.0)


class TestQuadtreeMask:
    """Mask creation and properties."""

    def test_create_mask(self, model_with_mask):
        mask = model_with_mask.quadtree_grid.data["mask"]
        assert mask is not None
        # All cells should be active (1) or boundary (2)
        assert (mask.values >= 0).all()

    def test_mask_has_active_cells(self, model_with_mask):
        mask = model_with_mask.quadtree_grid.data["mask"]
        assert (mask.values == 1).any()

    def test_to_gdf(self, model_with_mask):
        gdf = model_with_mask.quadtree_mask.to_gdf(option="active")
        assert len(gdf) > 0


def test_quadtree_exterior_matches_edge_polygonization():
    mesh = xu.Ugrid2d(
        node_x=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
        node_y=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        fill_value=-1,
        face_node_connectivity=np.array(
            [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
        ),
        crs="EPSG:32631",
    )
    grid_component = object.__new__(HurrywaveQuadtreeGrid)
    grid_component._data = xu.UgridDataset(xr.Dataset(), mesh)

    # Reference: the original (slow) per-edge implementation
    edge_nodes = mesh.edge_node_connectivity[mesh.exterior_edges, :]
    x = mesh.node_x[edge_nodes]
    y = mesh.node_y[edge_nodes]
    expected_lines = [
        shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))
    ]
    expected_merged = shapely.ops.linemerge(expected_lines)
    expected_polygons = list(shapely.ops.polygonize(expected_merged))

    exterior = grid_component.exterior

    assert exterior.crs == grid_component.crs
    assert len(exterior) == len(expected_polygons)
    assert shapely.union_all(exterior.geometry.array).equals(
        shapely.union_all(expected_polygons)
    )
    # Cached: the same object is returned on the next call
    assert grid_component.exterior is exterior


def test_quadtree_exterior_cache_refreshes_for_new_grid(model_with_grid):
    grid = model_with_grid.quadtree_grid
    first = grid.exterior
    assert grid.exterior is first
    # Building a new (larger) grid must not return the cached exterior
    grid.create(
        x0=X0, y0=Y0, nmax=2 * NMAX, mmax=2 * MMAX,
        dx=DX, dy=DY, rotation=0.0, epsg=EPSG,
    )
    second = grid.exterior
    assert second is not first
    assert second.geometry.area.sum() == pytest.approx(
        4.0 * first.geometry.area.sum()
    )


@pytest.fixture
def half_covered_elevation():
    """DEM covering only the western half of a 10 x 10 km grid (x < 5000 m)."""
    import hydromt  # noqa: F401 (registers the raster accessor)

    x = np.arange(-500.0, 5000.0, 50.0) + 25.0
    y = np.arange(-500.0, 10500.0, 50.0) + 25.0
    da = xr.DataArray(
        np.tile(-10.0 + x / 1000.0, (len(y), 1)),
        coords={"y": y, "x": x},
        dims=("y", "x"),
    )
    da.raster.set_crs(32631)
    da.raster.set_nodata(-9999.0)
    return da


@pytest.mark.parametrize("interpolate_na", [None, False, True])
def test_elevation_interpolate_na(tmp_path, half_covered_elevation, interpolate_na):
    from hydromt_hurrywave import HurrywaveModel

    model = HurrywaveModel(root=str(tmp_path), mode="w")
    model.quadtree_grid.create(0.0, 0.0, 10, 10, 1000.0, 1000.0, 0.0, epsg=32631)
    kwargs = {} if interpolate_na is None else {"interpolate_na": interpolate_na}
    model.quadtree_elevation.create([{"elevation": half_covered_elevation}], **kwargs)
    z = model.quadtree_grid.data["z"].values
    model.quadtree_mask.create(zmin=-20.0, zmax=10.0)
    active = model.quadtree_grid.data["mask"].values > 0
    if interpolate_na:
        # missing cells are filled with values close to the nearest data
        assert not np.any(np.isnan(z))
        assert active.all()
        assert np.nanmin(z) >= -10.0 and np.nanmax(z) <= -5.0
    else:
        # default: cells without data stay NaN and inactive
        assert int(np.isnan(z).sum()) == 50
        assert int(active.sum()) == 50
