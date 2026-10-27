"""Wave boundary conditions for HurryWave.

Two forcing modes are supported:

* **timeseries** – bulk wave parameters (Hs, Tp, wave direction, directional
  spreading) as ASCII time-series files (``.bhs``, ``.btp``, ``.bwd``,
  ``.bds``).  This mirrors the pattern used for SnapWave boundary conditions
  in hydromt_sfincs and stores data as an ``xarray.Dataset`` with dimensions
  ``(time, index)`` backed by a ``GeoDataset``.

* **spectra** – 2-D wave energy spectra stored in a single netCDF file
  (``.bsp``) with dimensions ``(time, stations, theta, sigma)``.  The spectra
  are stored as a plain ``xarray.Dataset`` (not a GeoDataset) because the
  spatial location information is already embedded in the dataset variables
  ``station_x`` and ``station_y``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from hydromt import hydromt_step
from hydromt.gis.vector import GeoDataset
from hydromt.model.components import ModelComponent
from pyproj import Transformer

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")

# Variable names for the time-series forcing mode
_TS_VARNAMES = ["hs", "tp", "wd", "ds"]

# Mapping variable name → config key for the time-series files
_TS_FILES = {
    "hs": "bhsfile",
    "tp": "btpfile",
    "wd": "bwdfile",
    "ds": "bdsfile",
}
_TS_DEFAULT_FILES = {
    "hs": "hurrywave.bhs",
    "tp": "hurrywave.btp",
    "wd": "hurrywave.bwd",
    "ds": "hurrywave.bds",
}


class HurrywaveBoundaryConditions(ModelComponent):
    """Wave boundary conditions component for HurryWave.

    Supports two mutually exclusive modes selected via the ``forcing``
    attribute:

    ``"timeseries"`` (default)
        Bulk wave parameters (Hs, Tp, Wd, Ds) stored as an
        ``xarray.Dataset`` with dimensions ``(time, index)`` plus spatial
        geometry embedded via ``GeoDataset``.

    ``"spectra"``
        2-D energy spectra stored as an ``xarray.Dataset`` with dimensions
        ``(time, stations, theta, sigma)``.  Point locations come from the
        ``station_x`` / ``station_y`` variables in the dataset.
    """

    # Variable names used in timeseries mode
    _default_varnames: List[str] = _TS_VARNAMES

    def __init__(self, model: "HurrywaveModel"):
        self._data: Optional[xr.Dataset] = None
        # "timeseries" or "spectra"
        self.forcing: str = "timeseries"
        super().__init__(model=model)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> xr.Dataset:
        """Internal xarray Dataset.  Calls ``read()`` on first access."""
        if self._data is None:
            self._initialize()
        return self._data

    def _initialize(self, skip_read: bool = False) -> None:
        # if self._data is None:
        self._data = xr.Dataset()
        if self.root.is_reading_mode() and not skip_read:
            self.read()

    @property
    def nr_points(self) -> int:
        """Number of boundary points."""
        # if self._data is None:
        #     return 0
        # if self.data is an empty xr.Dataset, return 0
        if len(self.data.dims) == 0:
            return 0
        if self.forcing == "timeseries":
            return int(self.data.sizes.get("index", 0))
        else:
            return int(self.data.sizes.get("stations", 0))

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        """Point locations as a GeoDataFrame.

        In *timeseries* mode this is derived from the embedded ``GeoDataset``
        geometry.  In *spectra* mode coordinates are read from the
        ``station_x`` / ``station_y`` variables in the dataset.
        """
        if self.nr_points == 0:
            return gpd.GeoDataFrame()
        if self.forcing == "timeseries":
            return self.data.vector.to_gdf().reset_index(drop=True)
        else:
            xs = self.data["station_x"].values
            ys = self.data["station_y"].values
            names = list(self.data["stations"].values)
            points = [shapely.geometry.Point(x, y) for x, y in zip(xs, ys)]
            return gpd.GeoDataFrame(
                {"name": names}, geometry=points, crs=self.model.crs
            )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def read(self) -> None:
        """Read boundary data from files, choosing mode based on config."""
        # Determine mode from config: if bspfile is set → spectra
        bspfile = self.model.config.get("bspfile")
        if bspfile is not None:
            # Boundary spectra take precedence if bspfile is set, even if bhsfile etc. are also set
            self.forcing = "spectra"
            # For CoSMoS, we also need to read the boundary points from the .bnd file,
            # even though locations are in the .bsp file
            print("reading bnd file")
            self._read_boundary_points()
            self._read_boundary_spectra()
        else:
            self.forcing = "timeseries"
            self._read_boundary_points()
            self._read_boundary_timeseries()

    def write(self) -> None:
        """Write boundary data to files."""
        if self.nr_points == 0:
            return
        self._write_boundary_points()
        if self.forcing == "timeseries":
            self._write_boundary_timeseries()
        else:
            self._write_boundary_spectra()

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _read_boundary_points(self) -> None:
        """Read ``.bnd`` file (x y coordinates, one point per line)."""
        abs_path = self.model.config.get_set_file_variable("bndfile")
        if abs_path is None or not abs_path.exists():
            return

        logger.info(f"Reading boundary points from {abs_path}")
        df = pd.read_csv(str(abs_path), index_col=False, header=None, sep=r"\s+", names=["x", "y"])

        gdf = gpd.GeoDataFrame(
            {"name": [str(i + 1).zfill(4) for i in range(len(df))]},
            geometry=[shapely.geometry.Point(row.x, row.y) for _, row in df.iterrows()],
            crs=self.model.crs,
        )
        # Initialise dataset with zero time series spanning model period
        tstart, tstop = self.model.get_model_time()
        time = pd.date_range(tstart, tstop, periods=2)
        self._data = self._make_empty_timeseries_dataset(gdf, time)

    def _read_boundary_timeseries(self) -> None:
        """Read the four bulk-parameter time-series files."""
        tref = self.model.config.get("tref")
        if tref is None or self.nr_points == 0:
            return

        for var, cfg_key in _TS_FILES.items():
            abs_path = self.model.config.get_set_file_variable(cfg_key)
            if abs_path is None or not abs_path.exists():
                continue
            logger.info(f"Reading boundary {var} from {abs_path}")
            df = _read_timeseries_file(str(abs_path), tref)
            da = xr.DataArray(
                df.values,
                dims=("time", "index"),
                coords={"time": df.index, "index": np.arange(df.shape[1])},
                name=var,
            )
            self._data[var] = da

    def _read_boundary_spectra(self) -> None:
        """Read 2-D spectral boundary conditions from a netCDF file."""
        abs_path = self.model.config.get_set_file_variable("bspfile")
        if abs_path is None or not abs_path.exists():
            return

        logger.info(f"Reading boundary spectra from {abs_path}")
        self._data = xr.open_dataset(str(abs_path))

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_boundary_points(self) -> None:
        """Write boundary point coordinates to ``.bnd`` file."""
        abs_path = self.model.config.get_set_file_variable(
            "bndfile", default="hurrywave.bnd"
        )
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing boundary points to {abs_path}")
        gdf = self.gdf
        with open(str(abs_path), "w") as fid:
            for _, row in gdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                if self.model.crs.is_geographic:
                    fid.write(f"{x:12.6f}{y:12.6f}\n")
                else:
                    fid.write(f"{x:12.1f}{y:12.1f}\n")

    def _write_boundary_timeseries(self) -> None:
        """Write Hs, Tp, Wd, Ds time-series files."""
        tref = self.model.config.get("tref")
        if tref is None:
            raise ValueError("tref must be set in config before writing boundary conditions.")

        for var, cfg_key in _TS_FILES.items():
            if var not in self._data:
                continue
            abs_path = self.model.config.get_set_file_variable(
                cfg_key, default=_TS_DEFAULT_FILES[var]
            )
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing boundary {var} to {abs_path}")
            da = self._data[var]
            seconds = (da.time.values.astype("datetime64[s]").astype(float) -
                       np.datetime64(tref, "s").astype(float))
            df = pd.DataFrame(da.values, index=seconds)
            _write_timeseries_file(df, str(abs_path))

    def _write_boundary_spectra(self) -> None:
        """Write 2-D boundary spectra to a netCDF file."""
        abs_path = self.model.config.get_set_file_variable(
            "bspfile", default="hurrywave.bsp"
        )
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing boundary spectra to {abs_path}")
        tref = self.model.config.get("tref")
        encoding = {}
        if tref is not None:
            encoding["time"] = {"units": f"seconds since {tref.strftime('%Y%m%d %H%M%S')}"}
        self._data.to_netcdf(str(abs_path), mode="w", encoding=encoding)

    # ------------------------------------------------------------------
    # Setters (public API)
    # ------------------------------------------------------------------

    @hydromt_step
    def set_timeseries(
        self,
        df: pd.DataFrame,
        gdf: Optional[gpd.GeoDataFrame] = None,
        varname: str = None,
    ) -> None:
        """Set or update a bulk wave parameter time series.

        Parameters
        ----------
        df : pd.DataFrame
            Timeseries with DatetimeIndex (rows = times, columns = point indices).
        gdf : GeoDataFrame, optional
            Point locations.  Required if no locations have been set yet.
        varname : str, optional
            One of ``"hs"``, ``"tp"``, ``"wd"``, ``"ds"``.
            Defaults to ``"hs"``.
        """
        self.forcing = "timeseries"
        varname = varname or "hs"

        if gdf is not None:
            tstart, tstop = self.model.get_model_time()
            time = pd.date_range(tstart, tstop, periods=2)
            self._data = self._make_empty_timeseries_dataset(gdf, time)

        if self.nr_points == 0:
            raise ValueError("Provide gdf to set_timeseries when no points exist yet.")

        tref = self.model.config.get("tref")
        if df.index.inferred_type in ("integer", "floating"):
            if tref is None:
                raise ValueError("tref must be set in config to convert numeric index to datetime.")
            df.index = tref + pd.to_timedelta(df.index, unit="s")

        da = xr.DataArray(
            df.values,
            dims=("time", "index"),
            coords={"time": df.index, "index": np.arange(df.shape[1])},
            name=varname,
        )
        self._data[varname] = da

    @hydromt_step
    def set_uniform(
        self,
        hs: float = 1.0,
        tp: float = 8.0,
        wd: float = 270.0,
        ds: float = 30.0,
        gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """Set spatially and temporally uniform wave boundary conditions.

        Parameters
        ----------
        hs, tp, wd, ds : float
            Uniform values for each wave parameter.
        gdf : GeoDataFrame, optional
            Boundary point locations.  Required on first call.
        """
        self.forcing = "timeseries"

        if gdf is not None or self.nr_points == 0:
            if gdf is None:
                raise ValueError("gdf is required when no boundary points have been set.")
            tstart, tstop = self.model.get_model_time()
            time = pd.date_range(tstart, tstop, periods=2)
            self._data = self._make_empty_timeseries_dataset(gdf, time)

        tstart, tstop = self.model.get_model_time()
        time = [tstart, tstop]
        n = self.nr_points

        for var, val in zip(_TS_VARNAMES, [hs, tp, wd, ds]):
            da = xr.DataArray(
                np.full((2, n), val),
                dims=("time", "index"),
                coords={"time": time, "index": np.arange(n)},
                name=var,
            )
            self._data[var] = da

    @hydromt_step
    def set_spectra(self, ds: xr.Dataset) -> None:
        """Set 2-D spectral boundary conditions from an xarray Dataset.

        The dataset must contain:

        * ``point_spectrum2d`` – dimensions ``(time, stations, theta, sigma)``
        * ``station_x``, ``station_y`` – 1-D coordinate arrays (``stations``)
        * coordinates ``time``, ``sigma``, ``theta``

        Parameters
        ----------
        ds : xr.Dataset
            Spectral dataset following the HurryWave ``.bsp`` conventions.
        """
        required = {"point_spectrum2d", "station_x", "station_y"}
        missing = required - set(ds.data_vars) - set(ds.coords)
        if missing:
            raise ValueError(f"Dataset is missing variables/coords: {missing}")
        self.forcing = "spectra"
        self._data = ds

    @hydromt_step
    def add_point(
        self,
        x: float,
        y: float,
        name: str = None,
        hs: float = 1.0,
        tp: float = 8.0,
        wd: float = 270.0,
        ds: float = 30.0,
    ) -> None:
        """Add a single boundary point with default uniform wave conditions.

        Parameters
        ----------
        x, y : float
            Coordinates in the model CRS.
        name : str, optional
            Point name (auto-generated if omitted).
        hs, tp, wd, ds : float
            Default wave parameter values for the new point.
        """
        if self.forcing == "spectra":
            raise ValueError("Cannot add individual points in spectra mode.")

        if name is None:
            name = str(self.nr_points + 1).zfill(4)

        gdf = gpd.GeoDataFrame(
            {"name": [name]},
            geometry=[shapely.geometry.Point(x, y)],
            crs=self.model.crs,
        )

        if self.nr_points == 0:
            tstart, tstop = self.model.get_model_time()
            time = pd.date_range(tstart, tstop, periods=2)
            self._data = self._make_empty_timeseries_dataset(gdf, time)
            # fill with provided values
            for var, val in zip(_TS_VARNAMES, [hs, tp, wd, ds]):
                self._data[var][:, 0] = val
        else:
            # Rebuild dataset with the additional point
            existing_gdf = self.gdf
            combined_gdf = gpd.GeoDataFrame(
                pd.concat([existing_gdf, gdf], ignore_index=True),
                crs=self.model.crs,
            )
            tstart, tstop = self.model.get_model_time()
            time = pd.date_range(tstart, tstop, periods=2)

            # Preserve existing values
            old_n = self.nr_points
            n_time = 2
            old_values = {}
            for var in _TS_VARNAMES:
                if var in self._data:
                    da = self._data[var]
                    # Ensure dimension order is (time, index)
                    if "time" in da.dims and "index" in da.dims:
                        da = da.transpose("time", "index")
                    old_values[var] = da.values.copy()

            self._data = self._make_empty_timeseries_dataset(combined_gdf, time)
            for var, val in zip(_TS_VARNAMES, [hs, tp, wd, ds]):
                if var in old_values:
                    self._data[var].values[:, :old_n] = old_values[var]
                self._data[var].values[:, old_n] = val

    def delete(self, index: Union[int, List[int]]) -> None:
        """Delete one or more boundary points by index.

        Parameters
        ----------
        index : int or list of int
            Indices to remove (0-based).
        """
        if self.forcing == "spectra":
            raise ValueError("Cannot delete individual points in spectra mode.")
        if self.nr_points == 0:
            return
        if not isinstance(index, list):
            index = [index]
        remaining = [i for i in range(self.nr_points) if i not in index]
        self._data = self._data.isel(index=remaining).assign_coords(
            index=np.arange(len(remaining))
        )

    def clear(self) -> None:
        """Remove all boundary data."""
        self._data = xr.Dataset()
        self.forcing = "timeseries"

    # ------------------------------------------------------------------
    # Boundary-point generation from grid mask
    # ------------------------------------------------------------------

    @hydromt_step
    def get_boundary_points_from_mask(
        self,
        bnd_dist: float = 50000.0,
        min_dist: Optional[float] = None,
    ) -> None:
        """Derive boundary point locations from the mask (cells with value 2).

        Points are distributed at roughly *bnd_dist* metre intervals along the
        boundary polylines.  Existing time-series data are reset.

        Parameters
        ----------
        bnd_dist : float
            Target spacing between boundary points (metres).
        min_dist : float, optional
            Maximum distance to the next cell to be considered part of the
            same polyline.  Defaults to ``2 * dx``.
        """
        mask = self.model.quadtree_mask.data.mask
        if mask is None:
            raise ValueError("Mask must be loaded before generating boundary points.")

        grid = self.model.quadtree_grid.data
        ibnd = np.where(mask.values == 2)[0]
        if min_dist is None and len(ibnd) > 0:
            # Two cell sizes of the boundary cells, in model CRS units.
            # xugrid exposes cell areas as Ugrid2d.area (not face_areas).
            areas = np.asarray(grid.grid.area)
            cell_size = float(np.sqrt(np.median(areas[ibnd])))
            min_dist = 2.0 * cell_size

        xy = grid.grid.face_coordinates
        xp, yp = xy[ibnd, 0], xy[ibnd, 1]

        used = np.zeros(len(xp), dtype=bool)
        polylines = []

        while not np.all(used):
            i1 = np.where(~used)[0][0]
            used[i1] = True
            polyline = [i1]

            for direction in ("forward", "backward"):
                while True:
                    if np.all(used):
                        break
                    idx_unused = np.where(~used)[0]
                    dst = np.hypot(xp[idx_unused] - xp[i1], yp[idx_unused] - yp[i1])
                    inear = np.nanargmin(dst)
                    if dst[inear] < min_dist:
                        inear_all = idx_unused[inear]
                        if direction == "forward":
                            polyline.append(inear_all)
                        else:
                            polyline.insert(0, inear_all)
                        used[inear_all] = True
                        i1 = inear_all
                    else:
                        break
                if direction == "forward":
                    i1 = polyline[0]

            if len(polyline) > 1:
                polylines.append(polyline)

        if self.model.crs.is_geographic:
            transformer = Transformer.from_crs(self.model.crs, 3857, always_xy=True)

        gdf_list = []
        ip = 0
        for polyline in polylines:
            x = xp[polyline]
            y = yp[polyline]
            pts = [(xi, yi) for xi, yi in zip(x, y)]
            line = shapely.geometry.LineString(pts)
            if self.model.crs.is_geographic:
                xm, ym = transformer.transform(x, y)
                line_m = shapely.geometry.LineString(list(zip(xm, ym)))
                num_pts = int(line_m.length / bnd_dist) + 2
            else:
                num_pts = int(line.length / bnd_dist) + 2

            for i in range(num_pts):
                pt = line.interpolate(i / float(num_pts - 1), normalized=True)
                gdf_list.append(
                    {"name": str(ip + 1).zfill(4), "geometry": pt}
                )
                ip += 1

        gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        tstart, tstop = self.model.get_model_time()
        time = pd.date_range(tstart, tstop, periods=2)
        self._data = self._make_empty_timeseries_dataset(gdf, time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_empty_timeseries_dataset(
        self, gdf: gpd.GeoDataFrame, time: pd.DatetimeIndex
    ) -> xr.Dataset:
        """Create a zero-filled Dataset backed by GeoDataset geometry."""
        n = len(gdf)
        idx = np.arange(n)
        das = []
        for var in _TS_VARNAMES:
            das.append(
                xr.DataArray(
                    np.zeros((len(time), n)),
                    dims=("time", "index"),
                    coords={"time": time, "index": idx},
                    name=var,
                )
            )
        ds = xr.merge(das)
        ds = GeoDataset.from_gdf(gdf, ds, keep_cols=True)
        return ds.assign_coords(index=np.arange(ds.sizes["index"]))


# ---------------------------------------------------------------------------
# File-level I/O helpers
# ---------------------------------------------------------------------------

def _read_timeseries_file(filename: str, tref) -> pd.DataFrame:
    """Read a fixed-width time-series file (seconds-since-tref, columns = points)."""
    df = pd.read_csv(filename, index_col=0, header=None, sep=r"\s+")
    df.index = tref + pd.to_timedelta(df.index, unit="s")
    return df


def _write_timeseries_file(df: pd.DataFrame, filename: str, floatfmt: str = ".3f") -> None:
    """Write a DataFrame to a fixed-width time-series file."""
    from io import StringIO
    buf = StringIO()
    for t, row in zip(df.index, df.values):
        buf.write(f"{t:12.1f}")
        for v in row:
            buf.write(f"  {v:{floatfmt}}")
        buf.write("\n")
    with open(filename, "w") as fid:
        fid.write(buf.getvalue())
