import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu

from hydromt import hydromt_step
from hydromt.model.components import ModelComponent

from hydromt_hurrywave.workflows.map_overlay import MaskOverlay

np.warnings = warnings

if TYPE_CHECKING:
    from hydromt_hurrywave import HurrywaveModel

logger = logging.getLogger(f"hydromt.{__name__}")

# HurryWave mask values:
#   0 = inactive
#   1 = active
#   2 = open boundary
_MASK_VALUES = {"inactive": 0, "active": 1, "open_boundary": 2}


class HurrywaveQuadtreeMask(ModelComponent):
    def __init__(
        self,
        model: "HurrywaveModel",
    ):
        # Mask values live in model.quadtree_grid.data["mask"]
        super().__init__(model=model)
        # Lazy mask-overlay renderer (holds the only internal state; all
        # mask data itself lives on model.quadtree_grid.data["mask"]).
        self._overlay = MaskOverlay()

    @property
    def data(self):
        """Delegate to the quadtree grid dataset."""
        return self.model.quadtree_grid.data

    @property
    def empty_mask(self):
        """Return an all-zero mask with the same shape as the grid."""
        return self.model.quadtree_grid.empty_mask

    @property
    def face_coordinates(self):
        return self.model.quadtree_grid.face_coordinates

    @property
    def has_open_boundaries(self) -> bool:
        """Return True if any cell has mask value 2 (open boundary)."""
        if "mask" not in self.data:
            return False
        return bool((self.data["mask"] == 2).any())

    def read(self):
        # Mask values are read as part of the quadtree grid nc file.
        pass

    def write(self):
        # Mask values are written as part of the quadtree grid nc file.
        pass

    @hydromt_step
    def create(
        self,
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_zmin: float = None,
        exclude_zmax: float = None,
        open_boundary_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        open_boundary_zmin: float = None,
        open_boundary_zmax: float = None,
        all_touched: bool = False,
    ):
        """Setup active model cells and open-boundary cells.

        Parameters
        ----------
        zmin, zmax : float, optional
            Elevation thresholds for active cells.
        include_polygon, exclude_polygon : str, Path, or GeoDataFrame, optional
            Polygons to include/exclude from the active domain.
        include_zmin, include_zmax : float, optional
            Elevation bounds applied within ``include_polygon``.
        exclude_zmin, exclude_zmax : float, optional
            Elevation bounds applied within ``exclude_polygon``.
        open_boundary_polygon : str, Path, or GeoDataFrame, optional
            Polygon selecting open-boundary cells (mask = 2).
        open_boundary_zmin, open_boundary_zmax : float, optional
            Elevation bounds for open-boundary cells.
        all_touched : bool, optional
            Include a cell if it touches (rather than contains) a geometry,
            by default False.
        """
        if isinstance(include_polygon, gpd.GeoDataFrame) and include_polygon.empty:
            include_polygon = None
        if isinstance(exclude_polygon, gpd.GeoDataFrame) and exclude_polygon.empty:
            exclude_polygon = None
        if isinstance(open_boundary_polygon, gpd.GeoDataFrame) and open_boundary_polygon.empty:
            open_boundary_polygon = None

        self.create_active(
            zmin=zmin,
            zmax=zmax,
            include_polygon=include_polygon,
            include_zmin=include_zmin,
            include_zmax=include_zmax,
            exclude_polygon=exclude_polygon,
            exclude_zmin=exclude_zmin,
            exclude_zmax=exclude_zmax,
            all_touched=all_touched,
        )

        if open_boundary_polygon is not None:
            self.create_boundary(
                include_polygon=open_boundary_polygon,
                include_zmin=open_boundary_zmin,
                include_zmax=open_boundary_zmax,
                all_touched=all_touched,
            )

        self._overlay.invalidate()

    @hydromt_step
    def create_active(
        self,
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        exclude_zmin: float = None,
        exclude_zmax: float = None,
        all_touched: bool = False,
        reset_mask: bool = True,
    ):
        """Define active (mask=1) model cells.

        Active cells are those with valid elevation (not NaN), optionally
        filtered by elevation thresholds and include/exclude polygons.

        Parameters
        ----------
        zmin, zmax : float, optional
            Elevation range for active cells.
        include_polygon, exclude_polygon : str, Path, or GeoDataFrame, optional
            Polygons that force cells in/out of the active domain.
        include_zmin, include_zmax : float, optional
            Elevation range applied within ``include_polygon``.
        exclude_zmin, exclude_zmax : float, optional
            Elevation range applied within ``exclude_polygon``.
        all_touched : bool, optional
            Rasterisation option, by default False.
        reset_mask : bool, optional
            Reset any existing mask before setting active cells, by default True.
        """
        logger.info("Building HurryWave mask ...")

        gdf_include, gdf_exclude = None, None
        bbox = self.model.region.to_crs(4326).total_bounds if self.model.region is not None else None

        if include_polygon is not None:
            gdf_include = self._load_polygon(include_polygon, bbox)
        if exclude_polygon is not None:
            gdf_exclude = self._load_polygon(exclude_polygon, bbox)

        uda_mask = self.data["mask"] if "mask" in self.data else None
        uda_mask0 = None
        if not reset_mask and uda_mask is not None:
            uda_mask0 = uda_mask > 0

        uda_mask = self.empty_mask > 0  # all False

        if zmin is not None or zmax is not None:
            if "z" not in self.data:
                raise ValueError("z required in combination with zmin / zmax")
            uda_dep = self.data["z"]
            # cells without a valid elevation are never active
            # (note: ``!= np.nan`` is always True, hence isnan)
            _msk = ~np.isnan(uda_dep)
            if zmin is not None:
                _msk = np.logical_and(_msk, uda_dep >= zmin)
            if zmax is not None:
                _msk = np.logical_and(_msk, uda_dep <= zmax)
            if uda_mask0 is not None:
                uda_mask = np.logical_and(uda_mask0, _msk)
            else:
                uda_mask = _msk
        elif uda_mask0 is not None:
            uda_mask = uda_mask0

        if gdf_include is not None:
            _msk = (
                xu.burn_vector_geometry(
                    gdf_include, self.data, fill=0, all_touched=all_touched
                ) > 0
            )
            if _msk.any():
                if include_zmin is not None or include_zmax is not None:
                    if "z" not in self.data:
                        raise ValueError("z required with include_zmin / include_zmax")
                    uda_dep = self.data["z"]
                    if include_zmin is not None:
                        _msk = np.logical_and(_msk, uda_dep >= include_zmin)
                    if include_zmax is not None:
                        _msk = np.logical_and(_msk, uda_dep <= include_zmax)
                uda_mask = np.logical_or(uda_mask, _msk)
            else:
                logger.warning("No cells found within include polygon!")

        if gdf_exclude is not None:
            _msk = (
                xu.burn_vector_geometry(
                    gdf_exclude, self.data, fill=0, all_touched=all_touched
                ) > 0
            )
            if _msk.any():
                if exclude_zmin is not None or exclude_zmax is not None:
                    if "z" not in self.data:
                        raise ValueError("z required with exclude_zmin / exclude_zmax")
                    uda_dep = self.data["z"]
                    if exclude_zmin is not None:
                        _msk = np.logical_and(_msk, uda_dep >= exclude_zmin)
                    if exclude_zmax is not None:
                        _msk = np.logical_and(_msk, uda_dep <= exclude_zmax)
                uda_mask = np.logical_and(uda_mask, ~_msk)
            else:
                logger.warning("No cells found within exclude polygon!")

        self.data["mask"] = xu.UgridDataArray(
            xr.DataArray(
                data=uda_mask.astype(np.uint8),
                dims=[self.data.grid.face_dimension],
            ),
            self.data.grid,
        )

    @hydromt_step
    def create_boundary(
        self,
        zmin: float = None,
        zmax: float = None,
        include_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        include_zmin: float = None,
        include_zmax: float = None,
        exclude_polygon: Union[str, Path, gpd.GeoDataFrame] = None,
        all_touched: bool = True,
        reset_bounds: bool = True,
    ):
        """Set open-boundary cells (mask=2) at the edge of the active domain.

        Parameters
        ----------
        zmin, zmax : float, optional
            Elevation bounds for candidate boundary cells.
        include_polygon : str, Path, or GeoDataFrame, optional
            Only boundary cells inside this polygon are selected.
        include_zmin, include_zmax : float, optional
            Elevation bounds within ``include_polygon``.
        exclude_polygon : str, Path, or GeoDataFrame, optional
            Boundary cells inside this polygon are excluded.
        all_touched : bool, optional
            Rasterisation option, by default True.
        reset_bounds : bool, optional
            Reset existing boundary cells to active before setting new ones,
            by default True.
        """
        if "mask" not in self.data:
            raise ValueError("Create active mask first (create_active).")

        uda_mask = self.data["mask"]

        if "z" not in self.data and (zmin is not None or zmax is not None):
            raise ValueError("z required with zmin / zmax")

        if reset_bounds:
            logger.debug("Resetting existing open-boundary (mask=2) cells to active.")
            uda_mask = uda_mask.where(uda_mask != np.uint8(2), np.uint8(1))
            if (
                zmin is None
                and zmax is None
                and include_polygon is None
                and exclude_polygon is None
            ):
                self.data["mask"] = xu.UgridDataArray(
                    xr.DataArray(
                        data=uda_mask, dims=[self.data.grid.face_dimension]
                    ),
                    self.data.grid,
                )
                return

        bbox = self.model.region.to_crs(4326).total_bounds if self.model.region is not None else None

        gdf_include = self._load_polygon(include_polygon, bbox) if include_polygon is not None else None
        gdf_exclude = self._load_polygon(exclude_polygon, bbox) if exclude_polygon is not None else None

        bounds = self._find_boundary_cells("mask")

        if zmin is not None:
            bounds = np.logical_and(bounds, self.data["z"] >= zmin)
        if zmax is not None:
            bounds = np.logical_and(bounds, self.data["z"] <= zmax)

        if gdf_include is not None:
            uda_include = (
                xu.burn_vector_geometry(
                    gdf_include, self.data, fill=0, all_touched=all_touched
                ) > 0
            )
            if include_zmin is not None:
                uda_include = np.logical_and(uda_include, self.data["z"] >= include_zmin)
            if include_zmax is not None:
                uda_include = np.logical_and(uda_include, self.data["z"] <= include_zmax)
            bounds = np.logical_and(bounds, uda_include)
            if not bounds.any():
                logger.warning("No boundary cells found within include polygon!")
                return

        if gdf_exclude is not None:
            uda_exclude = (
                xu.burn_vector_geometry(
                    gdf_exclude, self.data, fill=0, all_touched=all_touched
                ) > 0
            )
            bounds = np.logical_and(bounds, ~uda_exclude)

        ncells = np.count_nonzero(bounds.values)
        if ncells > 0:
            uda_mask = uda_mask.where(~bounds, np.uint8(2))

        self.data["mask"] = xu.UgridDataArray(
            xr.DataArray(data=uda_mask, dims=[self.data.grid.face_dimension]),
            self.data.grid,
        )

    def to_gdf(self, option="all"):
        """Return a GeoDataFrame with a point per active mask cell.

        Parameters
        ----------
        option : str, optional
            Filter cells: ``'all'`` (default), ``'active'`` (mask=1),
            or ``'open_boundary'`` (mask=2).
        """
        nr_cells = self.model.quadtree_grid.data.sizes["mesh2d_nFaces"]
        if nr_cells == 0:
            return gpd.GeoDataFrame()

        xz, yz = self.face_coordinates
        mask = self.data["mask"]
        if option == "all":
            iok = np.where(mask > 0)
        elif option == "active":
            iok = np.where(mask == 1)
        elif option == "open_boundary":
            iok = np.where(mask == 2)
        else:
            iok = np.where(mask > -999)

        gdf_list = [
            {"geometry": shapely.geometry.Point(xz[i], yz[i])} for i in iok[0]
        ]
        if gdf_list:
            return gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        return gpd.GeoDataFrame(gdf_list)

    def clear_overlay(self) -> None:
        """Invalidate the cached mask-overlay dataframe.

        Call this whenever the grid or mask is rebuilt so the next
        :py:meth:`map_overlay` render picks up fresh values.
        """
        self._overlay.invalidate()

    def map_overlay(
        self,
        file_name,
        xlim=None,
        ylim=None,
        colors=None,
        px: int = 2,
        width: int = 800,
        **kwargs,
    ) -> bool:
        """Render a PNG mask overlay.

        One-line wrapper around
        :py:class:`hydromt_hurrywave.workflows.map_overlay.MaskOverlay`.
        """
        if self.model.quadtree_grid.data is None or "mask" not in self.data:
            return False
        return self._overlay.render(
            x=self.face_coordinates[0][:],
            y=self.face_coordinates[1][:],
            mask=self.data["mask"].values[:],
            source_crs=self.model.crs,
            file_name=file_name,
            xlim=xlim,
            ylim=ylim,
            colors=colors,
            px=px,
            width=width,
        )

    # ------------------------------------------------------------------ internal

    def _load_polygon(
        self,
        polygon: Union[str, Path, gpd.GeoDataFrame],
        bbox,
    ) -> gpd.GeoDataFrame:
        """Load a polygon from path, data catalog name, or GeoDataFrame."""
        if isinstance(polygon, gpd.GeoDataFrame):
            return polygon.to_crs(self.model.crs)
        gdf = (
            self.data_catalog.get_geodataframe(polygon, bbox=bbox)
            .explode(index_parts=False)
            .reset_index(drop=True)
            .to_crs(self.model.crs)
        )
        if not gdf.geometry.geom_type.isin(["Polygon"]).all():
            raise ValueError("Polygon must contain only Polygon geometries.")
        return gdf

    def _find_boundary_cells(self, varname: str = "mask"):
        """Return a boolean array marking edge cells of the active domain.

        Uses the quadtree adjacency arrays (mu/nu/md/nd) stored in the grid
        dataset for correct handling of cells at different refinement levels.
        Active boundary cells are those adjacent to at least one inactive cell
        or at the grid exterior.
        """
        mu1 = self.data["mu1"].values[:] - 1
        mu2 = self.data["mu2"].values[:] - 1
        nu = self.data["nu"].values[:]
        nu1 = self.data["nu1"].values[:] - 1
        nu2 = self.data["nu2"].values[:] - 1
        md = self.data["md"].values[:]
        md1 = self.data["md1"].values[:] - 1
        md2 = self.data["md2"].values[:] - 1
        nd = self.data["nd"].values[:]
        nd1 = self.data["nd1"].values[:] - 1
        nd2 = self.data["nd2"].values[:] - 1

        mask = self.data[varname].values[:]
        nr_cells = self.data.sizes["mesh2d_nFaces"]
        bounds = np.zeros(nr_cells, dtype=bool)

        # Check left neighbors
        left_coarser = md <= 0
        left_finer1 = (md1 >= 0) & (mask[md1] == 0)
        left_finer2 = (md2 >= 0) & (mask[md2] == 0)
        bounds |= (left_coarser & left_finer1) | (~left_coarser & (left_finer1 | left_finer2))

        # Check right neighbors
        right_finer1 = (mu1 >= 0) & (mask[mu1] == 0)
        right_finer2 = (mu2 >= 0) & (mask[mu2] == 0)
        bounds |= right_finer1 | right_finer2

        # Check bottom neighbors
        below_coarser = nd <= 0
        below_finer1 = (nd1 >= 0) & (mask[nd1] == 0)
        below_finer2 = (nd2 >= 0) & (mask[nd2] == 0)
        bounds |= (below_coarser & (below_finer1 | below_finer2)) | (~below_coarser & (below_finer1 | below_finer2))

        # Check top neighbors
        above_coarser = nu <= 0
        above_finer1 = (nu1 >= 0) & (mask[nu1] == 0)
        above_finer2 = (nu2 >= 0) & (mask[nu2] == 0)
        bounds |= (above_coarser & (above_finer1 | above_finer2)) | (~above_coarser & (above_finer1 | above_finer2))

        # Grid exterior cells are always boundary candidates
        bounds[md1 == -1] = True  # left edge
        bounds[mu1 == -1] = True  # right edge
        bounds[nd1 == -1] = True  # bottom edge
        bounds[nu1 == -1] = True  # top edge

        # Keep only active cells
        bounds[mask == 0] = False

        return bounds
