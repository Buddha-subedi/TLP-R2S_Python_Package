
"""
TLP-R2S precipitation visualization.

Features
--------
1. Combined rainfall and snowfall using pcolormesh in one geographic subplot.
2. Separate vertical rain and snow colorbars.
3. Original precipitation colormaps and power scaling.
4. Invalid latitude/longitude filtering.
5. Longitude normalization to [-180, 180].
6. Dominant precipitation phase visualization.
7. Cartopy coastlines and country boundaries.
8. Mask discontinuous mesh cells across the international dateline.
9. Light-gray background for the complete valid SSMIS swath.
10. Annual statistics visualization.
11. High-resolution PNG output.

The functions preserve the original package API.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 1. ORIGINAL PLOTTING CONFIGURATION
# ============================================================

POWER = 0.4

SNOW_TICKS = [0.01, 0.15, 0.5, 1.0]
RAIN_TICKS = [0.1, 1.0, 3.0, 6.0]

SNOW_CMAP = LinearSegmentedColormap.from_list(
    "tlpr2s_snow",
    [
        "#E0E0E0",
        "#A9EAFE",
        "#00BFFF",
        "#0000FF",
        "#9400D3",
    ],
)

RAIN_CMAP = LinearSegmentedColormap.from_list(
    "tlpr2s_rain",
    [
        "gainsboro",
        "#008000",
        "#E3E025",
        "#FFB700",
        "#FF167F",
        "#8B0000",
    ],
)


# ============================================================
# 2. GEOLOCATION AND PRECIPITATION VALIDATION
# ============================================================

def _prepare_swath(
    longitude,
    latitude,
    rain,
    snow,
):
    """
    Validate and prepare the satellite swath.

    Invalid geolocation is excluded before plotting.

    Rain and snow are displayed using dominant-phase
    masking, consistent with the original implementation.
    """

    lon = np.asarray(
        longitude, dtype=np.float64
    ).copy()

    lat = np.asarray(
        latitude, dtype=np.float64
    ).copy()

    rain = np.asarray(
        rain, dtype=np.float64
    ).copy()

    snow = np.asarray(
        snow, dtype=np.float64
    ).copy()

    if not (
        lon.shape == lat.shape == rain.shape == snow.shape
    ):
        raise ValueError(
            "Plot shape mismatch: "
            f"lon={lon.shape}, "
            f"lat={lat.shape}, "
            f"rain={rain.shape}, "
            f"snow={snow.shape}"
        )

    if lon.ndim != 2:
        raise ValueError(
            "SSMIS swath arrays must be 2D."
        )

    # --------------------------------------------------------
    # Validate latitude and longitude
    # --------------------------------------------------------

    # Longitude values in [0, 360] are legitimate and
    # can be converted to [-180, 180].
    #
    # Values outside [-180, 360] are treated as invalid
    # rather than wrapping arbitrary fill values.

    valid_geo = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= -90.0)
        & (lat <= 90.0)
        & (lon >= -180.0)
        & (lon <= 360.0)
    )

    lon[valid_geo] = (
        (lon[valid_geo] + 180.0) % 360.0
    ) - 180.0

    # Keep valid dateline observations. Mesh discontinuities are
    # handled separately rather than deleting an entire longitude band.
    lon[~valid_geo] = np.nan
    lat[~valid_geo] = np.nan

    valid_count = np.count_nonzero(valid_geo)

    print(
        f"Valid geolocation pixels: "
        f"{valid_count:,} / {lat.size:,}",
        flush=True,
    )

    if valid_count == 0:
        raise ValueError(
            "No valid geographic coordinates remain "
            "after filtering."
        )

    # --------------------------------------------------------
    # Precipitation validation
    # --------------------------------------------------------

    rain[~np.isfinite(rain)] = np.nan
    snow[~np.isfinite(snow)] = np.nan

    # Preserve the original rainfall threshold.
    rain[rain < 0.05] = 0.0

    # Original snowfall threshold was zero.
    snow[snow < 0.0] = 0.0

    # Rain-dominant pixels.
    rain_valid = (
        valid_geo
        & np.isfinite(rain)
        & (rain > 0)
        & (
            ~np.isfinite(snow)
            | (rain >= snow)
        )
    )

    # Snow-dominant pixels.
    snow_valid = (
        valid_geo
        & np.isfinite(snow)
        & (snow > 0)
        & (
            ~np.isfinite(rain)
            | (snow > rain)
        )
    )

    print(
        f"Rain-dominant pixels: "
        f"{np.count_nonzero(rain_valid):,}",
        flush=True,
    )

    print(
        f"Snow-dominant pixels: "
        f"{np.count_nonzero(snow_valid):,}",
        flush=True,
    )

    return (
        lon,
        lat,
        rain,
        snow,
        rain_valid,
        snow_valid,
    )


# ============================================================
# 3. CONTINUOUS SWATH MESH HELPERS
# ============================================================

def _mesh_coordinates(lon, lat):
    """Return finite coordinates and a mask for unsafe swath cells.

    pcolormesh requires finite coordinate arrays even when its data is
    masked. Invalid coordinates are filled solely for mesh construction;
    their corresponding data cells remain masked.
    """
    from scipy.ndimage import distance_transform_edt

    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise ValueError("No valid swath geolocation")

    # Fill invalid centers with their nearest valid coordinate.
    if np.all(valid):
        lon_safe = lon.copy()
        lat_safe = lat.copy()
    else:
        indices = distance_transform_edt(~valid, return_distances=False,
                                         return_indices=True)
        lon_safe = lon[tuple(indices)].copy()
        lat_safe = lat[tuple(indices)].copy()

    # Suppress cells neighboring discontinuous coordinates. A 180-degree
    # longitude jump is a dateline seam, not a genuine footprint width.
    unsafe = ~valid.copy()
    for axis in (0, 1):
        lon_jump = np.abs(np.diff(lon_safe, axis=axis)) > 180.0
        lat_jump = np.abs(np.diff(lat_safe, axis=axis)) > 20.0
        jumps = lon_jump | lat_jump
        if axis == 0:
            unsafe[:-1, :] |= jumps
            unsafe[1:, :] |= jumps
        else:
            unsafe[:, :-1] |= jumps
            unsafe[:, 1:] |= jumps

    return lon_safe, lat_safe, unsafe


def _draw_swath_mesh(ax, lon, lat, values, *, cmap, vmin, vmax,
                     zorder, gamma=POWER):
    """Draw swath pixels as a continuous QuadMesh, not a scatter plot."""
    import cartopy.crs as ccrs

    if gamma is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin ** gamma, vmax=vmax ** gamma)
        values = np.ma.power(values, gamma)

    return ax.pcolormesh(
        lon, lat, values, shading="nearest", cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), rasterized=True, zorder=zorder,
        linewidth=0, antialiased=False,
    )


def save_combined_map(
    path,
    longitude,
    latitude,
    rain,
    snow,
    title,
    dpi=300,
):
    """
    Save one geographic map containing both rain and snow.

    Two vertical colorbars are displayed on the right:
        Upper: Snowfall rate
        Lower: Rainfall rate

    The original color palettes and power transformation
    are retained.
    """

    import cartopy.crs as ccrs

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Prepare swath
    # --------------------------------------------------------

    (
        lon,
        lat,
        rain,
        snow,
        rain_valid,
        snow_valid,
    ) = _prepare_swath(
        longitude,
        latitude,
        rain,
        snow,
    )

    # --------------------------------------------------------
    # Figure settings
    # --------------------------------------------------------

    with mpl.rc_context(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "black",
        }
    ):

        fig = plt.figure(
            figsize=(14, 8)
        )

        try:

            ax = fig.add_axes(
                [0.07, 0.12, 0.79, 0.78],
                projection=ccrs.PlateCarree(
                    central_longitude=0
                ),
            )

            # ------------------------------------------------
            # Geographic extent
            # ------------------------------------------------

            ax.set_global()

            ax.set_extent(
                [-180, 180, -90, 90],
                crs=ccrs.PlateCarree(),
            )

            # ------------------------------------------------
            # Full swath background and precipitation QuadMeshes
            # ------------------------------------------------
            mesh_lon, mesh_lat, unsafe = _mesh_coordinates(lon, lat)
            valid_mesh = ~unsafe
            print(f"Safe pcolormesh pixels: {np.count_nonzero(valid_mesh):,}",
                  flush=True)

            # Gray cells indicate valid satellite coverage, including
            # locations with no retrieved precipitation.
            swath_values = np.ma.array(
                np.ones(lat.shape, dtype=np.float32), mask=unsafe
            )
            _draw_swath_mesh(
                ax, mesh_lon, mesh_lat, swath_values,
                cmap=mcolors.ListedColormap(["#DADADA"]),
                vmin=0, vmax=1, zorder=1, gamma=None,
            )

            snow_values = np.ma.array(
                snow, mask=unsafe | ~snow_valid
            )
            rain_values = np.ma.array(
                rain, mask=unsafe | ~rain_valid
            )

            if np.any(snow_valid & valid_mesh):
                _draw_swath_mesh(
                    ax, mesh_lon, mesh_lat, snow_values,
                    cmap=SNOW_CMAP, vmin=0, vmax=1.0,
                    zorder=2,
                )
            else:
                warnings.warn("No valid snowfall pixels found.")

            if np.any(rain_valid & valid_mesh):
                _draw_swath_mesh(
                    ax, mesh_lon, mesh_lat, rain_values,
                    cmap=RAIN_CMAP, vmin=0, vmax=6.0,
                    zorder=3,
                )
            else:
                warnings.warn("No valid rainfall pixels found.")

            # ------------------------------------------------
            # Independent colorbar mappables
            # ------------------------------------------------

            # These ensure both colorbars are available
            # even if one precipitation type is absent.

            snow_sm = mpl.cm.ScalarMappable(
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=1.0 ** POWER,
                ),
                cmap=SNOW_CMAP,
            )

            snow_sm.set_array([])

            rain_sm = mpl.cm.ScalarMappable(
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=6.0 ** POWER,
                ),
                cmap=RAIN_CMAP,
            )

            rain_sm.set_array([])

            # ------------------------------------------------
            # Coastlines and country borders
            # ------------------------------------------------

            try:

                import cartopy.io.shapereader as shpreader

                coast = shpreader.natural_earth(
                    resolution="110m",
                    category="physical",
                    name="coastline",
                )

                border = shpreader.natural_earth(
                    resolution="110m",
                    category="cultural",
                    name="admin_0_boundary_lines_land",
                )

                ax.add_geometries(
                    shpreader.Reader(coast).geometries(),
                    ccrs.PlateCarree(),
                    edgecolor="#404040",
                    facecolor="none",
                    linewidth=0.4,
                    zorder=5,
                )

                ax.add_geometries(
                    shpreader.Reader(border).geometries(),
                    ccrs.PlateCarree(),
                    edgecolor="#606060",
                    facecolor="none",
                    linewidth=0.3,
                    linestyle=":",
                    zorder=5,
                )

            except Exception as exc:

                warnings.warn(
                    "Natural Earth coastline/border "
                    f"unavailable: {exc}. "
                    "Map saved without these features."
                )

            # ------------------------------------------------
            # Gridlines
            # ------------------------------------------------

            gl = ax.gridlines(
                draw_labels=True,
                xlocs=np.arange(-180, 181, 60),
                ylocs=np.arange(-90, 91, 20),
                color="gray",
                alpha=0.3,
                linestyle="--",
                linewidth=0.8,
            )

            gl.top_labels = False
            gl.right_labels = False

            gl.xlabel_style = {
                "size": 12,
                "weight": "bold",
                "color": "#404040",
            }

            gl.ylabel_style = {
                "size": 12,
                "weight": "bold",
                "color": "#404040",
            }

            # ------------------------------------------------
            # Title and frame
            # ------------------------------------------------

            ax.set_title(
                title,
                fontsize=18,
                fontweight="bold",
                pad=10,
                color="#202020",
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#404040")

            # ------------------------------------------------
            # Two original vertical colorbars
            # ------------------------------------------------

            cax_snow = fig.add_axes(
                [0.86, 0.55, 0.01, 0.30]
            )

            cax_rain = fig.add_axes(
                [0.86, 0.20, 0.01, 0.30]
            )

            # Snow colorbar
            cb_snow = fig.colorbar(
                snow_sm,
                cax=cax_snow,
            )

            cb_snow.set_label(
                "Snow Rate (mm hr⁻¹)",
                fontsize=12,
                fontweight="bold",
                color="#202020",
                labelpad=10,
            )

            cb_snow.set_ticks(
                [v ** POWER for v in SNOW_TICKS]
            )

            cb_snow.ax.set_yticklabels(
                [f"{v:.2f}" for v in SNOW_TICKS],
                fontweight="bold",
            )

            # Rain colorbar
            cb_rain = fig.colorbar(
                rain_sm,
                cax=cax_rain,
            )

            cb_rain.set_label(
                "Rain Rate (mm hr⁻¹)",
                fontsize=12,
                fontweight="bold",
                color="#202020",
                labelpad=10,
            )

            cb_rain.set_ticks(
                [v ** POWER for v in RAIN_TICKS]
            )

            cb_rain.ax.set_yticklabels(
                [f"{v:.1f}" for v in RAIN_TICKS],
                fontweight="bold",
            )

            # Colorbar borders and ticks
            for cb in (cb_snow, cb_rain):

                cb.ax.tick_params(
                    labelsize=10,
                    width=1.5,
                    length=5,
                    colors="#404040",
                )

                for spine in cb.ax.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor("#404040")

            # ------------------------------------------------
            # Save atomically
            # ------------------------------------------------

            tmp = path.with_name(
                "." + path.stem + ".partial.png"
            )

            try:

                fig.savefig(
                    tmp,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )

                os.replace(
                    tmp,
                    path,
                )

            finally:

                if tmp.exists():
                    tmp.unlink()

            print(
                f"Saved orbital precipitation map: {path}",
                flush=True,
            )

        finally:
            plt.close(fig)


# ============================================================
# 4. PUBLIC ORBIT PLOTTING FUNCTION
# ============================================================

def plot_orbit(
    latitude,
    longitude,
    rain,
    snow,
    *,
    title="TLP-R2S Precipitation Rate",
    save=None,
    dpi=300,
):
    """
    Generate combined rain/snow orbital visualization.

    Parameters
    ----------
    latitude, longitude : 2D arrays
        Satellite geolocation.

    rain, snow : 2D arrays
        Retrieved precipitation rates in mm/hr.

    title : str
        Figure title.

    save : str or Path
        Output PNG path.

    dpi : int
        Figure resolution.

    Returns
    -------
    pathlib.Path
        Saved figure location.
    """

    if save is None:
        raise ValueError(
            "save= is required for the reference map writer"
        )

    path = save
    import cartopy.crs as ccrs
    import re

    path = Path(path)

    # Automatically extract six-digit orbit number
    # from the output filename.
    match = re.search(r"_(\d{6})$", path.stem)

    if match and title == "TLP-R2S Precipitation Rate":
        title = f"Orbit: #{match.group(1)}"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Prepare swath
    # --------------------------------------------------------

    (
        lon,
        lat,
        rain,
        snow,
        rain_valid,
        snow_valid,
    ) = _prepare_swath(
        longitude,
        latitude,
        rain,
        snow,
    )

    # --------------------------------------------------------
    # Figure settings
    # --------------------------------------------------------

    with mpl.rc_context(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 12,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "black",
        }
    ):

        fig = plt.figure(
            figsize=(14, 8)
        )

        try:

            ax = fig.add_axes(
                [0.07, 0.12, 0.79, 0.78],
                projection=ccrs.PlateCarree(
                    central_longitude=0
                ),
            )

            # ------------------------------------------------
            # Geographic extent
            # ------------------------------------------------

            ax.set_global()

            ax.set_extent(
                [-180, 180, -90, 90],
                crs=ccrs.PlateCarree(),
            )

            # ------------------------------------------------
            # Full swath background and precipitation QuadMeshes
            # ------------------------------------------------
            mesh_lon, mesh_lat, unsafe = _mesh_coordinates(lon, lat)
            polar = np.isfinite(lon) & np.isfinite(lat) & (lat <= -80.0)
            mesh_mask = unsafe | polar
            valid_mesh = ~mesh_mask
            scatter_size = 0.5  # marker area in points squared

            scatter_kwargs = dict(
                s=scatter_size, marker="s", linewidths=0,
                edgecolors="none", transform=ccrs.PlateCarree(),
                rasterized=True,
            )

            if np.any(valid_mesh):
                _draw_swath_mesh(
                    ax, mesh_lon, mesh_lat,
                    np.ma.array(np.ones(lat.shape, dtype=np.float32), mask=mesh_mask),
                    cmap=mcolors.ListedColormap(["#DADADA"]),
                    vmin=0, vmax=1, zorder=1, gamma=None,
                )
            if np.any(polar):
                ax.scatter(lon[polar], lat[polar], c="#DADADA",
                           zorder=1, **scatter_kwargs)

            for values, valid, cmap, vmax, zorder, phase in (
                (snow, snow_valid, SNOW_CMAP, 1.0, 2, "snowfall"),
                (rain, rain_valid, RAIN_CMAP, 6.0, 3, "rainfall"),
            ):
                if np.any(valid & valid_mesh):
                    _draw_swath_mesh(
                        ax, mesh_lon, mesh_lat,
                        np.ma.array(values, mask=mesh_mask | ~valid),
                        cmap=cmap, vmin=0, vmax=vmax, zorder=zorder,
                    )
                selected = polar & valid
                if np.any(selected):
                    ax.scatter(
                        lon[selected], lat[selected],
                        c=values[selected] ** POWER, cmap=cmap,
                        norm=mcolors.Normalize(vmin=0, vmax=vmax ** POWER),
                        zorder=zorder, **scatter_kwargs,
                    )
                if not np.any(valid & (valid_mesh | polar)):
                    warnings.warn(f"No valid {phase} pixels found.")

            # ------------------------------------------------
            # Independent colorbar mappables
            # ------------------------------------------------

            # These ensure both colorbars are available
            # even if one precipitation type is absent.

            snow_sm = mpl.cm.ScalarMappable(
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=1.0 ** POWER,
                ),
                cmap=SNOW_CMAP,
            )

            snow_sm.set_array([])

            rain_sm = mpl.cm.ScalarMappable(
                norm=mcolors.Normalize(
                    vmin=0,
                    vmax=6.0 ** POWER,
                ),
                cmap=RAIN_CMAP,
            )

            rain_sm.set_array([])

            # ------------------------------------------------
            # Coastlines and country borders
            # ------------------------------------------------

            try:

                import cartopy.io.shapereader as shpreader

                coast = shpreader.natural_earth(
                    resolution="110m",
                    category="physical",
                    name="coastline",
                )

                border = shpreader.natural_earth(
                    resolution="110m",
                    category="cultural",
                    name="admin_0_boundary_lines_land",
                )

                ax.add_geometries(
                    shpreader.Reader(coast).geometries(),
                    ccrs.PlateCarree(),
                    edgecolor="#404040",
                    facecolor="none",
                    linewidth=0.4,
                    zorder=5,
                )

                ax.add_geometries(
                    shpreader.Reader(border).geometries(),
                    ccrs.PlateCarree(),
                    edgecolor="#606060",
                    facecolor="none",
                    linewidth=0.3,
                    linestyle=":",
                    zorder=5,
                )

            except Exception as exc:

                warnings.warn(
                    "Natural Earth coastline/border "
                    f"unavailable: {exc}. "
                    "Map saved without these features."
                )

            # ------------------------------------------------
            # Gridlines
            # ------------------------------------------------

            gl = ax.gridlines(
                draw_labels=True,
                xlocs=np.arange(-180, 181, 60),
                ylocs=np.arange(-90, 91, 20),
                color="gray",
                alpha=0.3,
                linestyle="--",
                linewidth=0.8,
            )

            gl.top_labels = False
            gl.right_labels = False

            gl.xlabel_style = {
                "size": 12,
                "weight": "bold",
                "color": "#404040",
            }

            gl.ylabel_style = {
                "size": 12,
                "weight": "bold",
                "color": "#404040",
            }

            # ------------------------------------------------
            # Title and frame
            # ------------------------------------------------

            ax.set_title(
                title,
                fontsize=18,
                fontweight="bold",
                pad=10,
                color="#202020",
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#404040")

            # ------------------------------------------------
            # Two original vertical colorbars
            # ------------------------------------------------

            cax_snow = fig.add_axes(
                [0.86, 0.55, 0.01, 0.30]
            )

            cax_rain = fig.add_axes(
                [0.86, 0.20, 0.01, 0.30]
            )

            # Snow colorbar
            cb_snow = fig.colorbar(
                snow_sm,
                cax=cax_snow,
            )

            cb_snow.set_label(
                "Snow Rate (mm hr⁻¹)",
                fontsize=12,
                fontweight="bold",
                color="#202020",
                labelpad=10,
            )

            cb_snow.set_ticks(
                [v ** POWER for v in SNOW_TICKS]
            )

            cb_snow.ax.set_yticklabels(
                [f"{v:.2f}" for v in SNOW_TICKS],
                fontweight="bold",
            )

            # Rain colorbar
            cb_rain = fig.colorbar(
                rain_sm,
                cax=cax_rain,
            )

            cb_rain.set_label(
                "Rain Rate (mm hr⁻¹)",
                fontsize=12,
                fontweight="bold",
                color="#202020",
                labelpad=10,
            )

            cb_rain.set_ticks(
                [v ** POWER for v in RAIN_TICKS]
            )

            cb_rain.ax.set_yticklabels(
                [f"{v:.1f}" for v in RAIN_TICKS],
                fontweight="bold",
            )

            # Colorbar borders and ticks
            for cb in (cb_snow, cb_rain):

                cb.ax.tick_params(
                    labelsize=10,
                    width=1.5,
                    length=5,
                    colors="#404040",
                )

                for spine in cb.ax.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor("#404040")

            # ------------------------------------------------
            # Save atomically
            # ------------------------------------------------

            tmp = path.with_name(
                "." + path.stem + ".partial.png"
            )

            try:

                fig.savefig(
                    tmp,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white",
                )

                os.replace(
                    tmp,
                    path,
                )

            finally:

                if tmp.exists():
                    tmp.unlink()

            print(
                f"Saved orbital precipitation map: {path}",
                flush=True,
            )

        finally:
            plt.close(fig)

    return Path(save)


# ============================================================
# 5. ANNUAL STATISTICS PLOTTING
# ============================================================

def _plot_map(
    ax,
    lon,
    lat,
    data,
    cmap,
    vmin=None,
    vmax=None,
):
    """
    Plot a gridded geographic field.
    """

    import cartopy.crs as ccrs

    ax.coastlines(
        linewidth=0.5
    )

    return ax.pcolormesh(
        lon,
        lat,
        data,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def _axes(figsize=(12, 5)):
    """
    Create Cartopy geographic axes.
    """

    import cartopy.crs as ccrs

    return plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.PlateCarree()
        },
    )


def plot_annual(
    statistics,
    field="mean_precip",
    *,
    save=None,
    show=False,
):
    """
    Plot annual sampled-swath statistics.

    Accepts an AnnualStatistics object or a saved NPZ.
    """

    from .annual import AnnualStatistics

    if isinstance(
        statistics,
        (str, Path),
    ):
        statistics = AnnualStatistics.load(
            statistics
        )

    allowed = {
        "mean_precip",
        "mean_rain",
        "mean_snow",
        "observation_count",
        "swath_coverage_count",
    }

    if field not in allowed:
        raise ValueError(
            f"field must be one of {sorted(allowed)}"
        )

    v = getattr(
        statistics,
        field,
    )

    fig, ax = _axes(
        figsize=(13, 5.5)
    )

    cmap = (
        "viridis"
        if "count" in field
        else "turbo"
    )

    mesh = _plot_map(
        ax,
        statistics.longitude,
        statistics.latitude,
        v,
        cmap,
    )

    ax.set_title(
        field.replace("_", " ").title()
        + " (sampled swaths)"
    )

    unit = (
        "count"
        if "count" in field
        else "mm h$^{-1}$"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label=unit,
        shrink=0.8,
    )

    fig.tight_layout()

    if save:
        fig.savefig(
            save,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, ax



def plot_annual_comparison(
    statistics,
    era5_snow,
    era5_rain,
    *,
    save=None,
    show=False,
    dpi=300,
    snow_vmax=0.105,
    rain_vmax=0.83,
    tlpr2s_snow_north_factor=1.0,
    tlpr2s_snow_south_factor=1.0,
    tlpr2s_rain_factor=1.0,
):
    """Plot a 2x2 TLP-R2S vs ERA5 mean-rate comparison.

    ERA5 inputs must be mean rates (mm/hr), aligned to the
    AnnualStatistics latitude/longitude grid.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    from .annual import AnnualStatistics

    if isinstance(statistics, (str, Path)):
        statistics = AnnualStatistics.load(statistics)

    lat = np.asarray(statistics.latitude, dtype=float)
    lon = np.asarray(statistics.longitude, dtype=float)
    shape = (lat.size, lon.size)

    era5_snow = np.asarray(era5_snow, dtype=float)
    era5_rain = np.asarray(era5_rain, dtype=float)

    if era5_snow.shape != shape or era5_rain.shape != shape:
        raise ValueError(
            f"ERA5 arrays must have shape {shape}; "
            f"got {era5_snow.shape} and {era5_rain.shape}"
        )

    tl_snow = np.asarray(statistics.mean_snow, dtype=float).copy()
    tl_rain = np.asarray(statistics.mean_rain, dtype=float).copy()

    tl_snow *= np.where(
        lat[:, None] >= 0,
        tlpr2s_snow_north_factor,
        tlpr2s_snow_south_factor,
    )
    tl_rain *= tlpr2s_rain_factor

    cmap_snow = LinearSegmentedColormap.from_list(
        "comparison_snow",
        ["white", "cyan", "blue", "blueviolet"],
        N=256,
    )
    cmap_rain = LinearSegmentedColormap.from_list(
        "comparison_rain",
        [
            "white", "darkgreen", "green", "yellow",
            "darkorange", "red", "magenta", "purple",
        ],
        N=256,
    )

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.035, right=0.88,
        bottom=0.06, top=0.91,
        wspace=0.015, hspace=0.025,
    )

    panels = [
        (tl_snow, cmap_snow, snow_vmax, "(a)"),
        (era5_snow, cmap_snow, snow_vmax, "(b)"),
        (tl_rain, cmap_rain, rain_vmax, "(c)"),
        (era5_rain, cmap_rain, rain_vmax, "(d)"),
    ]

    meshes = []

    for i, (values, cmap, vmax, panel) in enumerate(panels):
        ax = fig.add_subplot(
            gs[i // 2, i % 2],
            projection=ccrs.PlateCarree(),
        )

        mesh = ax.pcolormesh(
            lon, lat,
            np.ma.masked_invalid(values),
            vmin=0,
            vmax=vmax,
            cmap=cmap,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            shading="auto",
            rasterized=True,
        )
        meshes.append(mesh)

        ax.coastlines(alpha=0.3, linewidth=0.5)
        ax.set_extent(
            [-179.9, 179.9, -90, 90],
            crs=ccrs.PlateCarree(),
        )

        ax.text(
            0.89, 0.05, panel,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="right",
        )

        if i < 2:
            ax.set_title(
                ["TLP-R2S", "ERA5"][i],
                fontsize=12,
                fontweight="bold",
            )

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                boxstyle="round,pad=0.008",
                edgecolor="black",
                facecolor="none",
                linewidth=1.2,
            )
        )

        ax.gridlines(
            draw_labels=False,
            linestyle="--",
            alpha=0.3,
        )

    formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")

    for mesh_index, rect, label in [
        (0, [0.91, 0.55, 0.009, 0.28], "Snow (mm hr$^{-1}$)"),
        (2, [0.91, 0.14, 0.009, 0.28], "Rain (mm hr$^{-1}$)"),
    ]:
        cax = fig.add_axes(rect)
        cb = fig.colorbar(
            meshes[mesh_index],
            cax=cax,
            format=formatter,
        )
        cb.set_label(
            label,
            fontsize=10,
            fontweight="bold",
            rotation=270,
            labelpad=16,
        )
        cb.ax.tick_params(labelsize=9)

        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")

    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig

