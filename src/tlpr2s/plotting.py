
"""
TLP-R2S precipitation visualization.

Features
--------
1. Combined rainfall and snowfall in one geographic subplot.
2. Separate vertical rain and snow colorbars.
3. Original precipitation colormaps and power scaling.
4. Invalid latitude/longitude filtering.
5. Longitude normalization to [-180, 180].
6. Dominant precipitation phase visualization.
7. Cartopy coastlines and country boundaries.
8. Safe plotting across the international dateline.
9. Annual statistics visualization.
10. High-resolution PNG output.

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

    # Exclude points extremely close to the dateline
    # to avoid graphical discontinuities.
    dateline = (
        valid_geo
        & (np.abs(lon) >= 179.5)
    )

    valid_geo &= ~dateline

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
# 3. COMBINED RAIN/SNOW ORBITAL MAP
# ============================================================

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
            # Snowfall layer
            # ------------------------------------------------

            if np.any(snow_valid):

                snow_plot = ax.scatter(
                    lon[snow_valid],
                    lat[snow_valid],
                    c=np.power(
                        snow[snow_valid],
                        POWER,
                    ),
                    cmap=SNOW_CMAP,
                    vmin=0,
                    vmax=1.0 ** POWER,
                    s=0.6,
                    marker="s",
                    linewidths=0,
                    alpha=0.9,
                    rasterized=True,
                    transform=ccrs.PlateCarree(),
                    zorder=2,
                )

            else:
                warnings.warn(
                    "No valid snowfall pixels found."
                )

            # ------------------------------------------------
            # Rainfall layer
            # ------------------------------------------------

            if np.any(rain_valid):

                rain_plot = ax.scatter(
                    lon[rain_valid],
                    lat[rain_valid],
                    c=np.power(
                        rain[rain_valid],
                        POWER,
                    ),
                    cmap=RAIN_CMAP,
                    vmin=0,
                    vmax=6.0 ** POWER,
                    s=0.6,
                    marker="s",
                    linewidths=0,
                    alpha=0.9,
                    rasterized=True,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )

            else:
                warnings.warn(
                    "No valid rainfall pixels found."
                )

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

    save_combined_map(
        save,
        longitude,
        latitude,
        rain,
        snow,
        title,
        dpi=dpi,
    )

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
