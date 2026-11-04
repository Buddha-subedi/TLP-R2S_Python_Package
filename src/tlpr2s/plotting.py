"""Combined rain/snow map, matching the user's original palettes and two colorbars.

PlateCarree replaces Mercator because Mercator cannot represent +/-90 degrees.
Natural Earth coastlines/borders require Cartopy's dataset cache. A missing cache
produces a clear warning and a map without those optional features.
"""
import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib
# Respect the application's configured Matplotlib backend.
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

POWER = 0.4
SNOW_TICKS = [0.01, 0.15, 0.5, 1.0]
RAIN_TICKS = [0.1, 1.0, 3.0, 6.0]
SNOW_CMAP = LinearSegmentedColormap.from_list('tlpr2s_snow',
    ['#E0E0E0', '#A9EAFE', '#00BFFF', '#0000FF', '#9400D3'])
RAIN_CMAP = LinearSegmentedColormap.from_list('tlpr2s_rain',
    ['gainsboro', '#008000', '#E3E025', '#FFB700', '#FF167F', '#8B0000'])


def save_combined_map(path, longitude, latitude, rain, snow, title, dpi=300):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lon = np.asarray(longitude, dtype=np.float64)
    lat = np.asarray(latitude, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64).copy()
    snow = np.asarray(snow, dtype=np.float64).copy()
    if not (lon.shape == lat.shape == rain.shape == snow.shape):
        raise ValueError(f'Plot shape mismatch: {lon.shape}, {lat.shape}, {rain.shape}, {snow.shape}')
    rain[~np.isfinite(rain) | (rain < 0.05)] = 0
    snow[~np.isfinite(snow) | (snow < 0)] = 0
    dateline = (~np.isfinite(lon)) | (~np.isfinite(lat)) | (lon > 179.5) | (lon < -179.5)
    snow_layer = np.ma.masked_where(dateline | (rain > snow) | (snow <= 0), snow)
    rain_layer = np.ma.masked_where(dateline | (snow > rain) | (rain <= 0), rain)
    # Plot only nonzero dominant phase; retain original 0.4 power scaling.
    with mpl.rc_context({'text.usetex': False, 'font.family': 'serif',
                         'font.size': 12, 'axes.linewidth': 1.5,
                         'axes.edgecolor': 'black'}):
        fig = plt.figure(figsize=(14, 8))
        try:
            # Mercator is undefined at the poles. PlateCarree shows -90..90.
            ax = fig.add_axes([0.07, 0.12, 0.79, 0.78], projection=ccrs.PlateCarree(central_longitude=0))
            snow_plot = ax.pcolormesh(lon, lat, np.ma.power(snow_layer, POWER),
                vmin=0, vmax=1.0**POWER, cmap=SNOW_CMAP,
                transform=ccrs.PlateCarree(), shading='auto', alpha=0.9,
                rasterized=True)
            rain_plot = ax.pcolormesh(lon, lat, np.ma.power(rain_layer, POWER),
                vmin=0, vmax=6.0**POWER, cmap=RAIN_CMAP,
                transform=ccrs.PlateCarree(), shading='auto', alpha=0.9,
                rasterized=True)
            ax.set_global()
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            # Offline compute nodes may lack Natural Earth cache.
            try:
                import cartopy.io.shapereader as shpreader
                # Force dataset access here, so missing data does not fail at savefig.
                coast = shpreader.natural_earth(resolution='110m', category='physical', name='coastline')
                border = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_boundary_lines_land')
                import cartopy.feature as cf
                ax.add_geometries(shpreader.Reader(coast).geometries(), ccrs.PlateCarree(),
                                  edgecolor='#404040', facecolor='none', linewidth=0.2)
                ax.add_geometries(shpreader.Reader(border).geometries(), ccrs.PlateCarree(),
                                  edgecolor='#606060', facecolor='none', linewidth=0.2, linestyle=':')
            except Exception as exc:
                warnings.warn(f'Natural Earth coastline/border unavailable: {exc}. Map saved without borders.')
            gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-180,181,60),
                ylocs=np.arange(-90,91,20), color='gray', alpha=0.3,
                linestyle='--', linewidth=0.8)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 12, 'weight': 'bold', 'color': '#404040'}
            gl.ylabel_style = {'size': 12, 'weight': 'bold', 'color': '#404040'}
            ax.set_title(title, fontsize=18, fontweight='bold', pad=10, color='#202020')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor('#404040')
            cax_snow = fig.add_axes([0.86, 0.55, 0.01, 0.3])
            cax_rain = fig.add_axes([0.86, 0.20, 0.01, 0.3])
            cb_snow = fig.colorbar(snow_plot, cax=cax_snow)
            cb_snow.set_label('Snow Rate (mm hr⁻¹)', fontsize=12, fontweight='bold',
                              color='#202020', labelpad=10)
            cb_snow.set_ticks([v**POWER for v in SNOW_TICKS])
            cb_snow.ax.set_yticklabels([f'{v:.2f}' for v in SNOW_TICKS], fontweight='bold')
            cb_rain = fig.colorbar(rain_plot, cax=cax_rain)
            cb_rain.set_label('Rain Rate (mm hr⁻¹)', fontsize=12, fontweight='bold',
                              color='#202020', labelpad=10)
            cb_rain.set_ticks([v**POWER for v in RAIN_TICKS])
            cb_rain.ax.set_yticklabels([f'{v:.1f}' for v in RAIN_TICKS], fontweight='bold')
            for cb in (cb_snow, cb_rain):
                cb.ax.tick_params(labelsize=10, width=1.5, length=5, colors='#404040')
                for spine in cb.ax.spines.values():
                    spine.set_linewidth(1.5)
                    spine.set_edgecolor('#404040')
            tmp = path.with_name('.' + path.stem + '.partial.png')
            try:
                fig.savefig(tmp, dpi=dpi, bbox_inches='tight', facecolor='white')
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            plt.close(fig)


def plot_orbit(latitude, longitude, rain, snow, *, title='TLP-R2S Precipitation Rate', save=None, dpi=300):
    """Save the original combined rain/snow swath visualization with dual colorbars.

    All four arrays must share the same 2D shape. Returns the output Path.
    """
    if save is None:
        raise ValueError('save= is required for the reference map writer')
    save_combined_map(save, longitude, latitude, rain, snow, title, dpi=dpi)
    return Path(save)

def _plot_map(ax, lon, lat, data, cmap, vmin=None, vmax=None):
    import cartopy.crs as ccrs
    ax.coastlines(linewidth=0.5)
    return ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

def _axes(figsize=(12, 5)):
    import cartopy.crs as ccrs
    return plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})

def plot_annual(statistics,field='mean_precip',*,save=None,show=False):
    """Plot annual sampled-swath statistics from AnnualStatistics or saved NPZ."""
    from pathlib import Path
    from .annual import AnnualStatistics
    if isinstance(statistics,(str,Path)):
        statistics=AnnualStatistics.load(statistics)
    import matplotlib.pyplot as plt
    allowed={'mean_precip','mean_rain','mean_snow','observation_count','swath_coverage_count'}
    if field not in allowed:raise ValueError(f'field must be one of {sorted(allowed)}')
    v=getattr(statistics,field)
    fig,ax=_axes(figsize=(13,5.5))
    cmap='viridis' if 'count' in field else 'turbo'
    mesh=_plot_map(ax,statistics.longitude,statistics.latitude,v,cmap)
    ax.set_title(field.replace('_',' ').title()+' (sampled swaths)')
    unit='count' if 'count' in field else 'mm h$^{-1}$'
    fig.colorbar(mesh,ax=ax,label=unit,shrink=.8)
    fig.tight_layout()
    if save:fig.savefig(save,dpi=200,bbox_inches='tight')
    if show:plt.show()
    return fig,ax
