"""Optional matplotlib/cartopy visualizations."""
import numpy as np

def _plot_map(ax,lon,lat,data,cmap,vmin=None,vmax=None):
    try:
        import cartopy.crs as ccrs
        ax.coastlines(linewidth=0.5)
        return ax.pcolormesh(lon,lat,data,transform=ccrs.PlateCarree(),shading='auto',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)
    except ImportError:
        return ax.pcolormesh(lon,lat,data,shading='auto',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True)

def _axes(figsize=(12,5)):
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        return plt.subplots(figsize=figsize,subplot_kw={'projection':ccrs.PlateCarree()})
    except ImportError:
        fig,ax=plt.subplots(figsize=figsize)
        ax.set_xlabel('Longitude (degrees)');ax.set_ylabel('Latitude (degrees)')
        return fig,ax

def plot_orbit(latitude,longitude,precipitation,*,title='SSMIS precipitation swath',vmax=15,save=None,show=False):
    """Plot one 2D swath. Supply its 2D lat/lon arrays, matching precipitation shape."""
    import matplotlib.pyplot as plt
    lat=np.asarray(latitude);lon=np.asarray(longitude);v=np.asarray(precipitation)
    if lat.ndim!=2 or lon.shape!=lat.shape or v.shape!=lat.shape:raise ValueError('All inputs must have matching 2D shapes')
    fig,ax=_axes()
    # Mask invalid and dateline-crossing cells to prevent long wraparound artifacts.
    masked=np.ma.masked_invalid(np.where(v>=0,v,np.nan))
    mesh=_plot_map(ax,lon,lat,masked,'turbo',0,vmax)
    ax.set_title(title)
    fig.colorbar(mesh,ax=ax,label='Precipitation rate (mm h$^{-1}$)',shrink=.8)
    fig.tight_layout()
    if save:fig.savefig(save,dpi=200,bbox_inches='tight')
    if show:plt.show()
    return fig,ax

def plot_annual(statistics,field='mean_precip',*,save=None,show=False):
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
