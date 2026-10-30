"""Synthetic DEMONSTRATION ONLY: plots without pretrained models or real data."""
from pathlib import Path
import numpy as np
from tlpr2s import save_swath,aggregate_swaths
from tlpr2s.plotting import plot_orbit,plot_annual

def main():
    out=Path('demo_output');out.mkdir(exist_ok=True)
    for i in range(4):
        lat0=np.linspace(-30+i*15,5+i*15,90)[:,None]
        lon0=np.linspace(-100+i*35,-50+i*35,120)[None,:]
        lat=np.broadcast_to(lat0,(90,120)).copy()
        lon=np.broadcast_to(lon0,(90,120)).copy()
        rain=np.maximum(0,10*np.exp(-((lat-(i*15-10))**2/100+(lon-(i*35-75))**2/120)))
        snow=np.maximum(0,2*np.exp(-((lat-(i*15))**2/200+(lon-(i*35-65))**2/150)))
        phase=np.where(snow>rain,2,np.where(rain>0.1,1,0)).astype(np.int8)
        save_swath(out/f'synthetic_orbit_{i}.npz',lat,lon,phase,rain,snow)
        if i==0:plot_orbit(lat,lon,rain+snow,save=out/'synthetic_orbit.png')
    stats=aggregate_swaths(sorted(out.glob('synthetic_orbit_*.npz')),resolution=1.0)
    stats.save(out/'synthetic_annual.npz')
    plot_annual(stats,save=out/'synthetic_annual.png')
    plot_annual(stats,field='observation_count',save=out/'synthetic_coverage.png')
    print(f'Synthetic demonstration written to {out.resolve()}')
if __name__=='__main__':main()
