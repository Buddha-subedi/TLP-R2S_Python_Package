"""Aggregate multiple saved retrieved S4 swaths to a 721 x 1440 grid."""
from pathlib import Path
from tlpr2s import aggregate_swaths, plot_annual
paths=sorted(Path('retrieved_orbits').glob('*.npz'))
if not paths: raise SystemExit('Place retrieved swath NPZ files in retrieved_orbits/')
stats=aggregate_swaths(paths,resolution=0.25)
stats.save('annual_stats.npz')
plot_annual(stats,field='mean_precip',save='annual_mean.png')
plot_annual(stats,field='observation_count',save='annual_count.png')
