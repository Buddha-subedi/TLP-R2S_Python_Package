"""Command line tools for saved array inputs, swath plots and annual gridding."""
import argparse
from pathlib import Path
import numpy as np
from .arrays import save_swath,load_swath
from .annual import aggregate_swaths,AnnualStatistics

def main(argv=None):
    p=argparse.ArgumentParser(prog='tlpr2s',description='SSMIS 2D precipitation retrieval and mapping')
    sub=p.add_subparsers(dest='cmd',required=True)
    r=sub.add_parser('retrieve',help='Retrieve from NPZ with 30 feature arrays and latitude/longitude')
    r.add_argument('input');r.add_argument('--models',default=None);r.add_argument('--cdf',default=None);r.add_argument('--output',required=True);r.add_argument('--threads',type=int,default=1)
    o=sub.add_parser('plot-orbit');o.add_argument('input');o.add_argument('--output',required=True);o.add_argument('--field',choices=['rain','snow','surface_precip','phase'],default='surface_precip');o.add_argument('--vmax',type=float,default=15)
    a=sub.add_parser('annual');a.add_argument('files',nargs='+');a.add_argument('--output',required=True);a.add_argument('--resolution',type=float,default=.25)
    q=sub.add_parser('plot-annual');q.add_argument('input');q.add_argument('--output',required=True);q.add_argument('--field',default='mean_precip')
    d=sub.add_parser('download-models',help='Download the five official Hugging Face artifacts into the local cache')
    d.add_argument('--revision',default=None,help='Pin a model repository commit SHA or tag')
    d.add_argument('--cache-dir',default=None)
    d.add_argument('--offline',action='store_true',help='Use cache only')
    args=p.parse_args(argv)
    if args.cmd=='download-models':
        from .model_download import download_models
        print(download_models(revision=args.revision,cache_dir=args.cache_dir,local_files_only=args.offline))
    elif args.cmd=='retrieve':
        from .retrieval import PrecipitationRetriever
        with np.load(args.input,allow_pickle=False) as z:
            if 'latitude' not in z or 'longitude' not in z:raise ValueError('Need latitude and longitude in NPZ')
            lat=z['latitude'];lon=z['longitude'];data={k:z[k] for k in z.files if k not in ('latitude','longitude')}
        result=PrecipitationRetriever(args.models,args.cdf,args.threads).predict(data)
        save_swath(args.output,lat,lon,result.phase,result.rain,result.snow)
    elif args.cmd=='plot-orbit':
        from .plotting import plot_orbit
        d=load_swath(args.input)
        field=d['rain']+d['snow'] if args.field=='surface_precip' else d[args.field]
        plot_orbit(d['latitude'],d['longitude'],field,title=args.field,save=args.output,vmax=args.vmax)
    elif args.cmd=='annual':
        stats=aggregate_swaths(args.files,args.resolution);stats.save(args.output)
        print(f'Saved {args.output}; valid pixels: {int(stats.observation_count.sum())}')
    elif args.cmd=='plot-annual':
        from .plotting import plot_annual
        plot_annual(AnnualStatistics.load(args.input),args.field,save=args.output)
if __name__=='__main__':main()
