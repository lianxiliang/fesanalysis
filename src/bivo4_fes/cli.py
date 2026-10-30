"""Thin command-line wrappers around the same notebook functions."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import (PTSelection, analyze, iter_xyz, xyz_symbols, read_colvar,
               opes_logweights, compute_fes, PTResult)


def main():
    parser=argparse.ArgumentParser(description='BiVO4 PT geometry and OPES FES')
    sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('geometry',help='Stream XYZ; cache direct/indirect CVs')
    g.add_argument('trajectory');g.add_argument('--config',required=True)
    g.add_argument('--output',default='pt.npz');g.add_argument('--keep-paths',action='store_true')
    f=sub.add_parser('fes',help='Reweight named columns in a COLVAR-style table')
    f.add_argument('colvar');f.add_argument('--cvs',nargs='+',required=True)
    f.add_argument('--bias',nargs='+',default=['opes.bias'])
    f.add_argument('--bounds',nargs='+',type=float,required=True)
    f.add_argument('--bandwidth',nargs='+',type=float,required=True)
    f.add_argument('--bins',nargs='+',type=int,default=[101])
    f.add_argument('--temperature',type=float,default=350.)
    f.add_argument('--bias-unit',choices=['eV','kJ/mol'],default='kJ/mol')
    f.add_argument('--output-unit',choices=['eV','kJ/mol'],default='eV')
    f.add_argument('--method',choices=['local','separable','binned','exact'],default='local')
    f.add_argument('--start-time',type=float,default=-np.inf)
    f.add_argument('--output',default='fes.npz')
    args=parser.parse_args()
    if args.command=='geometry':
        cfg=json.loads(Path(args.config).read_text())
        s=PTSelection.from_symbols(xyz_symbols(args.trajectory),**cfg['selection'])
        settings=cfg.get('analysis',{})
        r=analyze(iter_xyz(args.trajectory,**cfg['trajectory']),s,
                  keep_paths=args.keep_paths,**settings)
        r.save(args.output)
        print(json.dumps({'frames':len(r.frame),'direct':int(r.valid('direct').sum()),
                          'indirect':int(r.valid('indirect').sum()),'output':args.output},indent=2))
    else:
        c=read_colvar(args.colvar);mask=c['time']>=args.start_time
        q=np.column_stack([c[k][mask] for k in args.cvs])
        bias=sum(c[k][mask] for k in args.bias)
        lw=opes_logweights(bias,temperature=args.temperature,unit=args.bias_unit)
        surface=compute_fes(q,lw,bounds=np.asarray(args.bounds).reshape(-1,2),
                            bins=args.bins,bandwidth=args.bandwidth,temperature=args.temperature,
                            unit=args.output_unit,method=args.method)
        surface.save(args.output)
        surface.to_plumed(str(Path(args.output).with_suffix('.dat')),names=args.cvs)
        print(json.dumps(surface.metadata,indent=2))


if __name__=='__main__':
    main()
