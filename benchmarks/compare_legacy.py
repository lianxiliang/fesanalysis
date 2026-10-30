"""Compare cached CVs to DirectPTOsi/IndirectPTOsi-mlp*.npy from ECToolkits.

Run BEFORE claiming reproduction of the colleague's FES. The hidden upstream
path-selection/ownership conventions could differ from the documented new ones.
Usage: python benchmarks/compare_legacy.py pt.npz legacy.npy --channel direct
Legacy rows must correspond to the original trajectory frame order.
"""
import argparse
import json
import numpy as np
from bivo4_fes import PTResult


def main():
    p=argparse.ArgumentParser();p.add_argument('cache');p.add_argument('legacy')
    p.add_argument('--channel',choices=['direct','indirect'],required=True)
    p.add_argument('--output',default='legacy_comparison.json');a=p.parse_args()
    result=PTResult.load(a.cache);old=np.load(a.legacy,allow_pickle=False)
    old=old[result.frame][:,[1,2]]
    c=result.channel(a.channel);new=np.column_stack((c['delta'],c['distance']))
    valid_old=np.all(np.isfinite(old),axis=1);valid_new=np.all(np.isfinite(new),axis=1)
    both=valid_old&valid_new;diff=new[both]-old[both]
    report={'frames':len(new),'valid_legacy':int(valid_old.sum()),'valid_new':int(valid_new.sum()),
            'different_validity_frames':int(np.sum(valid_old!=valid_new)),
            'max_abs_difference_delta_distance':np.max(abs(diff),axis=0).tolist() if len(diff) else None,
            'rmse_delta_distance':np.sqrt(np.mean(diff**2,axis=0)).tolist() if len(diff) else None}
    with open(a.output,'w') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
