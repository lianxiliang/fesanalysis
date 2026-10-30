"""Reproducible local timings; no claimed benchmark against ECToolkits.

Usage: python benchmarks/benchmark.py /path/to/output.xyz --output report.json
Full trajectory comparison tests independent MDAnalysis geometry vs the optimized
orthorhombic kernel. Both implement this project's documented classification.
"""
import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
from bivo4_fes import *


def elapsed(function):
    start=time.perf_counter();value=function()
    return value,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--output',default='benchmark.json')
    a=p.parse_args()
    selection=PTSelection.from_symbols(xyz_symbols(a.trajectory),151,[131,137,139,140,130])
    reader=lambda:iter_xyz(a.trajectory,cell=[10.39,10.18,36.17])
    frames,read_s=elapsed(lambda:list(reader()))
    # Explicit warm-up: first-use JIT is not hidden in the steady-state timing.
    _,warmup=elapsed(lambda:frame_paths(frames[0].positions,frames[0].box,selection))
    fast,fast_s=elapsed(lambda:analyze(frames,selection,cv_acceptor=130,progress=False,keep_paths=True))
    slow,slow_s=elapsed(lambda:analyze(frames,selection,cv_acceptor=130,progress=False,keep_paths=True,backend='reference'))
    checks={}
    for name in ['direct','indirect']:
        x,y=getattr(fast,name),getattr(slow,name)
        checks[name]={'valid_frames':int(fast.valid(name).sum()),
                      'same_missing_mask':bool(np.array_equal(np.isnan(x),np.isnan(y))),
                      'max_absolute_difference':float(np.nanmax(np.abs(x-y)))}
        np.testing.assert_allclose(x,y,atol=2e-5,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(fast.n_paths,slow.n_paths)
    np.testing.assert_allclose(fast.paths,slow.paths,atol=2e-5,rtol=0,equal_nan=True)
    _,stream_s=elapsed(lambda:analyze(reader(),selection,cv_acceptor=130,progress=False))
    rng=np.random.default_rng(16)
    q=rng.normal(0,.4,(4000,2));lw=rng.normal(0,3,len(q))
    kw=dict(bounds=[[-1,1],[-1,1]],bins=101,bandwidth=.08)
    compute_fes(q[:3],lw[:3],**kw)  # KDE compilation
    exact,exact_s=elapsed(lambda:compute_fes(q,lw,method='exact',**kw))
    local,local_s=elapsed(lambda:compute_fes(q,lw,method='local',**kw))
    binned,binned_s=elapsed(lambda:compute_fes(q,lw,method='binned',**kw))
    np.testing.assert_allclose(local.free_energy,exact.free_energy,atol=1e-10,rtol=0)
    relevant=exact.free_energy<.3
    report={'hardware':platform.platform(),'python':platform.python_version(),
            'versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','numba','MDAnalysis']},
            'geometry':{'frames':len(frames),'atoms':len(frames[0].positions),'read_seconds':read_s,
                        'jit_warmup_seconds':warmup,'optimized_seconds':fast_s,
                        'reference_seconds':slow_s,'speedup_over_own_reference':slow_s/fast_s,
                        'streaming_end_to_end_seconds':stream_s,'checks':checks,
                        'same_all_candidate_paths':True,'both_channels_frames':int(np.sum(fast.valid('direct')&fast.valid('indirect')))},
            'fes':{'samples':len(q),'grid':[101,101],'exact_seconds':exact_s,'local_seconds':local_s,
                   'binned_seconds':binned_s,'local_max_error_eV':float(np.max(abs(local.free_energy-exact.free_energy))),
                   'binned_max_error_below_0p3eV':float(np.max(abs(binned.free_energy[relevant]-exact.free_energy[relevant]))),
                   'note':'Binned is approximate. Local uses 12-sigma truncation.'},
            'limitations':['Not a comparison against the unavailable ECToolkits core.',
                           'Not a production-trajectory benchmark on CSCS.',
                           'No matching COLVAR for the supplied XYZ; no physical PT FES validated.']}
    Path(a.output).write_text(json.dumps(report,indent=2))
    fast.save(str(Path(a.output).with_name('validated_geometry.npz')))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
