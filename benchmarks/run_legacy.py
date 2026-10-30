"""Optional upstream run for comparison; requires the user's ECToolkits install.

This adapter follows the exact call signatures in the supplied check_*PT.py files.
It is intentionally NOT an installed dependency and was not executable here.
Run on the SAME full XYZ/order as the new cache before compare_legacy.py.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import MDAnalysis as mda
from bivo4_fes import PTSelection, xyz_symbols, as_box


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory')
    parser.add_argument('--cell',nargs=3,type=float,required=True)
    parser.add_argument('--donor',type=int,default=151)
    parser.add_argument('--acceptors',nargs='+',type=int,required=True)
    parser.add_argument('--slab-atoms',type=int,default=144)
    parser.add_argument('--output',default='legacy')
    args=parser.parse_args()
    from ectoolkits.analysis.proton_transfer_cv import ProtonTransferCV
    symbols=xyz_symbols(args.trajectory)
    selection=PTSelection.from_symbols(symbols,args.donor,args.acceptors,slab_atoms=args.slab_atoms)
    universe=mda.Universe(args.trajectory)
    universe.add_TopologyAttr('types',symbols)
    box=as_box(args.cell)
    def set_box(ts):
        ts.dimensions=box
        return ts
    universe.trajectory.add_transformations(set_box)
    report={'core_sha256':hashlib.sha256(inspect.getsource(ProtonTransferCV).encode()).hexdigest()}
    for channel,bridges in [('direct',0),('indirect',1)]:
        options=dict(atomgroup=universe.atoms,idxs_type1_o=[selection.donor],
                     idxs_type2_o=selection.acceptors.tolist(),extra_detail=True)
        if bridges:
            options.update(num_bridge=1,idxs_water_o=selection.waters.tolist())
        start=time.perf_counter()
        analysis=ProtonTransferCV(**options)
        analysis.run()
        result=np.asarray(analysis.results)
        filename=f'{args.output}_{channel}.npy'
        np.save(filename,result)
        report[channel]={'seconds':time.perf_counter()-start,'shape':list(result.shape),'file':filename}
    Path(f'{args.output}_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
