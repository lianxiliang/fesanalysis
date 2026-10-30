"""Small notebook helpers for smooth coordination and surface-site labels."""
import numpy as np
from MDAnalysis.lib.distances import distance_array


def coordination(positions,centers,neighbors,box,*,r0=3.2,nn=6):
    """Untruncated rational switch NN=nn, MM=2*nn, D0=0, summed per center.

    Indices are zero-based. r0=3.2,nn=6 reproduces the chosen Bi-water definition;
    r0=1.3,nn=12 is the O-H definition used by the supplied OPES input.
    """
    distances=distance_array(np.asarray(positions)[centers],np.asarray(positions)[neighbors],box)
    return (1./(1.+(distances/r0)**nn)).sum(axis=1)


def protonated_site(surface_cn,acceptors,*,threshold=.5,ambiguity=.15):
    """Exclusive surface-O label or -1 (none/shared/multiple, inspect separately).

    Assumes initially unprotonated lattice O and a single added surface proton.
    Do not use this to label initially hydroxylated surfaces without a baseline.
    Uses zero-based atom indices; not the identity of a permanently labelled H.
    """
    cn=np.asarray(surface_cn);sites=np.asarray(acceptors)
    order=np.argsort(cn,axis=1)
    top=order[:,-1];best=cn[np.arange(len(cn)),top]
    second=cn[np.arange(len(cn)),order[:,-2]] if cn.shape[1]>1 else np.zeros(len(cn))
    label=sites[top].copy()
    label[(best<threshold)|(best-second<ambiguity)|(second>=threshold)]=-1
    return label
