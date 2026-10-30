"""OPES frame weights, walls, populations, and explicitly labelled diagnostics."""
import numpy as np
from scipy.special import logsumexp

KB = {"eV": 8.617333262145e-5, "kJ/mol": 0.00831446261815324}


def kbt(temperature=350., unit="eV"):
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return KB[unit]*temperature


def opes_logweights(bias, *, temperature=350., unit="kJ/mol", delta_energy=None):
    """log(w)=bias/kBT, optionally -(U_target-U_sampled)/kBT.

    Choose which restraints to remove BEFORE summing the bias. No opes.rct term.
    Valid for a quasi-static/equilibrated OPES segment, not a remedy for unsampled
    hidden states. Do not pool independently shifted biases from different runs.
    delta_energy must use the SAME unit and the actual sampled Hamiltonian.
    """
    bias = np.asarray(bias, dtype=float)
    if not np.all(np.isfinite(bias)):
        raise ValueError("Nonfinite bias values.")
    return (bias if delta_energy is None else bias-np.asarray(delta_energy))/kbt(temperature,unit)


def normalized_weights(logweights):
    lw = np.asarray(logweights,dtype=float)
    if not lw.size or np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any():
        raise ValueError("Weights must contain at least one finite log-weight and no NaN/+inf.")
    return np.exp(lw-logsumexp(lw))


def weight_diagnostics(logweights):
    """Kish ESS measures weight concentration, NOT time-independent sample count."""
    w = normalized_weights(logweights)
    return {"frames":len(w), "kish_ess":float(1./np.dot(w,w)),
            "largest_weight_fraction":float(w.max())}


def upper_wall(x, at, kappa=300.):
    """PLUMED EXP=2, EPS=1, OFFSET=0: no factor 1/2."""
    return kappa*np.maximum(np.asarray(x)-at,0.)**2


def lower_wall(x, at, kappa=300.):
    return kappa*np.maximum(at-np.asarray(x),0.)**2


def original_walls(colvar, *, include_bi_wall=True):
    """Reconstruct ONLY the exact five walls supplied in the conversation.

    If you changed AT/KAPPA/EXP/EPS, use your own expression or printed wall_bias.
    Result is kJ/mol for the supplied input UNITS LENGTH=A.
    """
    c = colvar
    total = (upper_wall(c['cn'],2.2)+lower_wall(c['cn'],0.8)
             +upper_wall(c['d1'],4.0)+lower_wall(c['d1'],1.6))
    if include_bi_wall:
        total += upper_wall(c['d2'],3.5)
    return total


def populations(labels, logweights, *, mask=None):
    """Weighted exclusive categories, optionally conditional on a product mask.

    Keep an 'unassigned/water/shared' category; do not silently force argmax over
    surface O to label all frames. Multiple simultaneous protonations instead
    require separate indicator averages, not these normalized categories.
    """
    labels, lw = np.asarray(labels), np.asarray(logweights)
    if mask is not None:
        labels, lw = labels[mask], lw[mask]
    w = normalized_weights(lw)
    values, inv = np.unique(labels,return_inverse=True)
    return dict(zip(values.tolist(),np.bincount(inv,weights=w).tolist()))


def weighted_mean(observable, logweights, *, mask=None):
    x,lw = np.asarray(observable),np.asarray(logweights)
    if mask is not None:
        x,lw = x[mask],lw[mask]
    return np.tensordot(normalized_weights(lw),x,axes=(0,0))


def basin_delta_f(reactant, product, logweights, *, temperature=350., unit="eV"):
    """-kBT log(P_product/P_reactant), from frames rather than grid minima."""
    lw = np.asarray(logweights)
    r,p = np.asarray(reactant,dtype=bool),np.asarray(product,dtype=bool)
    if np.any(r&p) or not np.any(r) or not np.any(p):
        raise ValueError("Basins must be nonempty and disjoint.")
    return -kbt(temperature,unit)*(logsumexp(lw[p])-logsumexp(lw[r]))


def block_populations(labels, logweights, *, mask=None, nblocks=4):
    """Contiguous blocks for a convergence check, not an automatic error estimate."""
    labels,lw = np.asarray(labels),np.asarray(logweights)
    mask = np.ones(len(lw),bool) if mask is None else np.asarray(mask,bool)
    out=[]
    for ids in np.array_split(np.arange(len(lw)),nblocks):
        selected=ids[mask[ids]]
        out.append(populations(labels[selected],lw[selected]) if len(selected) else {})
    return out
