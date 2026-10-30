"""Frozen atom selections. Public constructors accept an explicit index base."""
from dataclasses import dataclass
import numpy as np


def by_z(symbols, positions, element, ranges, *, stop=None):
    """Select once from a reference structure, not again as atoms diffuse.

    Positions must already be wrapped into the intended slab image.
    Returns zero-based indices. stop=144 excludes water-derived oxygen.
    """
    symbols, positions = np.asarray(symbols), np.asarray(positions)
    mask = np.zeros(len(symbols), dtype=bool)
    for lo, hi in ranges:
        mask |= (positions[:, 2] >= lo) & (positions[:, 2] <= hi)
    mask &= symbols == element
    if stop is not None:
        mask &= np.arange(len(symbols)) < stop
    return np.flatnonzero(mask)


def notebook_bi_water(symbols, positions):
    """The first-frame z windows from the supplied Bi-water-CN script."""
    bi = by_z(symbols, positions, "Bi", [(21.5, 22.5), (7.5, 8.5)])
    water = by_z(symbols, positions, "O", [(24, 36), (0, 6)])
    return bi, water


@dataclass(frozen=True)
class PTSelection:
    """All stored indices are zero-based. Water sites can change protonation."""
    donor: int
    acceptors: np.ndarray
    waters: np.ndarray
    hydrogens: np.ndarray
    oxygens: np.ndarray

    @classmethod
    def from_symbols(cls, symbols, donor, acceptors, *, waters=None,
                     hydrogens=None, slab_atoms=144, index_base=1):
        """Infer H and water-derived O, or supply either explicitly.

        Atom ordering: the first slab_atoms atoms are the solid. Chemical identity
        is checked once here. Donor/acceptors are excluded from bridge candidates.
        This does NOT decide which surface O belong to the paper's O_si class.
        """
        symbols = np.asarray(symbols)
        if index_base not in (0, 1):
            raise ValueError("index_base must be 0 (Python) or 1 (PLUMED).")
        def indices(values, element):
            out = np.asarray(values, dtype=int) - index_base
            if out.ndim != 1 or np.any(out < 0) or np.any(out >= len(symbols)):
                raise ValueError("Atom indices outside the structure.")
            if not np.all(symbols[out] == element):
                raise ValueError(f"Expected {element} atoms; check index_base.")
            return np.unique(out)
        donor = int(indices([donor], "O")[0])
        acceptors = indices(acceptors, "O")
        if not len(acceptors) or donor in acceptors:
            raise ValueError("Provide at least one acceptor distinct from the donor.")
        h = (np.flatnonzero(symbols == "H") if hydrogens is None
             else indices(hydrogens, "H"))
        if not len(h):
            raise ValueError("Select at least one hydrogen.")
        if waters is None and not 0 <= slab_atoms <= len(symbols):
            raise ValueError("slab_atoms must lie between 0 and the atom count.")
        w = (np.flatnonzero((symbols == "O") & (np.arange(len(symbols)) >= slab_atoms))
             if waters is None else indices(waters, "O"))
        w = np.setdiff1d(w, np.r_[donor, acceptors])
        return cls(donor, acceptors, w, h, np.flatnonzero(symbols == "O"))

    @classmethod
    def colleague(cls, symbols):
        """Exact lists in check_*PT.py, apart from excluding acceptors as bridges.

        Warning: zero-based O241 was a water-derived surface atom in the author's
        run. This chemical reassignment is NOT assumed to hold in a new run.
        """
        waters = [144,151,157,162,168,169,180,183,185,186,187,188,189,191,
                  216,217,222,223,224,226,227,228,229,232,233,234,235,236,
                  239,240,242,243,244,247,315,321,324,327,330,339,342,345,
                  348,351,354,130,145,156,163,174,175,181,182,184,190,218,
                  219,220,221,225,230,231,237,238,245,246,312,318,333,336,357]
        return cls.from_symbols(symbols, 150, [130,241,136,139,129],
                                waters=waters, index_base=0)
