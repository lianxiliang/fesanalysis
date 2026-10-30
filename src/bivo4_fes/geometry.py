"""Local hydrogen-bond paths: direct and exactly one water bridge.

The CV equations and 3.5 A / 30 degree geometry follow the supplied paper.
Hydrogen ownership and competing-path selection are explicit conventions here;
exact parity with the unavailable ECToolkits implementation is not asserted.
"""
from dataclasses import dataclass, replace
import json
import numpy as np
from numba import njit
from MDAnalysis.lib.distances import distance_array, minimize_vectors, calc_bonds
from tqdm import tqdm

# Columns in a candidate row; every atom index is zero-based.
PATH_COLUMNS = ("mechanism", "delta", "distance", "acceptor", "bridge", "h1", "h2",
                "delta1", "delta2", "d1", "d2", "angle_score")


@njit(cache=True)
def _vec(a, b, box):
    v = b-a
    return v - box*np.rint(v/box)


@njit(cache=True)
def _owners(positions, box, oxygens, hydrogens):
    """Assign each H to its nearest oxygen, without a hard OH bond cutoff."""
    distances = np.empty((len(oxygens), len(hydrogens)))
    owner = np.empty(len(hydrogens), dtype=np.int64)
    for j, h in enumerate(hydrogens):
        best = np.inf
        chosen = -1
        for i, o in enumerate(oxygens):
            dx = positions[h,0]-positions[o,0]
            dy = positions[h,1]-positions[o,1]
            dz = positions[h,2]-positions[o,2]
            dx -= box[0]*np.rint(dx/box[0])
            dy -= box[1]*np.rint(dy/box[1])
            dz -= box[2]*np.rint(dz/box[2])
            r = np.sqrt(dx*dx+dy*dy+dz*dz)
            distances[i,j] = r
            if r < best:
                best, chosen = r, o
        owner[j] = chosen
    return owner, distances


@njit(cache=True)
def _edges(positions, box, hydrogens, owner, left, right, cutoff, cos_angle):
    """All valid H bonds for an ordered oxygen pair, in EITHER direction.

    Ownership only chooses the angle vertex; delta always means r(left,H)-r(right,H).
    Thus a product with H now on the acceptor is not discarded.
    """
    result = []
    oo = _vec(positions[left], positions[right], box)
    d = np.sqrt(np.dot(oo, oo))
    if d < 1e-12 or d >= cutoff:
        return result
    for j, h in enumerate(hydrogens):
        if owner[j] != left and owner[j] != right:
            continue
        lh = _vec(positions[left], positions[h], box)
        rh = _vec(positions[right], positions[h], box)
        l = np.sqrt(np.dot(lh, lh)); r = np.sqrt(np.dot(rh, rh))
        if min(l, r) < 1e-12:
            continue
        cosine = np.dot(oo, lh)/(d*l) if owner[j] == left else -np.dot(oo, rh)/(d*r)
        if cosine > cos_angle:
            result.append((h, l-r, d, 1.-cosine))
    return result


@njit(cache=True)
def _orthogonal_paths(positions, box, oxygens, hydrogens, donor, acceptors,
                      waters, cutoff, cos_angle):
    owner, distances = _owners(positions, box, oxygens, hydrogens)
    rows = []
    for a in acceptors:
        for h, delta, d, quality in _edges(positions, box, hydrogens, owner,
                                          donor, a, cutoff, cos_angle):
            rows.append((0.,delta,d,float(a),-1.,float(h),-1.,delta,np.nan,d,np.nan,quality))
    for w in waters:
        first = _edges(positions, box, hydrogens, owner, donor, w, cutoff, cos_angle)
        if len(first) == 0:
            continue
        for a in acceptors:
            second = _edges(positions, box, hydrogens, owner, w, a, cutoff, cos_angle)
            for h1, q1, d1, e1 in first:
                for h2, q2, d2, e2 in second:
                    if h1 != h2:
                        rows.append((1.,(q1+q2)/2.,(d1+d2)/2.,float(a),float(w),
                                     float(h1),float(h2),q1,q2,d1,d2,e1+e2))
    return rows, distances


def _general_paths(positions, box, s, cutoff, cos_angle):
    """Triclinic fallback with MDAnalysis' tested minimum-image routines."""
    distances = distance_array(positions[s.oxygens], positions[s.hydrogens], box)
    owners = s.oxygens[np.argmin(distances, axis=0)]
    def edge(left, right):
        oo = minimize_vectors(np.array([positions[right]-positions[left]]), box)[0]
        d = np.linalg.norm(oo)
        if not 1e-12 < d < cutoff:
            return []
        h = s.hydrogens[(owners == left) | (owners == right)]
        if not len(h):
            return []
        lh = minimize_vectors(positions[h]-positions[left], box)
        rh = minimize_vectors(positions[h]-positions[right], box)
        dl, dr = np.linalg.norm(lh,axis=1), np.linalg.norm(rh,axis=1)
        owner_left = owners[(owners == left) | (owners == right)] == left
        cosine = np.where(owner_left, (lh@oo)/(d*np.maximum(dl,1e-30)),
                          -(rh@oo)/(d*np.maximum(dr,1e-30)))
        use = (cosine > cos_angle) & (np.minimum(dl,dr) > 1e-12)
        return [(int(hh),float(q),d,float(1-c)) for hh,q,c in zip(h[use],(dl-dr)[use],cosine[use])]
    rows = []
    for a in s.acceptors:
        for h,q,d,e in edge(s.donor,a):
            rows.append((0,q,d,a,-1,h,-1,q,np.nan,d,np.nan,e))
    for w in s.waters:
        first = edge(s.donor,w)
        if not first:
            continue
        for a in s.acceptors:
            for h1,q1,d1,e1 in first:
                for h2,q2,d2,e2 in edge(w,a):
                    if h1 != h2:
                        rows.append((1,(q1+q2)/2,(d1+d2)/2,a,w,h1,h2,q1,q2,d1,d2,e1+e2))
    return rows, distances


def frame_paths(positions, box, selection, *, oo_cutoff=3.5, angle_cutoff=30., backend="auto"):
    """Return (all candidate paths, O-H distances), independently for both channels.

    The angle is at the nearest/covalently associated O: angle(O_other,O_owner,H).
    This is NOT the common O-H-O angle with a 150-degree threshold.
    Nearest-oxygen assignment also allows proton-shared configurations.
    backend='reference' uses MDAnalysis even for orthorhombic cells.
    """
    if backend not in ("auto", "reference"):
        raise ValueError("backend must be auto or reference.")
    s = selection
    p = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=float)
    cosine = np.cos(np.deg2rad(angle_cutoff))
    if backend == "auto" and np.allclose(box[3:],90.):
        rows, distances = _orthogonal_paths(p,box[:3],s.oxygens,s.hydrogens,s.donor,
                                           s.acceptors,s.waters,oo_cutoff,cosine)
    else:
        rows, distances = _general_paths(p,box,s,oo_cutoff,cosine)
    return np.asarray(rows,dtype=float).reshape(-1,len(PATH_COLUMNS)), distances


def choose_path(candidates):
    """Shortest total O-O connection, then most linear angles, then atom IDs.

    This is a reproducible geometric convention, not a kinetic commitment test.
    All paths can be retained to investigate sensitivity to this convention.
    """
    if not len(candidates):
        return np.full(len(PATH_COLUMNS),np.nan)
    order = np.lexsort((candidates[:,6],candidates[:,5],candidates[:,4],candidates[:,3],
                        candidates[:,11],candidates[:,2]))
    return candidates[order[0]]


def _select(candidates, selector):
    """Custom selectors only receive nonempty candidate arrays."""
    if not len(candidates):
        return np.full(len(PATH_COLUMNS), np.nan)
    row = np.asarray(selector(candidates), dtype=float)
    if row.shape != (len(PATH_COLUMNS),):
        raise ValueError("A path selector must return one row in PATH_COLUMNS order.")
    return row


def _selector_name(selector):
    if selector is choose_path:
        return "shortest O-O path; angle; IDs"
    return getattr(selector, "__qualname__", type(selector).__name__)


@dataclass
class PTResult:
    frame: np.ndarray
    step: np.ndarray
    time: np.ndarray
    direct: np.ndarray
    indirect: np.ndarray
    n_paths: np.ndarray
    donor_cn: np.ndarray
    surface_cn: np.ndarray
    d_cv: np.ndarray
    acceptors: np.ndarray
    metadata: dict
    paths: np.ndarray | None = None   # row ID in this result + PATH_COLUMNS

    def channel(self, name):
        """Notebook-friendly named views. Absent paths are NaN, never zero."""
        a = {"direct":self.direct, "indirect":self.indirect}[name]
        return {key:a[:,i] for i,key in enumerate(PATH_COLUMNS)}

    def valid(self, name):
        return np.isfinite(self.channel(name)["delta"])

    def reselect(self, path_selector=choose_path):
        """Select again from cached candidates; no trajectory I/O or new geometry.

        Returns a new result without changing this object's selections. The callback
        takes a nonempty (n_paths, 12) array and returns one candidate row. See
        PATH_COLUMNS for named column positions. Callables are not serialized:
        keep the function definition in your notebook for reproducibility.
        """
        if self.paths is None:
            raise ValueError("Candidate paths were not saved; use keep_paths=True.")
        direct = np.full_like(self.direct, np.nan)
        indirect = np.full_like(self.indirect, np.nan)
        # Cached candidate rows are grouped by frame by analyze/concatenate.
        if len(self.paths):
            ids = self.paths[:, 0].astype(int)
            cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
            for start, stop in zip(cuts[:-1], cuts[1:]):
                candidates = self.paths[start:stop, 1:]
                for code, target in enumerate((direct, indirect)):
                    target[ids[start]] = _select(candidates[candidates[:, 0] == code], path_selector)
        metadata = dict(self.metadata, selector=_selector_name(path_selector))
        return replace(self, direct=direct, indirect=indirect, metadata=metadata)

    def save(self, path):
        data = {k:v for k,v in vars(self).items() if k not in ("metadata","paths")}
        data["metadata"] = json.dumps(self.metadata)
        if self.paths is not None:
            data["paths"] = self.paths
        np.savez_compressed(path, **data)

    @classmethod
    def concatenate(cls, results):
        """Join ordered chunks of ONE run. Does not pool independent OPES biases."""
        results = list(results)
        first = results[0]
        if any(r.metadata != first.metadata or not np.array_equal(r.acceptors, first.acceptors)
               for r in results):
            raise ValueError("Chunks must use the same selections and analysis settings.")
        data = {key:np.concatenate([getattr(r,key) for r in results])
                for key in ("frame", "step", "time", "direct", "indirect", "n_paths",
                            "donor_cn", "surface_cn", "d_cv")}
        paths = None
        if all(r.paths is not None for r in results):
            chunks = []
            offset = 0
            for r in results:
                a = r.paths.copy()
                a[:,0] += offset
                chunks.append(a)
                offset += len(r.frame)
            paths = np.concatenate(chunks)
        return cls(**data,acceptors=first.acceptors,metadata=first.metadata,paths=paths)

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as f:
            data = {k:f[k] for k in f.files}
        data["metadata"] = json.loads(str(data["metadata"]))
        return cls(**data)


def analyze(frames, selection, *, cv_acceptor=None, oo_cutoff=3.5,
            angle_cutoff=30., keep_paths=False, progress=True, backend="auto",
            path_selector=choose_path):
    """One streaming pass for both PT channels, donor/surface CN, and original d1.

    cv_acceptor is ZERO-based (e.g.130 for PLUMED O131); unrelated to path selection.
    CN uses r0=1.3 A, NN=12, MM=24, D0=0, no finite-distance stretching.
    Only compact observables are retained, never all atomic coordinates.
    path_selector receives nonempty candidate rows and returns one row.
    """
    s = selection
    o_lookup = {o:i for i,o in enumerate(s.oxygens)}
    cn_rows = np.array([o_lookup[s.donor]]+[o_lookup[o] for o in s.acceptors])
    fidx=[]; steps=[]; times=[]; direct=[]; indirect=[]; counts=[]; cn=[]; ds=[]; all_paths=[]
    for row, frame in enumerate(tqdm(frames,disable=not progress,desc="PT geometry",unit="frame")):
        p, box = frame.positions, frame.box
        paths, oh = frame_paths(p,box,s,oo_cutoff=oo_cutoff,angle_cutoff=angle_cutoff,backend=backend)
        d = paths[paths[:,0]==0]; q = paths[paths[:,0]==1]
        direct.append(_select(d,path_selector)); indirect.append(_select(q,path_selector));counts.append([len(d),len(q)])
        cn.append((1./(1.+(oh[cn_rows]/1.3)**12)).sum(axis=1))
        if cv_acceptor is not None:
            ds.append(calc_bonds(p[s.donor],p[cv_acceptor],box=box))
        else:
            ds.append(np.nan)
        fidx.append(frame.index);steps.append(frame.step);times.append(frame.time)
        if keep_paths and len(paths):
            all_paths.append(np.column_stack((np.full(len(paths),row),paths)))
    if not fidx:
        raise ValueError("No trajectory frames selected.")
    cn = np.asarray(cn)
    metadata = dict(oo_cutoff=oo_cutoff,angle_cutoff=angle_cutoff,index_base=0,
                    donor=int(s.donor),cv_acceptor=None if cv_acceptor is None else int(cv_acceptor),
                    selector=_selector_name(path_selector),version="0.1.1",
                    ownership="nearest oxygen, all O",channels_overlap=True,
                    waters=s.waters.tolist(),hydrogens=s.hydrogens.tolist(),
                    oxygens=s.oxygens.tolist())
    return PTResult(np.asarray(fidx),np.asarray(steps),np.asarray(times),np.asarray(direct),
                    np.asarray(indirect),np.asarray(counts),cn[:,0],cn[:,1:],np.asarray(ds),
                    s.acceptors,metadata,
                    (np.concatenate(all_paths) if all_paths else np.empty((0,13))) if keep_paths else None)


def all_path_samples(result, channel, frame_logweights, *, split_per_frame=True):
    """Optional alternative to one selected path per frame.

    Split each frame's weight equally over its candidate paths by default.
    False gives a path-occurrence density, not a frame-probability density.
    Neither option makes overlapping direct/indirect channels disjoint.
    """
    if result.paths is None:
        raise ValueError("Rerun geometry with keep_paths=True.")
    code = {"direct":0,"indirect":1}[channel]
    rows = result.paths[result.paths[:,1]==code]
    ids = rows[:,0].astype(int)
    logw = np.asarray(frame_logweights)[ids].copy()
    if split_per_frame:
        logw -= np.log(result.n_paths[ids,code])
    return rows[:,[2,3]],logw,ids


def analyze_chunks(frames,selection,*,chunk_size=10000,**kwargs):
    """Bound observable memory too: yield independently saveable PTResult chunks.

    Geometry is still a single trajectory pass. The temporary coordinate chunk is
    deliberately copied because MDAnalysis reuses a position buffer. Use a small
    chunk_size for very large systems. No process pool or hidden parallelism.
    """
    from itertools import islice
    from .io import Frame
    iterator=iter(frames)
    while True:
        chunk=[Frame(f.index,f.positions.copy(),f.box.copy(),f.step,f.time)
               for f in islice(iterator,chunk_size)]
        if not chunk:
            break
        yield analyze(chunk,selection,**kwargs)
