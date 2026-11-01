"""Streaming coordinate readers and strict COLVAR/time alignment."""
from dataclasses import dataclass
from pathlib import Path
import gzip
import re
import numpy as np
from MDAnalysis.lib.mdamath import triclinic_box


@dataclass
class Frame:
    index: int
    positions: np.ndarray
    box: np.ndarray                  # MDAnalysis [a,b,c,alpha,beta,gamma], angstrom
    step: int = -1
    time: float = np.nan             # ps, never guessed from a default reader dt


def as_box(cell):
    a = np.asarray(cell, dtype=float)
    if a.shape == (3,):
        a = np.r_[a, [90., 90., 90.]]
    elif a.shape == (3, 3):
        a = triclinic_box(*a)
    if a.shape != (6,) or np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError("cell must be three lengths, six cell parameters, or 3x3 vectors.")
    if np.any(a[3:] >= 180):
        raise ValueError("Cell angles must lie strictly between 0 and 180 degrees.")
    return a


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def xyz_symbols(path):
    with _open(path) as f:
        n = int(f.readline()); f.readline()
        return np.array([f.readline().split()[0] for _ in range(n)])


def iter_xyz(path, *, cell=None, timestep_ps=None, frame_dt_ps=None,
             time_origin_ps=0., start=0, stop=None, stride=1):
    """Sequential plain/extended XYZ reader; no full-file scan or trajectory list.

    Supports species,x,y,z as the first four fields. Reads 'Timestep: N' comments
    and extended-XYZ Lattice. A plain XYZ needs cell. For decimated trajectories,
    timestep_ps is the ORIGINAL MD timestep, not the retained-frame spacing.
    Without timing information time is NaN; geometry is still usable.
    Full periodicity in all three directions is assumed.
    """
    if start < 0 or stride < 1:
        raise ValueError("start>=0 and stride>=1 are required.")
    fixed_box = None if cell is None else as_box(cell)
    reference = None
    with _open(path) as f:
        i = 0
        while stop is None or i < stop:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line); comment = f.readline()
            lines = [f.readline() for _ in range(n)]
            if not lines[-1]:
                raise ValueError(f"Truncated XYZ at frame {i}.")
            if i < start or (i-start) % stride:
                i += 1; continue
            props = re.search(r'Properties=(\S+)', comment)
            if props and not props[1].startswith("species:S:1:pos:R:3"):
                raise ValueError("Use MDAnalysis or ASE for this XYZ Properties layout.")
            rows = [line.split() for line in lines]
            symbols = np.array([r[0] for r in rows])
            if reference is None:
                reference = symbols
            elif not np.array_equal(reference, symbols):
                raise ValueError(f"Atom order/species changed at frame {i}.")
            pos = np.asarray([r[1:4] for r in rows], dtype=float)
            box = fixed_box
            if box is None:
                lattice = re.search(r'Lattice="([^"]+)"', comment)
                if lattice:
                    box = as_box(np.fromstring(lattice[1], sep=" ").reshape(3, 3))
                else:
                    raise ValueError("Plain XYZ has no cell; supply cell explicitly.")
            match = re.search(r'(?:Timestep:\s*|step=)(\d+)', comment, re.I)
            step = int(match[1]) if match else -1
            time = np.nan
            if timestep_ps is not None and step >= 0:
                time = time_origin_ps + step*timestep_ps
            elif frame_dt_ps is not None:
                time = time_origin_ps + i*frame_dt_ps
            yield Frame(i, pos, np.array(box), step, time)
            i += 1


def iter_mdanalysis(universe, *, cell=None, timestep_ps=None, frame_dt_ps=None,
                    time_origin_ps=0., use_reader_time=False, start=0, stop=None, stride=1):
    """Adapt any MDAnalysis Universe, including chained dumps/XTC/DCD.

    Coordinates are borrowed until the next iteration. Atom order must be fixed.
    For LAMMPS use timestep_ps and ts.data['step']. Reader time is opt-in because
    some formats fabricate a 1-ps timestep. Reset/overlapping segments must be
    aligned separately, not silently concatenated with duplicate timestamps.
    """
    fixed = None if cell is None else as_box(cell)
    for ts in universe.trajectory[start:stop:stride]:
        box = fixed if fixed is not None else ts.dimensions
        if box is None:
            raise ValueError("Missing cell; supply cell or a trajectory with box data.")
        step = int(ts.data.get("step", -1))
        time = np.nan
        if timestep_ps is not None and step >= 0:
            time = time_origin_ps + step*timestep_ps
        elif frame_dt_ps is not None:
            time = time_origin_ps + ts.frame*frame_dt_ps
        elif use_reader_time:
            time = time_origin_ps + ts.time
        yield Frame(ts.frame, ts.positions, as_box(box), step, time)


def read_colvar(path, *, timestep_ps=None):
    """Read named PLUMED columns or i-PI EXTRAS_BIAS output."""
    fields = None
    steps = []
    ipi_format = False

    with _open(path) as f:
        for line in f:
            s = line.strip()

            # Standard PLUMED format
            if s.startswith("#! FIELDS"):
                names = s.split()[2:]
                if fields is not None and names != fields:
                    raise ValueError("COLVAR column schema changed within the file.")
                fields = names

            # i-PI EXTRAS_BIAS format
            elif s.startswith("#EXTRAS_BIAS("):
                ipi_format = True

                names = s.split("(", 1)[1].split(")", 1)[0].split(",")
                names = [name.strip() for name in names]

                if fields is not None and names != fields:
                    raise ValueError("COLVAR column schema changed within the file.")
                fields = names

                match = re.search(r"Step:\s*(\d+)", s)
                if match is None:
                    raise ValueError("Cannot read Step from i-PI COLVAR header.")
                steps.append(int(match.group(1)))

    if fields is None:
        raise ValueError(
            "Missing '#! FIELDS' or '#EXTRAS_BIAS(...)# Step:' header."
        )

    with _open(path) as f:
        data = np.loadtxt(f, comments="#", ndmin=2)

    if data.shape[1] != len(fields):
        raise ValueError("COLVAR column count does not match its header.")

    result = dict(zip(fields, data.T))

    if ipi_format:
        if len(steps) != len(data):
            raise ValueError(
                "Number of i-PI Step headers does not match number of data rows."
            )

        result["step"] = np.asarray(steps, dtype=int)

        if timestep_ps is not None:
            result["time"] = result["step"] * timestep_ps

    return result

def align_colvar(times, colvar, *, tolerance=1e-5):
    """Exact-nearest time matching, not bias interpolation or blind row slicing.

    Returns one COLVAR row index per trajectory frame. Times are in ps.
    Failure is deliberate: mismatched coordinates and weights give a wrong FES.
    """
    t, wanted = np.asarray(colvar["time"]), np.asarray(times)
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("COLVAR times must be strictly increasing; split restart segments.")
    if not np.all(np.isfinite(wanted)):
        raise ValueError("Trajectory times missing; set timestep_ps or frame_dt_ps.")
    if np.any(np.diff(wanted) <= 0):
        raise ValueError("Trajectory times must increase; handle restart segments separately.")
    right = np.clip(np.searchsorted(t, wanted), 0, len(t)-1)
    left = np.maximum(right-1, 0)
    idx = np.where(abs(t[left]-wanted) < abs(t[right]-wanted), left, right)
    if np.any(abs(t[idx]-wanted) > tolerance) or len(np.unique(idx)) != len(idx):
        raise ValueError("Frames do not match unique COLVAR rows within tolerance.")
    return idx


def check_cv_alignment(result, colvar, rows, *, d_name="d1", cn_name="cn",
                       distance_tolerance=2e-3, cn_tolerance=2e-3):
    """Independent identity check after time matching; uses recomputed d_cv/CN."""
    dd = np.max(abs(result.d_cv - colvar[d_name][rows]))
    dc = np.max(abs(result.donor_cn - colvar[cn_name][rows]))
    if not np.isfinite(dd + dc) or dd > distance_tolerance or dc > cn_tolerance:
        raise ValueError(f"CV mismatch: max distance error={dd:.4g} A, CN={dc:.4g}. "
                         "Check run identity, cell, atom IDs, timestep and CN definition.")
    return {"max_distance_error": float(dd), "max_cn_error": float(dc)}


def write_colvar(path,columns):
    """Export any named CV/bias table for reuse in Python or PLUMED READ.

    Include a time column when it exists. This does not add or recompute a bias.
    """
    names=list(columns)
    values=np.column_stack([np.asarray(columns[k]) for k in names])
    with open(path,'w') as f:
        f.write('#! FIELDS '+' '.join(names)+'\n')
        np.savetxt(f,values,fmt='%.12g')
