# bivo4-fes

**Notebook-first proton-transfer geometry, OPES reweighting, and minimum-energy paths.**

One trajectory is the unit of analysis. Repeat the same workflow for any number
of independent models; nothing assumes four. Geometry is calculated once and
cached, so changing a bandwidth, bias convention, state definition, or plot does
not require reading the heavy trajectory again.


## v0.1.2: optional fast exact FES backend

Use `compute_fes(..., method="separable")` for an exact diagonal-bandwidth 2D Gaussian KDE using chunked matrix products. It includes local weight-concentration ESS and retains a log-space fallback for very small kernel products. This does not change trajectory geometry or the default `method="local"`. Existing notebooks/functions still work.

See [notebook 06](notebooks/06_fast_fes_backend.ipynb) and the [ESTK comparison](docs/estk_review.md). On the recorded 249,571-frame ordinary-CV benchmark, the new method took 0.66 s; the local method took about 12.35 s. Both agree numerically with an independent exact KDE in the tested region. These are warmed local kernel timings, not CSCS trajectory-throughput promises. The new method is not guaranteed fastest for extremely narrow kernels or remote grid tails.

The current release passes 50 tests. Historical validation sections below describe the earlier 38/44-test releases. ESTK's single-run estimator is good; its default independent-run combination and NaN-gap MEP handling require the cautions documented in the comparison. `binned` remains an explicit approximation, not an exact substitute.

## Install and start

```bash
python -m pip install -e '.[notebooks,test]'
pytest -q
jupyter lab notebooks/
```

Python >=3.10. Runtime dependencies: NumPy, SciPy, MDAnalysis, Numba, Matplotlib,
and tqdm. No CP2Kdata, ECToolkits, PLUMED installation, or custom plotting style
is required. Numba compiles the hot loops at first use and caches the result.
`NUMBA_DISABLE_JIT=1` offers a slower debugging mode.

The first four notebooks show the standard workflow; notebook 05 shows custom functions:

| Notebook | Purpose |
|---|---|
| `01_geometry.ipynb` | Real sample trajectory; selections, PT paths, atom identities, coordination, caching |
| `02_reweighting_and_fes.ipynb` | Ordinary CVs, original OPES weights, direct/indirect FES, populations |
| `03_minimum_energy_path.ipynb` | Your parametric string interface, convergence, optional automatic initialization |
| `04_multiple_runs_and_extensions.ipynb` | Independent runs, common-reference averages, block checks, future observables |
| `05_custom_functions.ipynb` | Replace the selector, reselect from cache, define new CVs |

Notebooks 02-04 run on **explicitly labelled synthetic examples** until you set
production paths. The real sample in notebook 01 tests geometry only. Its matching
COLVAR was not supplied: it must not be paired with the earlier `COLVAR0` by row
number. Nothing in the example output is a converged physical PT barrier.

## The complete one-run workflow

### 1. Geometry, once

```python
import numpy as np
from bivo4_fes import *

trajectory = 'output.xyz'
symbols = xyz_symbols(trajectory)
selection = PTSelection.from_symbols(
    symbols,
    donor=151,
    acceptors=[131, 137, 139],   # YOUR candidate surface acceptors
    slab_atoms=144,
    index_base=1,               # these are PLUMED atom IDs
)
frames = iter_xyz(
    trajectory,
    cell=[10.39, 10.18, 36.17],  # confirm your actual simulation cell
    timestep_ps=0.0005,         # EXAMPLE: original MD step, not saved-frame interval
)
pt = analyze(frames, selection, cv_acceptor=130, keep_paths=True)
pt.save('pt.npz')
```

The `PTSelection` constructor converts your chosen index base. All stored indices
and lower-level function arguments are **zero-based**: `cv_acceptor=130` means
PLUMED O131. Water-derived oxygen is inferred from atoms after the first 144 slab
atoms, not from current H coordination. Alternatively supply `waters=[...]` and
`hydrogens=[...]` in the constructor's chosen index base. `by_z` and
`notebook_bi_water` implement frozen first-frame spatial selections.

`PTSelection.colleague(symbols)` reproduces the lists in the supplied scripts,
except that an endpoint cannot also be its own water bridge. In particular the
colleague's **Python atom 241 = PLUMED atom 242** was reassigned to the surface.
Do not assume that happened in your new run. The supplied scripts' Python atom
139 is PLUMED atom 140, not O139.

`analyze` extracts BOTH mechanisms in one pass, plus donor/surface O-H coordination
and the original distance CV for checking trajectory/COLVAR identity. It stores
only compact observables, not the entire trajectory. Set `keep_paths=False` when
candidate-path diagnostics are no longer needed.

### 2. Match the original bias to the SAME configurations

```python
pt = PTResult.load('pt.npz')
c = read_colvar('COLVAR')
rows = align_colvar(pt.time, c)
check_cv_alignment(pt, c, rows)   # recomputed d1 and cn must agree

bias = c['total_bias'][rows]     # only when this contains the biases to remove
lw = opes_logweights(bias, temperature=350, unit='kJ/mol')
keep = pt.time >= 2000.0         # EXAMPLE discard time in ps; choose from convergence
```

For the ORIGINAL five walls given in the conversation, instead use
`(c['opes.bias'] + original_walls(c))[rows]`. This reconstructs precisely those
walls, not arbitrary input files. Do not add the walls twice. To retain a restraint
in the target ensemble, do not remove that restraint's bias. OPES `rct` is NOT
subtracted for this workflow.

The reader never guesses a missing time. A plain XYZ with `Timestep: N` uses
`N*timestep_ps`; a file without steps can instead use an explicitly supplied
`frame_dt_ps`. Decimation changes frame spacing, not the physical MD timestep.
Restart segments with reset/duplicated times must be treated separately.

### 3. Any ordinary CV or newly computed PT CV -> the same FES function

```python
ordinary = compute_fes(
    np.column_stack((c['d1'][rows][keep], c['cn'][rows][keep])),
    lw[keep], bounds=[[2, 4.5], [0.6, 2.6]], bins=201,
    bandwidth=[0.04, 0.04], temperature=350, unit='eV',
)

import numpy as np
surfaces = {}
for channel in ('direct', 'indirect'):
    q = pt.channel(channel)
    surfaces[channel] = compute_fes(
        np.column_stack((q['delta'], q['distance']))[keep], lw[keep],
        bounds=[[-1.5, 1.5], [2.3, 3.5]], bins=[301, 201],
        bandwidth=[0.03, 0.03], temperature=350, unit='eV',
    )
    surfaces[channel].save(f'{channel}.npz')
    surfaces[channel].to_plumed(f'{channel}.dat', ['delta', 'dOO'])
    surfaces[channel].plot(vmax=1.5)
```

`bandwidth` is in the coordinates' units, **not grid cells**. The numbers above
are demonstration choices, not fitted parameters for your data. The colleague's
notebook uses 0.003 A and 101 points over delta=[-1.5,1.5], which samples the density
on a grid much coarser than that bandwidth. Test grid/bandwidth convergence rather
than assuming the same appearance implies the same barrier.

Default normalization is conditional on valid configurations of that channel.
`log_density` retains normalization; `free_energy` is minimum-shifted for plotting.
See `docs/methods.md` before comparing separate channel free energies or rates.

### 4. Minimum-energy path

```python
fes = surfaces['direct']
x, y = fes.axes
# Choose endpoints on adequately sampled, connected support.
S = String2D(x, y, fes.free_energy, indexing='ij')
S.compute_mep(begin=[-0.8, 2.8], end=[0.8, 2.8], mid=[[0.0, 2.5]],
              dt=0.01, maxsteps=2000, npts=35, flexible=False)
if not S.converged:
    raise RuntimeError('Inspect convergence and sampling before using this path.')
path, energy = S.get_mep_energy_profile()
S.plot_mep()
```

Endpoints here are examples, not inferred for your trajectory. `minimax_path`
provides an optional grid initializer. It is not itself a relaxed MFEP. You may
also keep using your original `stringmethod_parametric.py`; it is included in
`reference/`. Improvements to the bundled `String2D` are itemized below.

## Customize from your notebook

The FES functions accept ordinary arrays from any source. `analyze` also accepts
`path_selector=my_function`, and `pt.reselect(my_function)` changes selections
from cached candidates without reading the trajectory again. See
[custom functions](docs/custom_functions.md) and notebook 05. No plugin framework
or custom class hierarchy is required.

## Why it is fast

- **One sequential read**, both PT channels in the same frame loop.
- **Local geometry**: candidate O-O contacts are rejected before hydrogen loops;
  the orthorhombic hot loops are compiled with Numba. General triclinic cells use
  MDAnalysis' minimum-image geometry, not naive fractional-coordinate rounding.
- **Compact `.npz` caches** with saved frame IDs, steps, times, selections, and
  per-path atom identities. No object arrays or pickle required.
- **Local Gaussian KDE**, accumulating in log space at grid points. It avoids an
  all-frames x all-grid-points allocation and retains rare weighted basins.
- **Optional binned KDE** for preview/large jobs; an explicit approximation with
  a minimum grid-resolution requirement. No FFT roundoff floor or arbitrary
  probability floor is used to invent finite free energies in empty regions.

`method='local'` (default) evaluates Gaussian kernels within 12 sigma along each
coordinate. `method='exact'` is the slower, nontruncated reference.
`method='binned'` bins coordinates before positive convolution: refine the grid
and compare its relevant barrier region with `local` before reporting results.
The reported local ESS is a **weight-concentration diagnostic**, not a count of
independent time samples. Empty support remains infinite/masked.

## Other trajectory formats and very large runs

MDAnalysis handles LAMMPS dumps, chained trajectories, XTC, DCD, etc.:

```python
import MDAnalysis as mda
u = mda.Universe('topology.xyz', ['part0.dump', 'part1.dump'], format='LAMMPSDUMP')
# Confirm topology order equals the sorted atom-ID order in the dumps.
frames = iter_mdanalysis(u, timestep_ps=0.0005)
pt = analyze(frames, selection, cv_acceptor=130)
```

Reader time is opt-in (`use_reader_time=True`) because some readers fabricate a
1-ps interval. Supply a correct cell for XYZ. XYZ support is deliberately limited
to species,x,y,z as the first columns; use MDAnalysis for other layouts.

To bound the observable cache as well as trajectory memory:

```python
for n, chunk in enumerate(analyze_chunks(frames, selection, chunk_size=5000,
                                         cv_acceptor=130, progress=False)):
    chunk.save(f'pt_chunk_{n:04d}.npz')
# Concatenate observables later; coordinates are not reloaded:
pt = PTResult.concatenate(PTResult.load(p) for p in sorted(chunk_files))
```

The temporary coordinate chunk is bounded by `chunk_size`. Normal `analyze`
keeps only one coordinate frame at a time. One run per process/job is the simplest
parallelization across four MLPs. There are no hidden worker pools or duplicated
trajectories. Do not concatenate independently shifted OPES biases across models.

## Command line (optional)

```bash
bivo4-fes geometry output.xyz --config examples/geometry_config.json --output pt.npz
bivo4-fes fes COLVAR --cvs d1 cn --bias total_bias \
  --bounds 2 4.5 0.6 2.6 --bins 201 201 --bandwidth .04 .04 --output fes.npz
```

Geometry CLI reads XYZ; for other formats use the notebook/Python API above.
For PT reweighting the notebook shows the complete explicit time-alignment step.

## Validation and limits

Original v0.1.0 validation: **38 local tests passed; all four original notebooks executed**.
The v0.1.1 extension adds six tests; see `docs/comparison_status.md` for the rerun.
The GitHub Actions workflow is supplied, but has not run on a published repository.

See `docs/validation.md` and `benchmarks/validation.json` for measurements, test
scope, and limitations. `pytest` covers periodic boundaries, reverse/shared
protons, both channels, unequal grids, weight offsets, exact KDE comparison,
units, chunk concatenation, timestamps, missing support and reversed MEP endpoints.

**Important: this is not yet a verified drop-in reproduction of ECToolkits'
path selector.** The uploaded drivers import its core but do not contain it.
The repository is visible, but that source file and the second colleague's
repository could not be retrieved in this execution environment. This package
implements the published CV equations with explicit, documented ownership and
competing-path rules. Those rules may differ from the upstream implementation.

Compare against saved legacy results before claiming reproduction:

```bash
python benchmarks/compare_legacy.py pt.npz DirectPTOsi-mlp0.npy --channel direct
python benchmarks/compare_legacy.py pt.npz IndirectPTOsi-mlp0.npy --channel indirect
```

An optional `benchmarks/run_legacy.py` adapter follows the uploaded call signature
for running an installed ECToolkits on the same XYZ. It was not executable here.
Use identical acceptor and bridge selections in both implementations.

Alternatively, reuse **the legacy CV arrays themselves** with this package's
faster `compute_fes`. That preserves the original PT classification exactly;
matching bias/frame alignment is still your responsibility. No speedup versus
ECToolkits is claimed without running that comparison.

The short sample cannot establish FES convergence, physical mechanism rates,
production performance on CSCS, or ML potential accuracy. Reweighting does not
create missing configurations. A graph connection is not a reactive trajectory.

## Package map

```text
src/bivo4_fes/
  selection.py     atom IDs, reference-frame spatial selections
  io.py            streaming coordinates, COLVAR, timestamp/identity matching
  geometry.py      direct/one-bridge paths and compact caches
  coordination.py  smooth coordination and conservative site assignment
  reweight.py      OPES log-weights, walls, populations, block diagnostics
  fes.py           1D/2D KDE, grids, multi-model descriptive averages
  mep.py           parametric string and optional grid initializer
  cli.py           small command-line entry points
```

The string method retains the parametric initialization and public workflow.
Changes fix the `xy` gradient-axis ordering, zero-gradient division, repeated-call
state, and unsupported-grid extrapolation. Bilinear regular-grid interpolation
replaces repeated scattered interpolation. Nonconvergence is never presented as
success. Metric scales are explicit; default `(1,1)` reproduces the raw CV metric.

## Sources and attribution

- Zhuang & Pasquarello, *Angew. Chem. Int. Ed.* **2025**, e202507071,
  https://doi.org/10.1002/anie.202507071, and the user-supplied analysis notebooks.
- [ECToolkits](https://github.com/robinzyb/ECToolkits): the original analysis API.
  Its source is not vendored here. The project identifies an LGPL-3.0 license;
  preserve that license if you later copy code from it.
- [Enhanced sampling toolkit](https://github.com/giopale/enhanced-sampling-toolkit):
  requested reference; source could not be inspected here, no code copied.
- [String method](https://github.com/apallath/stringmethod), Akash Pallath, MIT,
  and the user-supplied parametric modifications.
- [PLUMED OPES documentation](https://www.plumed.org/doc-v2.10/user-doc/html/_o_p_e_s__m_e_t_a_d.html)
  for quasi-static reweighting, total bias, and the `rct` distinction.
- [MDAnalysis distances](https://docs.mdanalysis.org/stable/documentation_pages/lib/distances.html)
  for the general periodic geometry backend.

New package code is MIT-licensed; see `LICENSE` and `reference/README.md`.
The user's coordinate data and upstream references are **not** relicensed by that
statement. Example research data and generated outputs are git-ignored. Review
permissions before publishing anything beyond the code.
