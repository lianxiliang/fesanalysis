# Original v0.1.0 validation report

Run `pytest -q` and `python benchmarks/benchmark.py /path/to/output.xyz`.
The JSON report records the actual environment and timings. JIT warm-up and
file reading are reported separately from geometry computation.

## Supplied trajectory

- 2,401 frames; 360 atoms; plain XYZ with original timestep comments.
- Fixed cell used for the test: 10.39 x 10.18 x 36.17 A. The XYZ contains no cell;
  this is the supplied study's cell, not a value inferred from the coordinates.
- Test donor: PLUMED 151; acceptors: 131, 137, 139, 140, 130.
- Independent MDAnalysis minimum-image implementation versus Numba orthorhombic
  implementation of the **same explicitly defined new selection rules**.
- Same candidate counts and missing-frame masks. Maximum discrepancy in numerical
  geometry columns: about 4.6e-7 (distance columns in A). Both selected and all
  retained candidate paths were compared.
- 1,066 frames have a direct connection; 774 have an indirect connection;
  206 have both. These are geometric connections, NOT counts of reaction events.

One measured run (Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0,
Numba 0.65.1, MDAnalysis 2.10.0; see JSON for rerun values):

| Task | Time |
|---|---:|
| Parse full plain XYZ into benchmark memory | 0.71 s |
| Optimized geometry on frames in memory | 0.74 s |
| Independent MDAnalysis/Python reference | 5.49 s |
| Streaming read + optimized analysis | 1.36 s |

Approximately 7.4x faster for geometry than this project's own reference. These
numbers are NOT speedups over ECToolkits, are not a scaling proof, and exclude
first compilation unless the warm-up row is explicitly included. Production
`analyze` streams; the benchmark intentionally holds coordinates to isolate compute.

## KDE validation

Synthetic 4,000-sample, 101 x 101-grid comparison at bandwidth=0.08:

- Local Gaussian vs untruncated reference: maximum FES discrepancy ~2e-15 eV
  in this particular test, with a 12-sigma cutoff.
- Approximate binned KDE: maximum discrepancy ~0.0081 eV below a 0.3-eV FES cutoff
  in this test. This is not a universal bound; test your bandwidth/grid/path.
- Timings ~0.65 s local, ~3.12 s exact, ~0.011 s binned in the recorded run.

Additional tests cover extremely unequal weights, large constant bias shifts,
normalization of conditional versus joint densities, and masking empty support.

## Limits that remain

1. ECToolkits' actual PT core was unavailable, so compatibility with its path
   selection remains unverified. `compare_legacy.py` is supplied for this check.
2. No matching bias file accompanied the subsampled XYZ. Recomputed donor CN and
   distance disagree with the earlier COLVAR0 beyond the first frame for both
   tested MD timesteps (0.5 and 1 fs). A physical direct/indirect FES is NOT validated.
3. The provided sample is too short for a convergence or rate claim.
4. Optional MDAnalysis formats use their upstream readers; the core tests do not
   cover every trajectory format or every cluster filesystem.
5. Notebook demonstration FESs are labelled synthetic, not presented as research
   results. Cold JIT, memory use, and parallel filesystem throughput will differ
   on CSCS. No background workers are spawned by the package.


For the new v0.1.1 tests and actual branch-access limitations see [comparison status](comparison_status.md).
