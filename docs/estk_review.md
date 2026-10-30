# Comparison of the supplied ESTK branch with bivo4-fes

## Conclusions

The ESTK branch contains a genuinely useful, accurate, fast **single-run Gaussian KDE** implementation. Its speed is not merely due to coarser plotting or omitted reweighting. On the supplied COLVAR, its unbinned estimator and the original bivo4-fes local estimator agree to numerical precision in the tested free-energy region. The main useful addition to bivo4-fes is an optional exact separable-matrix estimator, now available as `method='separable'` in v0.1.2.

Do not replace the entire PT pipeline on the strength of the timing claim: the refactored ESTK package does not yet implement trajectory-to-PT geometry. Its `REFACTOR.md` lists PT CVs as TODO; the legacy command calls external ECToolkits `ProtonTransferCV`. Therefore this is a comparison of the shared FES step, not a head-to-head comparison of proton-transfer detection or complete trajectory processing.

Two independently reproduced cautions matter: the default cross-run probability combination is not invariant to independent constant bias shifts, and a path crossing a NaN gap can be returned as converged with a finite barrier. Neither invalidates the successfully tested single-run KDE.

## Inputs and experimental setup

- ESTK: the exact uploaded `enhanced-sampling-toolkit-claude-gifted-sagan-lagdek.zip`.
- Original baseline: uploaded `bivo4-fes-v0.1.1.zip`.
- New backend: independently implemented `bivo4-fes-v0.1.2/src/bivo4_fes/separable.py`.
- Actual CV input: `COLVAR0`, 277,301 records. The benchmark retains the last 249,571 after dropping the first 27,730 (10%). This is a reproducible selection, not a finding that this much equilibration is sufficient.
- Coordinates: `(d1, cn)`; grid 101 x 101, d1=[2,4.5], cn=[0.6,2.6]; bandwidths 0.05 in each coordinate's own units; T=350 K; reported energy unit eV.
- Identical log weights in every estimator: `(opes.bias + reconstructed original five walls)/(RT)`, with the original KAPPA=300 quadratic wall definitions. Log-weight range is about 57.66.
- Single BLAS/OpenMP thread, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Numba 0.65.1. This is the local review environment, not CSCS.
- FES timings exclude imports, warm-up, compilation, file parsing, and writing. The first four current timings below are medians of three warm repeats. Baseline local timing is the median of two repeats, with a separate fresh single-repeat check.

File hashes and raw timings are in `results/`. The original trajectory is not paired with COLVAR0: the earlier matching issue remains. This review reconstructs an **ordinary CV FES from COLVAR0 itself**, not a newly validated direct/indirect physical FES.

## 1. Measured FES performance and numerical agreement

| Estimator | Time, s | Maximum absolute grid error below 1.5 eV |
|---|---:|---:|
| ESTK exact, `binned=False` | 0.574 | Reference |
| ESTK default automatic binning | 0.316 | 0.00197 eV |
| bivo4-fes v0.1.1 local | 12.348 | 1.93e-14 eV |
| bivo4-fes v0.1.2 separable | 0.660 | 1.55e-15 eV |
| bivo4-fes binned | 0.071 | 0.03553 eV |

All errors compare independently minimum-shifted surfaces at identical grid coordinates against ESTK's unbinned calculation where its FES is below 1.5 eV. They are numerical differences between KDE estimators, **not errors relative to physical truth, uncertainty bars, or differences in a converged barrier**. All these estimators returned finite values at all 10,201 grid points in this real-data test.

Fair-workload detail: the bivo4-fes calls also calculate a local weight-concentration ESS and contribution counts. ESTK's timed calls above calculate the density only. Adding a second ESTK KDE call for the squared kernels to obtain the same ESS took 1.573 s in the warm median. This is a comparison adapter, not a claim that ESTK could not be further optimized to calculate both together. The two ESS calculations agree to about 9.24e-14 relative error in the region below 1.5 eV.

A separate independent 4,000-sample synthetic test compared against bivo4-fes's direct untruncated log-sum-exp reference: ESTK exact differed by at most 3.33e-16 eV below 0.6 eV; the original local estimator by 1.65e-15 eV. Thus the exact speedup was not obtained by changing the estimator in these tests.

The binned methods are different algorithms: ESTK linearly deposits weights on an auxiliary grid with spacing sigma/8; bivo4-fes currently deposits them on the requested output grid before positive Gaussian convolution. ESTK's approximation is more accurate in this particular comparison; bivo4-fes's is faster. Grid refinement can change both error and timing. No method is globally fastest for every bandwidth, grid, or sample size.

### Why the exact implementation is faster

For diagonal Gaussian bandwidths,

    P[a,b] = sum_i w_i exp(-(x_a-x_i)^2/(2 sigma_x^2))
                         exp(-(y_b-y_i)^2/(2 sigma_y^2)).

Factor the two one-dimensional Gaussian arrays and use matrix multiplication. The exact statistical sum is unchanged, but there are far fewer exponential/logarithmic operations than an all-grid/all-sample log-sum-exp loop. Optimized BLAS does the multiplication. Both the inspected ESTK implementation and the independently written v0.1.2 backend rescale exponentials and use log-space fallback when products risk underflow.

The existing local method evaluates truncated kernels and updates two log sums at every visited grid point. It remains valuable when kernels occupy very few grid points, and it remains the default for backward compatibility. The separable method may need costly direct fallback far from narrow-kernel support; the measured speedup is not a universal guarantee.

ESTK automatically switches on frame pre-binning above 200,000 frames. `kde_on_grid(..., binned=False)` disables that approximation; the high-level wrapper does not expose this switch directly. The 249,571-frame benchmark exercises the automatic binned branch.

## 2. Important issue: independent-run averaging

The default ESTK `combine(mode='probability')` assigns run shares proportional to

    W_r = sum_i exp(V_ri / kT).

For **independently biased runs**, adding a constant C to every bias value in one run changes W_r by exp(C/kT), although it leaves that run's forces and normalized reweighted distribution unchanged. Consequently an arbitrary offset changes the combined surface.

Reproduced two-run experiment:

| Quantity | Before | After adding +5 kT to run B |
|---|---:|---:|
| Run A share | 0.5000 | 0.00669 |
| Run B share | 0.5000 | 0.99331 |
| Change to run B's individual FES | — | 0 |
| Largest change to combined FES | — | 14.509 kJ/mol, about 0.150 eV |

This matters for the user's four independently trained MLPs and independent OPES histories. They should not acquire arbitrary thermodynamic shares from the magnitudes of their bias zeros. Pooling genuinely common-ensemble/common-normalization samples, such as suitable common-bias walkers, is a different situation; the test does not reject that use case.

ESTK's `mode='equal'` passes this offset-invariance test. However, it produces an equal mixture of normalized **probability densities**, not an arithmetic mean of reference-aligned **free energies**. These are different summaries:

    mixture:     -kT log(mean_r P_r(s))
    FES average: mean_r [F_r(s)-F_r(s_ref)].

For the intended four-MLP descriptive mean/spread, retain the reference-aligned per-model FES average (and each model's own result). Neither summary makes a disagreement in physical states or sampling disappear. Do not interpret a mixture as one uniquely defined DFT Hamiltonian.

ESTK's `legacy` mode uses yet another bias-derived set of weights on minimum-shifted FESs; it exists to reproduce previous outputs, not to make these quantities interchangeable.

## 3. MEP comparison

On a complete smooth analytic two-well surface with a known 1.0 eV saddle, both methods converged to a 1.0 eV barrier. Maximum pointwise path separation was about 6.04e-5 in the chosen coordinate units. Times were 0.114 s for bivo4-fes and 0.106 s for ESTK in this small test. Different interpolation order, tolerances, initial strings, and endpoint handling can otherwise create differences.

One edge case is reproducible: give ESTK a NaN strip between the two wells. It fills nonfinite values for optimization, reports a converged path, and its barrier property uses `nanmax` on the interpolated energies. In the test, 5 of 35 images had NaN energies but the reported barrier was 0.93858 eV. That is not a validated barrier across the missing region. bivo4-fes rejects that initial path because it crosses unsupported cells. With an infinity strip, this test produced an infinite rather than a finite ESTK barrier; the demonstrated finite-barrier problem specifically concerns NaN values.

This is a small but scientifically important guard, not a reason to replace an otherwise useful MEP method. No MEP can recover an unobserved barrier merely by filling the missing grid cells.

## 4. Other differences and useful ideas

ESTK has broader workflow automation: generic run discovery, PLUMED/LAMMPS metadata parsing, reconstruction of wall energies, time joins, adaptive bandwidth selection, general-dimensional KDE, full covariance bandwidths, and kernel-conditioned averages of other observables. Its CLI and surface metadata are substantial strengths. The current bivo4-fes focus is a transparent notebook/Python workflow, fast PT geometry, candidate caches, editable path-selection callbacks, and arbitrary input CV/weight arrays.

The refactored ESTK source has no PT geometry implementation yet. The legacy command imports external ECToolkits, contains an `" extra_detail"` keyword with a leading blank, and constructs output times using COLVAR spacing. The behavior of that keyword cannot be established without the external class. These legacy details do not provide an independently benchmarkable replacement PT engine or justify a claim about whole-trajectory throughput.

The legacy reweighting copy in the ZIP submits one task per grid point through a 24-process pool, carrying full CV/bias arrays in the task arguments. That provides a plausible source of large communication and allocation overhead relative to the new matrix implementation, but this review did **not** measure an hours-long original production job or reproduce the full historical speed ratio.

Walls: ESTK defaults to removing only the main OPES bias; `include_walls=True` removes the reconstructed wall energies as well. This is a target-ensemble choice, not automatically a bug. Both implementations must receive equivalent bias conventions for a numerical comparison. The benchmark uses the same explicitly reconstructed total bias in both.

## 5. Numerical limits shared by approximate methods

An intentionally extreme test uses two separated configurations with log weights 0 and -1000. ESTK exact, bivo4-fes local, and the new separable backend preserve the weak peak and its 30.1607 eV relative height. ESTK's binned route loses the weak weight before binning and gives 75.4017 eV at that point; bivo4-fes's binned route gives infinity. This is a floating-point underflow limitation of both approximate paths, not observed in the supplied COLVAR whose log-weight span is only 57.66. It argues for retaining an exact reference and checking tails, not for labeling every binned result invalid.

A finite value from an untruncated Gaussian KDE also does not establish that a region was sampled adequately. Keep the existing support/statistical checks and block comparisons; these optimizations do not cure deficient CVs, poor trajectories, or MLP uncertainty.

## 6. Delivered change and testing

v0.1.2 adds `method='separable'`, a small independently written, chunked exact-diagonal-2D backend. It also computes squared-kernel sums for local ESS. It does not copy ESTK source or introduce an ESTK dependency. Its 1D case uses the existing exact reference. Existing methods and default behavior remain unchanged; geometry and path-selection code are unchanged.

- Original bivo4-fes v0.1.1: 44 tests passed.
- Updated bivo4-fes v0.1.2: 50 tests passed, including new unequal-grid, chunking, bias-offset, extreme-weight, and exact-reference checks.
- ESTK's reweighting, bandwidth and combination modules: 38 tests passed.
- ESTK's supplied old-output regression comparisons: 5 tests passed. This is valuable evidence for backward numerical agreement of its single-run reconstruction.
- Complete ESTK-suite runs timed out in this environment. Several additional modules passed in isolation, but a complete all-tests-passed claim is not made. The passing tests above are the verified scope; a timeout is not used as evidence that the KDE is incorrect.
- Analytic MEP, gap, gauge-invariance, and extreme-weight examples were executed independently; see JSON results.

Use v0.1.2 in a notebook by changing only `method='separable'` in `compute_fes(...)`. Keep manual functions, CV arrays, and candidate-selection callbacks. There is no need to adopt the other project's configuration machinery just to use this mathematical optimization.
