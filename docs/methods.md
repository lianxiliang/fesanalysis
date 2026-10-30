# Definitions and scientific choices

## Frame geometry, not trajectory kinetics

For each frame, for each hydrogen, find the nearest oxygen using periodic
minimum-image distances. This oxygen determines the angle vertex and covalent
ownership. There is no assumption that the transferred H has a fixed atom ID,
and no hard covalent cutoff that would discard a symmetric shared proton.

A connection between oxygen A and B is admissible when:

1. `r(A,B) < oo_cutoff` (default 3.5 A).
2. A hydrogen owned by A or B makes an angle smaller than `angle_cutoff`
   (default 30 degrees) between the owner-to-other-O vector and owner-to-H vector.

The angle vertex is **an oxygen**, not the hydrogen. This implements the
O-O-H convention in the supplied paper. Both orientations are accepted so that
product configurations remain present after the proton switches owner.

For an ordered connection A -> B through H:

```
delta = r(A,H) - r(B,H)
d     = r(A,B)
```

A direct candidate is donor -> surface acceptor. An indirect candidate is
donor -> water-derived oxygen -> surface acceptor with **distinct H atoms** on
the two links. Endpoints cannot also be bridges. The indirect coordinates are:

```
delta = (delta1 + delta2)/2
d     = (d1 + d2)/2
```

The individual `delta1`, `delta2`, `d1`, `d2` are retained. Inspecting
`F(delta1,delta2)` can reveal intermediates hidden by their average.

### Competing paths

By default select one candidate **per channel per frame**: minimum average O-O
distance, then maximum angular linearity, then deterministic atom-ID ties.
This is a documented analysis convention, not something verified from ECToolkits'
unavailable core. It is a starting point to be compared against that code.

Direct and indirect connections can coexist. They are evaluated independently,
not `if direct: ... elif indirect: ...`. The result therefore does not partition
all configurations into two mutually exclusive reaction mechanisms. Neither label
says a chemical event actually occurred in that frame.

`keep_paths=True` retains every admissible connection. `all_path_samples` offers
an alternative density: distribute a frame's statistical weight equally over its
candidates. Without that division the result is a *path-occurrence* density,
which gives multiply connected frames more weight. These are different
observables; choose and report one. Do not mix the definitions between runs.

The current nearest-oxygen rule and H-bond geometry should be verified visually
for representative reactants, shared-proton configurations and products. Multiple
proton defects or unusually stretched bonds can require a different definition.

## Atom selection

The original scripts use zero-based donor 150 and acceptors
`[130, 241, 136, 139, 129]`. Their comment says atom 241 became part of the solid.
That is a chemical observation about that trajectory, not a transferable rule.

The API accepts manually chosen IDs or frozen first-frame z selections.
`slab_atoms=144` is only a convenient ordering convention for this system.
Not all near-surface oxygens are necessarily in the paper's O_si class: that
classification depends on bonding to the selected Bi. Review the selected atoms
rather than silently defining O_si from z alone.

## Reweighting

For a suitably equilibrated, quasi-static OPES segment at T, remove the specified
bias using `log w = V_removed/(kB*T)`. The same weight belongs to a configuration
regardless of the new observable plotted. Do not subtract `opes.rct`.

Choose the target ensemble first. If all restraints are to be removed, include
all wall biases once. If a Bi-O restraint defines the target ensemble, retain it
and remove only OPES and the other selected biases. Removing a strong restraint
can produce severe weight concentration; it cannot recover states never sampled.

Align by time AND compare recalculated original CVs. Index matching is insufficient
for a decimated trajectory or files from different runs. Independently restarted
segments may have duplicate timestamps and need explicit handling.

For category C and a product mask P:

```
p(C | P) = sum_t exp(logw_t) * I_P(t) * I_C(t)
           / sum_t exp(logw_t) * I_P(t)
```

Category -1 means unassigned/shared/multiple, not automatically solvent hydronium.
Distinguishing these requires additional oxygen/hydrogen analysis. No forceful
argmax over surface oxygens is used when none is convincingly protonated.

Kish ESS is `(sum w)^2/sum(w^2)`. It quantifies weight concentration only. Time
correlations can further reduce information; contiguous block checks are included,
but no automatic independent-sample uncertainty is asserted.

## Channel-restricted FES normalization

`compute_fes` accumulates weighted Gaussian kernels. Missing CVs (`NaN`) are
excluded together with their weights. The default density is conditional on the
selected frames, e.g. `P(delta,d | direct connection)`.

Pass `log_normalizer=logsumexp(all_retained_frame_logweights)` instead to calculate
a joint density `P(delta,d,direct connection)`. This retains the statistical weight
of that channel relative to all retained frames. The `free_energy` array is always
minimum-shifted for plotting; use `-kBT*log_density` when absolute *relative*
normalization between two observables is important. Coordinates and measures must
also be consistent before comparing these densities.

Do NOT infer a direct/indirect rate ratio simply from two independently shifted,
restricted FESs. Their state definitions, reactant normalization, hidden slow
variables, sampling, and kinetic prefactors need additional justification.
A geometry path is not a committor-defined transition state. Ordinary OPES time
is biased and does not directly give unbiased residence times/hopping rates.

## KDE choices

- `local`: no coordinate binning; each Gaussian is evaluated at grid points up
  to `cutoff` sigmas in each dimension. Default cutoff=12. Log-space accumulation
  avoids underflow and cancellation in rare-event densities.
- `exact`: full Gaussian tails; an intentionally slow reference, with bounded
  temporary arrays. Compare small subsets or a reduced grid.
- `binned`: positive binning + separable convolution. Faster, but approximate.
  Requires grid spacing <= bandwidth/2, and still needs convergence tests.

The local backend reports a conservative absolute density-tail bound. KDE values
outside sampled regions are not evidence of sampling. Gaussian smoothing and grid
resolution must be tested near the intended path. No arbitrary density floor is
inserted. Current KDE coordinates are **nonperiodic**; do not use this directly
for angles on a circle without adding periodic kernels.

## String method

The bundled interface follows the supplied parametric string code. Gradient
interpolation and arc-length reparameterization use consistent axis ordering.
Both image points and segments between images are checked against the supported
grid. No path is allowed through an artificial capped version of an empty FES.

`flexible=False` fixes endpoints; otherwise endpoints relax. `dt` bounds the
largest scaled step in this implementation. `scales=(sx,sy)` defines the metric
via q=(x/sx,y/sy); gradients and arc length are transformed consistently.
The default is the original raw Euclidean metric, not a claim of a unique
physical CV metric.

`converged=True` means the numerical displacement criterion was satisfied without
boundary backtracking. Inspect both `convergence_history` and
`normal_force_history`, repeat with smaller dt and more images, and verify FES
sampling separately. `minimax_path` is a grid initializer only; it may take a
different channel from a manually supplied string.

## Four models

Build each FES independently with its OWN trajectory and bias. `average_fes`
aligns surfaces at a common coordinate and returns their arithmetic mean and
population standard deviation on common finite support. This describes model/run
spread; it is not an exact free energy for one Hamiltonian and is not a standard
error. Do not pool raw independently shifted biases. Keep within-run block
variation separate from between-model variation.


## Optional exact separable estimator (v0.1.2)

`method="separable"` computes the same untruncated Gaussian sum as `exact`, for a diagonal 2D bandwidth. The sum is factorized into axis kernels and evaluated with chunked matrix multiplication; squared factors also provide the second kernel moment for ESS. Grid points vulnerable to exponential underflow are recomputed directly in log space. The ordinary `local` default is unchanged.

Here `count` is all retained finite samples, like the untruncated exact reference, not a local coverage count. Finite Gaussian tails do not establish adequate sampling. The local ESS is a weight-concentration diagnostic, not a count of time-independent configurations. Do not infer support merely from a finite free energy.

There is no automatic coordinate binning in this backend. The existing `binned` method is faster but approximate and can lose extremely small global weights. See the completed ESTK comparison for quantified errors on the provided data.
