> Historical v0.1.1 report. The ESTK ZIP was subsequently supplied; see [the completed comparison](estk_review.md).

# Comparison status and v0.1.1 validation

## Scope

Requested comparison: giopale/enhanced-sampling-toolkit, branch
`claude/gifted-sagan-lagdek`, versus bivo4-fes.

The branch could not be retrieved through the available GitHub page, raw-file,
archive, or git routes. Its code was NOT inspected or executed. No speedup,
correctness, or algorithm-equivalence claim against that branch is made.
No actual ECToolkits PT-core comparison was possible either.

## Fresh local measurements

The uploaded 2,401-frame, 360-atom XYZ was used with a fixed
10.39 x 10.18 x 36.17 A orthorhombic box. This box was supplied by the
study; it is not embedded in plain XYZ. Python 3.13.5 on Linux-6.18.57-x86_64-with-glibc2.41.
BLAS/OpenMP/Numba thread counts were restricted to one. Timings exclude imports
and first JIT compilation; kernels were warmed by the tests and warm-up call.
These are individual timings, not confidence intervals or production scaling tests.

| Task | Seconds |
|---|---:|
| Geometry on coordinates already loaded | 0.660 |
| Streaming XYZ read plus geometry | 1.395 |
| Own independent MDAnalysis geometry reference | 5.555 |
| Exact KDE, synthetic 4,000 samples, 101 x 101 grid | 2.740 |
| Local log-space KDE, same synthetic samples | 0.559 |
| Approximate binned KDE, same synthetic samples | 0.0091 |

The geometry implementations used identical documented candidate rules, yielding
1,066 direct-connected frames, 774 indirect-connected frames, and 206 frames with
both. These are not reaction-event counts. Candidate/missing masks agree; the
largest numerical geometry discrepancy was about 4.6e-7.

Local KDE and the untruncated reference differed by
2.1e-15 eV in this synthetic benchmark. Binned KDE differed
by up to 0.0081 eV below the reference's 0.3 eV
cutoff. This is not an error guarantee for another bandwidth, grid, or trajectory.
No physical PT FES was validated because a matching OPES bias for the uploaded
XYZ was not supplied. The earlier COLVAR0 was previously found not to match.

## Flexibility change

v0.1.1 adds a user-supplied `path_selector` and cached `PTResult.reselect`.
No defaults in geometry, reweighting, FES estimation, or MEP were changed.
Default v0.1.0 and v0.1.1 outputs agree exactly on all 2,401 test frames, including
all stored candidates and coordination numbers. Reselecting this cache took
about 0.015 seconds; no positions were read or distances recomputed.

44 tests passed (38 previous + 6 callback/cache tests). The new
`05_custom_functions.ipynb` notebook was executed. The original four notebook
outputs are retained from v0.1.0, not claimed as freshly rerun here.

## Fair cross-project test once the branch is available

1. Use the same positions, cell, atom selections, original bias, and retained frames.
2. Time coordinate parsing, path construction, KDE, and MEP separately, distinguishing
   cold compilation from warm execution and cached CVs from full trajectory work.
3. Compare missing masks, all available candidate identities, selected donor/acceptor/
   bridge/hydrogens, component CVs, and their units before comparing FES pixels.
4. To isolate KDE implementations, feed identical CV and log-weight arrays and use
   the same physical bandwidth, grid points, normalization, and energy unit.
5. Compare densities/support and barriers on a common reference; a visually similar
   contour plot or a passed timing test is not enough to establish equivalence.

A different scientific selection rule can be intentional. Record and compare it
as a different observable rather than silently calling it an implementation bug.
