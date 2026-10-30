# Upstream attribution and compatibility

`stringmethod_parametric.py` is the unmodified user-supplied implementation,
credited to Akash Pallath and marked MIT. It is retained for comparison, not
imported by default. The new `bivo4_fes.mep` retains its parametric string algorithm
and main notebook-facing method names, with documented numerical fixes.

No ECToolkits or enhanced-sampling-toolkit source is copied. Their complete code
could not be retrieved in this environment. The uploaded check_*PT.py scripts
provide the ECToolkits call signature and atom lists, but do not specify its
hydrogen-ownership or competing-path selection algorithm. Thus exact numerical
compatibility must be measured against that core or its saved result arrays.

`benchmarks/compare_legacy.py` compares the exact numerical arrays rather than
comparing pictures. A disagreement requires examination, not smoothing away.
The user-supplied PT notebook's columns 1 and 2 are respectively delta and d.
They can be fed straight into the new KDE to preserve legacy CVs.

The public ECToolkits repository identifies LGPL-3.0. If vendoring its code later,
retain its license, notices and source obligations instead of labelling that code
MIT. Using it as an optional installed dependency avoids copying it into this
package. Review its current license before distribution.
