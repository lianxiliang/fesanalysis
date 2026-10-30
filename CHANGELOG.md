# v0.1.2

- Added optional `compute_fes(..., method="separable")`: exact diagonal 2D Gaussian KDE using chunked matrix products, including second moments for local ESS and log-space underflow fallback. 1D falls back to the existing exact reference.
- No change to the default `local` method or PT geometry/selection rules. No new runtime dependency, ESTK source copying, or mandatory configuration.
- Added six tests (50 passing tests in the review environment), an executed notebook, and the frozen-archive comparison report.
- A real-COLVAR benchmark gives 0.66 s for the new backend versus about 12.35 s for the original local backend. This is not an end-to-end trajectory timing or a universal speed bound.

# Changes

## 0.1.1

- Optional `analyze(..., path_selector=...)` callback for notebook-defined rules.
- `PTResult.reselect(...)` applies another selector to cached candidates without
  rereading or reanalyzing trajectory coordinates. Existing v0.1.0 caches work
  when they were saved with `keep_paths=True`.
- Public `PATH_COLUMNS` and `choose_path` helpers.
- Notebook 05 and a short custom-functions guide.
- Six tests for custom selection, unchanged defaults, empty channels, and caching.

The geometry definitions, default path selector, weighting formulas, FES
estimators, and MEP algorithm are unchanged. No reproduction claim against the
unretrieved ECToolkits core or giopale branch has been added.
