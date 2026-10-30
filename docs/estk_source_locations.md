# Source locations in the user-supplied ESTK archive

Line numbers below refer to the unchanged files extracted from the supplied ZIP. These excerpts document the review; no source is vendored in bivo4-fes.

## `REFACTOR.md`, lines 130-149

SHA256: `2bbc6a43ed53165011dbe437f1df356a8ecd4ca928065b2b150b1f180f48cfef`

```text
130: ## Status overview
131: 
132: | # | Functionality | Status |
133: |---|---|---|
134: | 1 | [Packaging and entry point](#1-packaging-and-entry-point) | in progress |
135: | 2 | [Configuration](#2-configuration) | done |
136: | 3 | [Logging and error handling](#3-logging-and-error-handling) | done |
137: | 4 | [Run discovery and simulation metadata](#4-run-discovery-and-simulation-metadata) | done |
138: | 5 | [CV data loading](#5-cv-data-loading) | done |
139: | 6 | [Frame weights and KDE reweighting](#6-frame-weights-and-kde-reweighting) | done |
140: | 7 | [Combining runs](#7-combining-runs) | done |
141: | 8 | [Bandwidth selection](#8-bandwidth-selection) | done |
142: | 9 | [Surface file format](#9-surface-file-format) | done |
143: | 10 | [Minimum-energy path (string method)](#10-minimum-energy-path-string-method) | done |
144: | 11 | [Barrier convergence](#11-barrier-convergence) | done |
145: | 12 | [Surface plotting](#12-surface-plotting) | done |
146: | 13 | [CVs along the path](#13-cvs-along-the-path) | todo |
147: | 14 | [Structure extraction](#14-structure-extraction) | todo |
148: | 15 | [Proton-transfer CVs](#15-proton-transfer-cvs) | todo |
149: | 16 | [Diagnostic plots](#16-diagnostic-plots) | done |
```

## `src/estk/reweight/kde.py`, lines 135-195

SHA256: `85b952e00a2833101a321a93311d630835346351e8e865aa209e7e2199d0acc0`

```text
135: # Below this, a factorised sum may have lost terms to underflow: recompute in log space.
136: _UNDERFLOW = 1e-200
137: 
138: 
139: def _separable_2d(
140:     x: np.ndarray,
141:     log_weights: np.ndarray,
142:     grid: Grid,
143:     transform: np.ndarray,
144:     quantities: Mapping[str, np.ndarray],
145:     memory_bytes: int,
146: ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
147:     """Fast path for a diagonal bandwidth on a 2D grid.
148: 
149:     The kernel factorises, exp(-dx^2/2 - dy^2/2) = A[a, i] B[b, i], so each sum over frames is
150:     the matrix product A diag(w) B^T. Rows of A and B are scaled by their largest entry to stay
151:     in range; grid points whose scaled sum is tiny (possible loss to underflow) are recomputed
152:     with the direct log-space sums.
153:     """
154:     gx = grid.axes()[0] * transform[0, 0]
155:     gy = grid.axes()[1] * transform[1, 1]
156:     n_frames = len(x)
157:     chunk = max(1, int(memory_bytes // (8 * 3 * (len(gx) + len(gy)))))
158: 
159:     def factors(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
160:         log_a = log_weights[None, start:stop] - 0.5 * (gx[:, None] - x[None, start:stop, 0]) ** 2
161:         log_b = -0.5 * (gy[:, None] - x[None, start:stop, 1]) ** 2
162:         return log_a, log_b
163: 
164:     shift_a = np.full(len(gx), -np.inf)
165:     shift_b = np.full(len(gy), -np.inf)
166:     for start in range(0, n_frames, chunk):
167:         log_a, log_b = factors(start, min(start + chunk, n_frames))
168:         shift_a = np.maximum(shift_a, log_a.max(axis=1))
169:         shift_b = np.maximum(shift_b, log_b.max(axis=1))
170:     shift_a[~np.isfinite(shift_a)] = 0.0
171: 
172:     sum0 = np.zeros((len(gx), len(gy)))
173:     sums = {name: np.zeros_like(sum0) for name in quantities}
174:     for start in range(0, n_frames, chunk):
175:         stop = min(start + chunk, n_frames)
176:         log_a, log_b = factors(start, stop)
177:         a = np.exp(log_a - shift_a[:, None])
178:         b = np.exp(log_b - shift_b[:, None])
179:         sum0 += a @ b.T
180:         for name, q in quantities.items():
181:             sums[name] += (a * q[None, start:stop]) @ b.T
182: 
183:     log_max = (shift_a[:, None] + shift_b[None, :]).reshape(-1)
184:     sum0 = sum0.reshape(-1)
185:     sums = {name: s.reshape(-1) for name, s in sums.items()}
186:     log_p, averages = _finish(log_max, sum0, sums)
187: 
188:     redo = sum0 < _UNDERFLOW
189:     if redo.any():
190:         g = grid.coordinates()[redo] @ transform
191:         exact_p, exact_avg = _log_space(g, x, log_weights, quantities, memory_bytes)
192:         log_p[redo] = exact_p
193:         for name in averages:
194:             averages[name][redo] = exact_avg[name]
195:     return log_p, averages
```

## `src/estk/reweight/kde.py`, lines 210-250

SHA256: `85b952e00a2833101a321a93311d630835346351e8e865aa209e7e2199d0acc0`

```text
210: def _bin_frames(
211:     points: np.ndarray,
212:     log_weights: np.ndarray,
213:     transform: np.ndarray,
214:     quantities: Mapping[str, np.ndarray],
215: ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
216:     """Occupied bins of a grid ``_BIN_STEP`` bandwidths fine, as weighted frames.
217: 
218:     Linear binning: each frame's weight is shared between the 2^d corners of its cell. A bin's
219:     quantity is the weighted mean of its frames' values (exact for the kernel averages, since
220:     every frame of a bin then sits at the bin's centre).
221:     """
222:     n, d = points.shape
223:     # the bandwidth along each CV: 1 / |column of the whitening matrix|
224:     widths = 1.0 / np.linalg.norm(transform, axis=1)
225:     step = _BIN_STEP * widths
226:     lo, hi = points.min(axis=0), points.max(axis=0)
227:     cells = np.floor((hi - lo) / step).astype(int) + 2
228:     if np.prod(cells.astype(float)) > _MAX_BINS:
229:         return points, log_weights, dict(quantities)
230:     weights = np.exp(log_weights - log_weights.max())
231:     position = (points - lo) / step
232:     base = np.clip(np.floor(position).astype(int), 0, cells - 2)
233:     frac = position - base
234:     total = np.zeros(int(np.prod(cells)))
235:     sums = {name: np.zeros_like(total) for name in quantities}
236:     for corner in range(2**d):
237:         offset = np.array([(corner >> k) & 1 for k in range(d)])
238:         share = weights * np.prod(np.where(offset, frac, 1 - frac), axis=1)
239:         index = np.ravel_multi_index(tuple((base + offset).T), cells)
240:         total += np.bincount(index, share, minlength=total.size)
241:         for name, q in quantities.items():
242:             sums[name] += np.bincount(index, share * q, minlength=total.size)
243:     occupied = np.flatnonzero(total > 0)
244:     centres = lo + np.column_stack(np.unravel_index(occupied, cells)) * step
245:     with np.errstate(invalid="ignore"):
246:         means = {name: s[occupied] / total[occupied] for name, s in sums.items()}
247:     return centres, np.log(total[occupied]) + log_weights.max(), means
248: 
249: 
250: def _whitening(bandwidth: np.ndarray, ndim: int) -> np.ndarray:
```

## `src/estk/reweight/combine.py`, lines 1-20

SHA256: `53267891fcfde26ad887e4028d542e274b02fe48cdd95f55e96ce9d6cc3e4db7`

```text
1: """Combining the surfaces of several runs into one.
2: 
3: Modes
4: -----
5: ``probability`` (default)
6:     Sum the runs' densities, each with its total weight ``sum_i exp(V_i / kT)``. This is the
7:     same as reweighting all frames of all runs together. Kernels are normalised per run, so runs
8:     with different bandwidths are combined correctly.
9: ``equal``
10:     Average the runs' normalised densities: every run counts the same.
11: ``legacy``
12:     What the original scripts did, to compare with old ``-AVG.dat`` files: average the per-run
13:     free energies (each shifted to minimum 0) with run weights ``sum_i exp((V_i - V_max) / kT)``,
14:     where ``V_max`` is each run's own largest bias. Those weights are not on a common scale
15:     between runs. The error column is the unweighted standard deviation between runs.
16: 
17: For ``probability`` and ``equal``, the error column is the spread of the runs around the
18: combined value, weighted like the combination (free energies: the spread of the per-run
19: free energies; properties: of the per-run averages, weighted by each run's density there).
20: """
```

## `src/estk/reweight/combine.py`, lines 53-67

SHA256: `53267891fcfde26ad887e4028d542e274b02fe48cdd95f55e96ce9d6cc3e4db7`

```text
53:     kbt = float(first.metadata["kbt"])
54:     log_totals = np.array([_meta(s, "log_weight_total") for s in items])
55: 
56:     if mode == "legacy":
57:         return _legacy(runs, items, log_totals)
58: 
59:     # Log of each run's normalised density on the grid (kernels normalised to integrate to 1).
60:     log_density = np.array([_log_density(s, kbt) for s in items])  # (runs, *grid)
61:     run_log_weight = log_totals if mode == "probability" else np.zeros(len(items))
62:     run_log_weight = run_log_weight - _logsumexp(run_log_weight)  # shares sum to 1
63:     shares = np.exp(run_log_weight)
64: 
65:     expand = (slice(None),) + (None,) * first.grid.ndim
66:     terms = run_log_weight[expand] + log_density  # log of each run's contribution
67:     log_total = _logsumexp_axis0(terms)
```

## `src/estk/paths/mep.py`, lines 55-78

SHA256: `acf9b304ae6692f0869b617c4a543eb921bd435c4c3845ce9aa959ba9a55a13e`

```text
55:     def barrier(self) -> float:
56:         """Highest energy along the path minus the energy at the start."""
57:         return float(np.nanmax(self.energies) - self.energies[0])
58: 
59:     @property
60:     def reverse_barrier(self) -> float:
61:         """Highest energy along the path minus the energy at the end."""
62:         return float(np.nanmax(self.energies) - self.energies[-1])
63: 
64:     @property
65:     def reaction_energy(self) -> float:
66:         """Energy at the end minus the energy at the start."""
67:         return float(self.energies[-1] - self.energies[0])
68: 
69:     def profile_barriers(self) -> dict[str, float]:
70:         """The barrier (highest minus start) of each profile in ``profiles``."""
71:         return {k: float(np.nanmax(v) - v[0]) for k, v in self.profiles.items()}
72: 
73:     def save(self, path: Path | str) -> Path:
74:         path = Path(path)
75:         fields = ["s", *self.names, "fes"]
76:         columns = [self.s, *self.points.T, self.energies]
77:         if self.error is not None:
78:             fields.append("fes.err")
```

## `src/estk/paths/mep.py`, lines 219-242

SHA256: `acf9b304ae6692f0869b617c4a543eb921bd435c4c3845ce9aa959ba9a55a13e`

```text
219: 
220:     values = np.array(surface.values, dtype=float)
221:     finite = np.isfinite(values)
222:     ceiling = float(values[finite].max()) if cap is None else float(cap)
223:     capped = np.where(finite, np.minimum(values, ceiling), ceiling)
224: 
225:     knots = [list(start), *[list(v) for v in via], list(end)]
226:     if not via:
227:         # As the original scripts: the initial guess goes through the midpoint.
228:         knots = [list(start), list((np.array(start) + np.array(end)) / 2), list(end)]
229:     result = evolve_string(
230:         capped,
231:         surface.grid.axes(),
232:         initial_string(knots, images),
233:         dt=dt,
234:         max_steps=max_steps,
235:         fixed_ends=fixed_ends,
236:     )
237:     if not result.converged:
238:         log.warning("the string did not converge in %d steps", max_steps)
239: 
240:     points = result.points
241:     energies = evaluate(surface, points)
242:     error = evaluate(surface, points, surface.error) if surface.error is not None else None
```

## `to_refactor/commands/proton_transfer_cv.py`, lines 100-136

SHA256: `83ed73d0e915ea781634154a0946c701e125b6ad25dc53ebca7f555a743728d7`

```text
100:         logger.info('#'*40 + ' computing direct PT')
101:         options_dir = {
102:             'atomgroup': atomgroup,
103:             'idxs_type1_o': indices_type1,
104:             'idxs_type2_o': indices_type2,
105:             ' extra_detail': True
106:         }
107: 
108:         direct = ProtonTransferCV(**options_dir)
109:         direct.run()
110: 
111:         mask = ~np.isnan(direct.results[:, 1])
112:         logger.info(f'proton transfers found: {np.count_nonzero(mask)} ({np.count_nonzero(mask) / len(u.trajectory) * 100:.3g}% of {len(u.trajectory)} frames)')
113: 
114:         direct.results[:, 0] *= timestep
115:         write_pt_cv(direct.results[:, 0:2], jj / f'CV-{cvs[idx_delta]}', **{'name_cv_x': f'{cvs[idx_delta]}'})
116:         write_pt_cv(direct.results[:, 0:4:2], jj / f'CV-{cvs[idx_delta-1]}', **{'name_cv_x': f'{cvs[idx_delta-1]}'})
117: 
118:     # indirect PT
119:     if pp.kind is None or pp.kind == 'indirect':
120:         logger.info('#'*20 + 'computing indirect PT')
121:         options_ind = {
122:             'atomgroup': atomgroup,
123:             'idxs_type1_o': indices_type1,
124:             'idxs_type2_o': indices_type2,
125:             'num_bridge': 1,
126:             'idxs_water_o': indices_water,
127:             ' extra_detail': True
128:         }
129: 
130:         indirect = ProtonTransferCV(**options_ind)
131:         indirect.run()
132: 
133:         mask = ~np.isnan(indirect.results[:, 1])
134:         logger.info(f'proton transfers found: {np.count_nonzero(mask)} ({np.count_nonzero(mask) / len(u.trajectory) * 100:.3g}% of {len(u.trajectory)} frames)')
135: 
136:         indirect.results[:, 0] *= timestep
```

## `to_refactor/ectoolkits_reweight.py`, lines 645-668

SHA256: `89ae2cc0ce2e7503409d96ff8f7674593af6805fdedfaa361dee340f5e06e287`

```text
645: #                                               point_y=fes_y[i, j],
646: #                                               sigma_x=bandwidth_x,
647: #                                               sigma_y=bandwidth_y,
648: #                                               kbT=kbT,
649: #                                               dim=dim,
650: #                                               )
651: 
652: # parallel version
653: 
654:         tasks = [(i, j, fes_x, fes_y, cv_x, cv_y, bias, bandwidth_x, bandwidth_y, kbT, dim)
655:                         for i in range(num_grid_points_x)
656:                         for j in range(num_grid_points_y)]
657: 
658:         allweights=np.zeros(len(bias))
659:         with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
660:                 for idx, (i, j, value, weights) in enumerate(tqdm(executor.map(calc_one_FES_point_wrapper, tasks), 
661:                                                                   desc='working', total=len(tasks), ncols=80)):
662:                         fes[i, j] = value
663:                         allweights += weights
664:         if len(special_points)>0:
665:             for kk, pt in enumerate(special_points):
666:                 
667:                 logger.info(f'special point [{pt[0]:.3f},{pt[1]:.3f}] has weight {np.sum(special_points_weights[kk])/np.sum(allweights):.4g}')
668:                 thr=.9
```
