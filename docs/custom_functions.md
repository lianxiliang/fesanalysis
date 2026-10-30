# Custom functions, not a fixed pipeline

The package's stages are independent. Use whichever ones you need. `compute_fes`
accepts plain NumPy coordinates and log-weights; they may come from PLUMED,
ECToolkits, a different project, or your own functions. There is no CV registry,
required configuration object, prescribed number of runs, or mandatory trajectory
reader at the FES stage. You can also pass any computed grid to your own MEP code.

## A different path-selection rule

`analyze(..., path_selector=my_function)` now accepts a notebook-defined function:

```python
import numpy as np
from bivo4_fes import analyze, PATH_COLUMNS

angle = PATH_COLUMNS.index('angle_score')

def most_linear(candidates):
    return candidates[np.argmin(candidates[:, angle])]

pt = analyze(frames, selection, keep_paths=True, path_selector=most_linear)
```

This is an example alternative convention, not a recommendation that it is
physically better than selecting the shortest O-O route. The callback receives
only nonempty arrays and returns one candidate row. The mechanism is in the
`mechanism` column (0: direct, 1: indirect), so a single function can apply different
rules to the channels. The default rule is unchanged.

Already have a cache made with `keep_paths=True`? No need to read positions again:

```python
alternative = pt.reselect(most_linear)
```

`pt` is unchanged. The cache contains the candidate paths, not a pickled callback;
keep your callback definition in the notebook. Reselection changes which of the
stored candidates is chosen, NOT the hydrogen-assignment or H-bond criteria that
generated those candidates. Changing those criteria requires recomputing geometry.

For complete control, call `frame_paths` directly in your own frame loop, use its
returned arrays, or replace that step entirely with your own implementation.

## A different CV

For example, retain the two separate indirect-transfer coordinates rather than
averaging them:

```python
from bivo4_fes import compute_fes

def my_coordinates(pt):
    q = pt.channel('indirect')
    return np.column_stack((q['delta1'], q['delta2']))

# logweights must correspond to these same frames, in this same order.
fes = compute_fes(
    my_coordinates(pt), logweights,
    bounds=[[-1.5, 1.5], [-1.5, 1.5]],
    bins=201, bandwidth=[0.03, 0.03], temperature=350,
)
```

The bounds and bandwidth are illustrative; choose them for your data. No changes
to package source are required. `compute_fes` also accepts a 1D array.

Essential checks remain: one weight per frame, meaningful units and indices,
valid periodic geometry, and finite inputs. The program does not force a
particular acceptor list, CV definition, state mask, or number of trajectories.
