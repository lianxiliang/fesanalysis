"""Check optimized sums against the existing independent log-space reference."""
import numpy as np
import pytest
from bivo4_fes import compute_fes
from bivo4_fes.separable import gaussian_moments
from bivo4_fes.fes import _exact


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_moments_match_reference(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(71, 2))
    w = rng.normal(0, 30, len(q))
    axes = (np.linspace(-3, 3, 23), np.linspace(-2, 2, 19))
    bw = np.array([.12, .35])
    # A tiny budget forces many chunks; includes off-diagonal coverage gaps.
    a = gaussian_moments(q, w, axes, bw, memory_mb=.005)
    b = _exact(q, w, axes, bw)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-13)


def test_extreme_logweights_and_remote_grid():
    q = np.array([[0., 0.], [2., 2.], [0., 2.]])
    w = np.array([0., -1000., -500.])
    options = dict(bounds=[[-.1, 2.1], [-.1, 2.1]], bins=[13, 11], bandwidth=.04)
    with pytest.warns(UserWarning):
        a = compute_fes(q, w, method='separable', **options)
    with pytest.warns(UserWarning):
        b = compute_fes(q, w, method='exact', **options)
    np.testing.assert_allclose(a.log_density, b.log_density, atol=2e-10)
    np.testing.assert_allclose(a.local_ess, b.local_ess, atol=2e-10)
    assert np.isfinite(a.log_density).all()


def test_bias_shift_and_nan_mask():
    rng = np.random.default_rng(7)
    q = rng.normal(0, .1, (33, 2)); q[0] = np.nan
    w = rng.normal(size=len(q)); w[1] = -np.inf
    opts = dict(bounds=[[-.3, .3]]*2, bins=15, bandwidth=.1, method='separable')
    a = compute_fes(q, w, **opts)
    b = compute_fes(q, w+10000., **opts)
    np.testing.assert_allclose(a.free_energy, b.free_energy, atol=2e-12)
    np.testing.assert_allclose(a.local_ess, b.local_ess, rtol=1e-10)
    assert a.metadata['selected_frames'] == 31


def test_1d_fallback():
    q = np.linspace(-1, 1, 15)
    opts = dict(bounds=[[-1, 1]], bins=15, bandwidth=.2)
    a = compute_fes(q, np.zeros(15), method='separable', **opts)
    b = compute_fes(q, np.zeros(15), method='exact', **opts)
    np.testing.assert_array_equal(a.free_energy, b.free_energy)
