"""Exact diagonal 2D Gaussian sums using bounded-size matrix products.

This is an independent implementation of Gaussian separability, not vendored
ESTK code. Two passes keep exponentials scaled. Very small second moments are
recomputed with log-sum-exp, so rare weighted regions are not lost to underflow.
"""
import numpy as np
from scipy.special import logsumexp


def gaussian_moments(samples, logweights, axes, bandwidth, memory_mb=64):
    """Return log(sum kernels), log(sum kernels**2), and sample counts.

    Input arrays are already checked by compute_fes. Counts include all samples,
    as for the untruncated exact backend; they are not local support counts.
    Temporary memory scales with chunk_size*(nx+ny), not frames*nx*ny.
    """
    x, y = axes
    n = len(samples)
    chunk = max(1, min(n, int(memory_mb * 2**20 / (32 * (len(x) + len(y))))))
    offset = float(np.max(logweights))
    half_weight = .5 * (logweights - offset)

    def factors(start, stop):
        a = -.5 * ((x[:, None] - samples[None, start:stop, 0]) / bandwidth[0])**2
        b = -.5 * ((y[:, None] - samples[None, start:stop, 1]) / bandwidth[1])**2
        a += half_weight[None, start:stop]
        b += half_weight[None, start:stop]
        return a, b

    # Each axis gets its own scale; a single global exponential is insufficient.
    sx, sy = np.full(len(x), -np.inf), np.full(len(y), -np.inf)
    for start in range(0, n, chunk):
        a, b = factors(start, min(n, start + chunk))
        sx = np.maximum(sx, a.max(axis=1))
        sy = np.maximum(sy, b.max(axis=1))

    mass = np.zeros((len(x), len(y)))
    second = np.zeros_like(mass)
    for start in range(0, n, chunk):
        a, b = factors(start, min(n, start + chunk))
        np.exp(a - sx[:, None], out=a)
        np.exp(b - sy[:, None], out=b)
        mass += a @ b.T
        a *= a
        b *= b
        second += a @ b.T

    scale = sx[:, None] + sy[None, :] + offset
    with np.errstate(divide='ignore'):
        lp = np.log(mass) + scale
        lp2 = np.log(second) + 2 * scale

    # A product can underflow even when its two axis factors are scaled well.
    # The second moment is the stricter test and also protects local ESS.
    ii, jj = np.where(second < 1e-180)
    for start in range(0, len(ii), 32):
        rows, cols = ii[start:start+32], jj[start:start+32]
        points = np.column_stack((x[rows], y[cols]))
        p, p2 = np.full(len(points), -np.inf), np.full(len(points), -np.inf)
        for frame in range(0, n, 2048):
            stop = min(n, frame + 2048)
            z = logweights[frame:stop, None] - .5 * np.sum(
                ((samples[frame:stop, None, :] - points[None, :, :]) / bandwidth)**2,
                axis=2,
            )
            p = np.logaddexp(p, logsumexp(z, axis=0))
            p2 = np.logaddexp(p2, logsumexp(2*z, axis=0))
        lp[rows, cols], lp2[rows, cols] = p, p2

    return lp, lp2, np.full(mass.shape, n, dtype=np.int64)
