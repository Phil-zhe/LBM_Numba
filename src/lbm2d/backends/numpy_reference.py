from __future__ import annotations

import numpy as np

from ..lattice import CX, CY, OPPOSITE, equilibrium


def reference_step(f, solid, wall_velocity, omega):
    """Explicit readable closed/solid-wall step for small-grid oracles."""
    rho = np.ones(f.shape[1:], dtype=f.dtype)
    u = np.zeros((2, *f.shape[1:]), dtype=f.dtype)
    for y in range(f.shape[1]):
        for x in range(f.shape[2]):
            if solid[y, x]:
                continue
            rho[y, x] = np.sum(f[:, y, x])
            u[0, y, x] = np.dot(f[:, y, x], CX) / rho[y, x]
            u[1, y, x] = np.dot(f[:, y, x], CY) / rho[y, x]
    feq = equilibrium(rho, u, f.dtype)
    f_post = f - f.dtype.type(omega) * (f - feq)
    f_post[0] = rho - np.sum(f_post[1:], axis=0)
    f_post[:, solid] = 0
    out = np.zeros_like(f)
    for y in range(f.shape[1]):
        for x in range(f.shape[2]):
            if solid[y, x]:
                continue
            for q in range(9):
                ys, xs = y - CY[q], x - CX[q]
                if solid[ys, xs]:
                    uw_dot_e = CX[q] * wall_velocity[0, ys, xs] + CY[q] * wall_velocity[1, ys, xs]
                    out[q, y, x] = f_post[OPPOSITE[q], y, x] + 6 * f.dtype.type(1 / 9 if q < 5 and q else 4 / 9 if q == 0 else 1 / 36) * rho[y, x] * uw_dot_e
                else:
                    out[q, y, x] = f_post[q, ys, xs]
    return out
