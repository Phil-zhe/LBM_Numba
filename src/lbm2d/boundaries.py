from __future__ import annotations

import numpy as np


def zou_he_left(distributions, ux, uy=0.0):
    """Reconstruct left velocity boundary in-place for D2Q9 arrays."""
    f = distributions
    rho = (f[0] + f[2] + f[4] + 2 * (f[3] + f[6] + f[7])) / (1 - ux)
    f[1] = f[3] + (2 / 3) * rho * ux
    f[5] = f[7] + 0.5 * (f[4] - f[2]) + (1 / 6) * rho * ux + 0.5 * rho * uy
    f[8] = f[6] + 0.5 * (f[2] - f[4]) + (1 / 6) * rho * ux - 0.5 * rho * uy
    return rho


def zou_he_right(distributions, rho, uy):
    """Reconstruct right density boundary in-place for D2Q9 arrays."""
    f = distributions
    ux = -1 + (f[0] + f[2] + f[4] + 2 * (f[1] + f[5] + f[8])) / rho
    f[3] = f[1] - (2 / 3) * rho * ux
    f[6] = f[8] + 0.5 * (f[4] - f[2]) - (1 / 6) * rho * ux + 0.5 * rho * uy
    f[7] = f[5] + 0.5 * (f[2] - f[4]) - (1 / 6) * rho * ux - 0.5 * rho * uy
    return ux

