from __future__ import annotations

import numpy as np

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
WEIGHTS = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
                    1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
CS2 = 1.0 / 3.0
CS = 1.0 / np.sqrt(3.0)


def equilibrium(rho: np.ndarray, u: np.ndarray, dtype=None) -> np.ndarray:
    """Readable full-field equilibrium distribution, shape ``(9, ny, nx)``."""
    dtype = np.dtype(dtype or rho.dtype)
    out = np.empty((9, *rho.shape), dtype=dtype)
    u2 = u[0] * u[0] + u[1] * u[1]
    for q in range(9):
        eu = CX[q] * u[0] + CY[q] * u[1]
        out[q] = dtype.type(WEIGHTS[q]) * rho * (
            dtype.type(1.0) + dtype.type(3.0) * eu
            + dtype.type(4.5) * eu * eu - dtype.type(1.5) * u2
        )
    return out


def viscosity_parameters(re: float, reference_velocity: float, length: float):
    if re <= 0 or reference_velocity <= 0 or length <= 0:
        raise ValueError("re, reference_velocity and length must be positive")
    nu = reference_velocity * length / re
    tau = 0.5 + 3.0 * nu
    omega = 1.0 / tau
    if not tau > 0.5 or not 0.0 < omega < 2.0:
        raise ValueError("unstable BGK parameters")
    return nu, tau, omega

