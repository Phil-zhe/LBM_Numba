from __future__ import annotations

import numpy as np

from .lattice import CS


def field_diagnostics(rho, u, solid, initial_mass):
    fluid = ~solid
    r = rho[fluid].astype(np.float64)
    ux = u[0, fluid].astype(np.float64)
    uy = u[1, fluid].astype(np.float64)
    speed = np.sqrt(ux * ux + uy * uy)
    mass = float(r.sum())
    return {
        "min_rho": float(r.min()),
        "max_rho": float(r.max()),
        "rho_deviation": float(np.max(np.abs(r - 1.0))),
        "max_speed": float(speed.max()),
        "ma_max": float(speed.max() / CS),
        "mass": mass,
        "mass_drift": abs(mass - initial_mass) / initial_mass,
        "nonfinite_count": int(np.size(r) - np.count_nonzero(np.isfinite(r))
                               + np.size(ux) - np.count_nonzero(np.isfinite(ux))
                               + np.size(uy) - np.count_nonzero(np.isfinite(uy))),
    }


def velocity_residual(u, previous, solid):
    fluid = ~solid
    current = u[:, fluid].astype(np.float64)
    old = previous[:, fluid].astype(np.float64)
    denominator = max(float(np.linalg.norm(current)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(current - old) / denominator)

