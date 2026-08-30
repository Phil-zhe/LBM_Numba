from __future__ import annotations

import csv
from pathlib import Path
import numpy as np


def _load_reference(path: Path, coordinate: str, value: str):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (np.array([float(r[coordinate]) for r in rows]),
            np.array([float(r[value]) for r in rows]))


def physical_fluid_coordinates(n: int):
    return (np.arange(1, n - 1, dtype=np.float64) - 0.5) / (n - 2)


def centerlines(rho, u, lid_velocity):
    del rho
    ny, nx = u.shape[1:]
    xp = physical_fluid_coordinates(nx)
    yp = physical_fluid_coordinates(ny)
    # Interpolate between numerical lines when physical 0.5 is not a node.
    ux_vertical = np.array([np.interp(0.5, xp, u[0, y, 1:-1]) for y in range(1, ny - 1)])
    uy_horizontal = np.array([np.interp(0.5, yp, u[1, 1:-1, x]) for x in range(1, nx - 1)])
    y_out = np.concatenate(([0.0], yp, [1.0]))
    u_out = np.concatenate(([0.0], ux_vertical / lid_velocity, [1.0]))
    x_out = np.concatenate(([0.0], xp, [1.0]))
    v_out = np.concatenate(([0.0], uy_horizontal / lid_velocity, [0.0]))
    return x_out, v_out, y_out, u_out


def validate_against_ghia(rho, u, lid_velocity, reference_dir):
    reference_dir = Path(reference_dir)
    gx, gv = _load_reference(reference_dir / "ghia_re100_v.csv", "x", "v_over_U_lid")
    gy, gu = _load_reference(reference_dir / "ghia_re100_u.csv", "y", "u_over_U_lid")
    x, v, y, uu = centerlines(rho, u, lid_velocity)
    vn = np.interp(gx, x, v)
    un = np.interp(gy, y, uu)
    eu, ev = un - gu, vn - gv
    return {
        "ghia_u_rmse": float(np.sqrt(np.mean(eu * eu))),
        "ghia_v_rmse": float(np.sqrt(np.mean(ev * ev))),
        "ghia_u_max_error": float(np.max(np.abs(eu))),
        "ghia_v_max_error": float(np.max(np.abs(ev))),
    }, (x, v, y, uu, gx, gv, gy, gu)

