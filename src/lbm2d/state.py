from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .lattice import equilibrium


@dataclass
class HostState:
    f: np.ndarray
    f_post: np.ndarray
    f_next: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    solid: np.ndarray
    wall_velocity: np.ndarray
    current_step: int = 0
    macroscopic_valid: bool = True


def cavity_geometry(nx: int, ny: int, lid_velocity: float, dtype):
    solid = np.zeros((ny, nx), dtype=np.bool_)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True
    wall_velocity = np.zeros((2, ny, nx), dtype=dtype)
    wall_velocity[0, -1, 1:-1] = dtype(lid_velocity)
    return solid, wall_velocity


def initial_cavity_state(nx: int, ny: int, lid_velocity: float, dtype) -> HostState:
    dtype = np.dtype(dtype)
    solid, wall_velocity = cavity_geometry(nx, ny, lid_velocity, dtype.type)
    rho = np.ones((ny, nx), dtype=dtype)
    u = np.zeros((2, ny, nx), dtype=dtype)
    f = equilibrium(rho, u, dtype)
    f[:, solid] = 0
    f_post = np.zeros_like(f)
    f_next = np.zeros_like(f)
    return HostState(f, f_post, f_next, rho, u, solid, wall_velocity)


def cylinder_geometry(nx, ny, center_x, center_y, diameter):
    y, x = np.mgrid[0:ny, 0:nx]
    dy = (y - center_y + ny / 2) % ny - ny / 2
    return (x - center_x) ** 2 + dy ** 2 <= (diameter / 2) ** 2


def initial_cylinder_state(config, dtype) -> HostState:
    dtype = np.dtype(dtype)
    solid = cylinder_geometry(config.nx, config.ny, config.cylinder_center_x,
                              config.cylinder_center_y, config.cylinder_diameter)
    wall_velocity = np.zeros((2, config.ny, config.nx), dtype=dtype)
    rho = np.ones((config.ny, config.nx), dtype=dtype)
    u = np.zeros((2, config.ny, config.nx), dtype=dtype)
    u[0, ~solid] = dtype.type(config.inlet_velocity)
    y = np.arange(config.ny, dtype=np.float64)[:, None]
    x = np.arange(config.nx, dtype=np.float64)[None, :]
    raw = (config.initial_perturbation_amplitude * config.inlet_velocity
           * np.sin(2 * np.pi * (y - config.cylinder_center_y) / config.ny
                    + config.initial_perturbation_phase)
           * np.exp(-((x - (config.cylinder_center_x + config.cylinder_diameter))
                      / config.cylinder_diameter) ** 2))
    for ix in range(config.nx):
        fluid_y = ~solid[:, ix]
        values = raw[fluid_y, ix]
        u[1, fluid_y, ix] = (values - values.mean()).astype(dtype)
    f = equilibrium(rho, u, dtype)
    f[:, solid] = 0
    rho[solid] = 1
    u[:, solid] = 0
    return HostState(f, np.zeros_like(f), np.zeros_like(f), rho, u, solid, wall_velocity)
