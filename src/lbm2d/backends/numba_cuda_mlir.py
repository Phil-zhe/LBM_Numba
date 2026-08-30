from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba_cuda_mlir import cuda

from ..lattice import CX, CY, OPPOSITE, WEIGHTS
from ..state import HostState
from . import device_adapter as device


@cuda.jit
def macroscopic_collision_kernel(f, f_post, rho, u, solid, cx, cy, weights, omega, nx, ny):
    x, y = cuda.grid(2)
    if x >= nx or y >= ny:
        return
    if solid[y, x]:
        rho[y, x] = 1.0
        u[0, y, x] = 0.0
        u[1, y, x] = 0.0
        for q in range(9):
            f_post[q, y, x] = 0.0
        return
    moving_mass = f[1, y, x]
    for q in range(2, 9):
        moving_mass += f[q, y, x]
    r = moving_mass + f[0, y, x]
    mx = 0.0
    my = 0.0
    for q in range(9):
        fq = f[q, y, x]
        mx += fq * cx[q]
        my += fq * cy[q]
    ux = mx / r
    uy = my / r
    rho[y, x] = r
    u[0, y, x] = ux
    u[1, y, x] = uy
    u2 = ux * ux + uy * uy
    for q in range(9):
        eu = cx[q] * ux + cy[q] * uy
        feq = weights[q] * r * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u2)
        f_post[q, y, x] = f[q, y, x] - omega * (f[q, y, x] - feq)
    # Put only storage-rounding closure into q=0; momentum is unchanged.
    moving_sum = f_post[1, y, x]
    for q in range(2, 9):
        moving_sum += f_post[q, y, x]
    f_post[0, y, x] = r - moving_sum


@cuda.jit
def streaming_bounce_kernel(f_post, f_next, rho, solid, wall_velocity,
                            cx, cy, weights, opposite, source_x, source_y,
                            nx, ny):
    x, y = cuda.grid(2)
    if x >= nx or y >= ny:
        return
    if solid[y, x]:
        for q in range(9):
            f_next[q, y, x] = 0.0
        return
    for q in range(9):
        # Precomputed signed indices avoid unsigned cuda.grid arithmetic.
        xs = source_x[q, x]
        ys = source_y[q, y]
        if xs < 0 or xs >= nx or ys < 0 or ys >= ny:
            continue
        if solid[ys, xs]:
            wall_dot = cx[q] * wall_velocity[0, ys, xs] + cy[q] * wall_velocity[1, ys, xs]
            f_next[q, y, x] = (
                f_post[opposite[q], y, x]
                + 6.0 * weights[q] * rho[y, x] * wall_dot
            )
        else:
            f_next[q, y, x] = f_post[q, ys, xs]


@cuda.jit
def left_zou_he_kernel(f_next, inlet_velocity, nx, ny):
    y = cuda.grid(1)
    if y >= ny:
        return
    ux = inlet_velocity
    uy = 0.0
    rho = (f_next[0, y, 0] + f_next[2, y, 0] + f_next[4, y, 0]
           + 2.0 * (f_next[3, y, 0] + f_next[6, y, 0] + f_next[7, y, 0])) / (1.0 - ux)
    f_next[1, y, 0] = f_next[3, y, 0] + (2.0 / 3.0) * rho * ux
    f_next[5, y, 0] = (f_next[7, y, 0] + 0.5 * (f_next[4, y, 0] - f_next[2, y, 0])
                         + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy)
    f_next[8, y, 0] = (f_next[6, y, 0] + 0.5 * (f_next[2, y, 0] - f_next[4, y, 0])
                         + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy)


@cuda.jit
def right_zou_he_kernel(f_next, outlet_density, nx, ny):
    y = cuda.grid(1)
    if y >= ny:
        return
    r_inner = 0.0
    my_inner = 0.0
    for q in range(9):
        fq = f_next[q, y, nx - 2]
        r_inner += fq
        if q == 2 or q == 5 or q == 6:
            my_inner += fq
        elif q == 4 or q == 7 or q == 8:
            my_inner -= fq
    uy = my_inner / r_inner
    rho = outlet_density
    ux = -1.0 + (f_next[0, y, nx - 1] + f_next[2, y, nx - 1] + f_next[4, y, nx - 1]
                 + 2.0 * (f_next[1, y, nx - 1] + f_next[5, y, nx - 1]
                          + f_next[8, y, nx - 1])) / rho
    f_next[3, y, nx - 1] = f_next[1, y, nx - 1] - (2.0 / 3.0) * rho * ux
    f_next[6, y, nx - 1] = (f_next[8, y, nx - 1]
                              + 0.5 * (f_next[4, y, nx - 1] - f_next[2, y, nx - 1])
                              - (1.0 / 6.0) * rho * ux + 0.5 * rho * uy)
    f_next[7, y, nx - 1] = (f_next[5, y, nx - 1]
                              + 0.5 * (f_next[2, y, nx - 1] - f_next[4, y, nx - 1])
                              - (1.0 / 6.0) * rho * ux - 0.5 * rho * uy)


@cuda.jit
def macroscopic_kernel(f, rho, u, solid, cx, cy, nx, ny):
    x, y = cuda.grid(2)
    if x >= nx or y >= ny:
        return
    if solid[y, x]:
        rho[y, x] = 1.0
        u[0, y, x] = 0.0
        u[1, y, x] = 0.0
        return
    moving_mass = f[1, y, x]
    for q in range(2, 9):
        moving_mass += f[q, y, x]
    r = moving_mass + f[0, y, x]
    mx = 0.0
    my = 0.0
    for q in range(9):
        fq = f[q, y, x]
        mx += fq * cx[q]
        my += fq * cy[q]
    rho[y, x] = r
    u[0, y, x] = mx / r
    u[1, y, x] = my / r


@dataclass
class GPUFields:
    f: object
    f_post: object
    f_next: object
    rho: object
    u: object
    solid: object
    wall_velocity: object
    cx: object
    cy: object
    weights: object
    opposite: object
    source_x: object
    source_y: object


class CudaMLIRSolver:
    backend = "numba-cuda-mlir"
    block = (16, 16)

    def __init__(self, state: HostState, omega: float, boundary="closed",
                 inlet_velocity=0.0, outlet_density=1.0):
        if not cuda.is_available():
            raise RuntimeError("requested backend numba-cuda-mlir is unavailable")
        self.host = state
        self.ny, self.nx = state.solid.shape
        self.omega = state.f.dtype.type(omega)
        self.boundary = boundary
        self.inlet_velocity = state.f.dtype.type(inlet_velocity)
        self.outlet_density = state.f.dtype.type(outlet_density)
        source_x = np.empty((9, self.nx), dtype=np.int32)
        source_y = np.empty((9, self.ny), dtype=np.int32)
        for q in range(9):
            source_x[q] = np.arange(self.nx, dtype=np.int32) - CX[q]
            source_y[q] = np.arange(self.ny, dtype=np.int32) - CY[q]
            if boundary == "open":
                source_y[q] %= self.ny
        self.fields = GPUFields(
            device.to_device(state.f), device.to_device(state.f_post),
            device.to_device(state.f_next), device.to_device(state.rho),
            device.to_device(state.u), device.to_device(state.solid),
            device.to_device(state.wall_velocity),
            device.to_device(CX), device.to_device(CY),
            device.to_device(WEIGHTS.astype(state.f.dtype)), device.to_device(OPPOSITE),
            device.to_device(source_x), device.to_device(source_y),
        )
        self.grid = ((self.nx + 15) // 16, (self.ny + 15) // 16)
        self.current_step = state.current_step
        self.macroscopic_valid = state.macroscopic_valid

    def step(self):
        d = self.fields
        macroscopic_collision_kernel[self.grid, self.block](
            d.f, d.f_post, d.rho, d.u, d.solid, d.cx, d.cy, d.weights,
            self.omega, self.nx, self.ny,
        )
        streaming_bounce_kernel[self.grid, self.block](
            d.f_post, d.f_next, d.rho, d.solid, d.wall_velocity,
            d.cx, d.cy, d.weights, d.opposite, d.source_x, d.source_y,
            self.nx, self.ny,
        )
        if self.boundary == "open":
            blocks = (self.ny + 255) // 256
            left_zou_he_kernel[blocks, 256](d.f_next, self.inlet_velocity, self.nx, self.ny)
            right_zou_he_kernel[blocks, 256](d.f_next, self.outlet_density, self.nx, self.ny)
        d.f, d.f_next = d.f_next, d.f
        self.current_step += 1
        self.macroscopic_valid = False

    def refresh_macroscopic(self):
        d = self.fields
        macroscopic_kernel[self.grid, self.block](d.f, d.rho, d.u, d.solid, d.cx, d.cy, self.nx, self.ny)
        self.macroscopic_valid = True

    def get_fields(self):
        if not self.macroscopic_valid:
            self.refresh_macroscopic()
        device.synchronize()
        device.copy_to_host(self.fields.rho, self.host.rho)
        device.copy_to_host(self.fields.u, self.host.u)
        self.host.current_step = self.current_step
        self.host.macroscopic_valid = True
        return self.host.rho, self.host.u

    def copy_current_f(self):
        device.synchronize()
        return device.copy_to_host(self.fields.f)

    def restore_f(self, checkpoint):
        device.copy_to_device(self.fields.f, checkpoint)
        self.macroscopic_valid = False

    def synchronize(self):
        device.synchronize()
