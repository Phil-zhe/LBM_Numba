from __future__ import annotations

import numpy as np

from ..lattice import CX, CY, OPPOSITE, WEIGHTS
from ..state import HostState


class NumpyCPUSolver:
    backend = "numpy-cpu"

    def __init__(self, state: HostState, omega: float, boundary="closed",
                 inlet_velocity=0.0, outlet_density=1.0):
        self.state = state
        self.dtype = state.f.dtype
        self.omega = self.dtype.type(omega)
        self.boundary = boundary
        self.inlet_velocity = self.dtype.type(inlet_velocity)
        self.outlet_density = self.dtype.type(outlet_density)
        ny, nx = state.solid.shape
        self._u2 = np.empty((ny, nx), dtype=self.dtype)
        self._eu = np.empty_like(self._u2)
        self._tmp = np.empty_like(self._u2)
        self._shifted = np.empty_like(self._u2)
        self._xshifted = np.empty_like(self._u2)
        self._fluid = ~state.solid
        self._stream_mask = np.zeros((9, ny, nx), dtype=np.bool_)
        self._bounce_mask = np.zeros_like(self._stream_mask)
        self._wall_coeff = np.zeros((9, ny, nx), dtype=self.dtype)
        self._prepare_links()

    def _slices(self, c, n):
        if c > 0:
            return slice(c, n), slice(0, n - c)
        if c < 0:
            return slice(0, n + c), slice(-c, n)
        return slice(0, n), slice(0, n)

    def _prepare_links(self):
        ny, nx = self.state.solid.shape
        for q in range(9):
            if self.boundary == "open":
                yd, ys = slice(0, ny), slice(0, ny)
            else:
                yd, ys = self._slices(int(CY[q]), ny)
            xd, xs = self._slices(int(CX[q]), nx)
            dest_fluid = self._fluid[yd, xd]
            if self.boundary == "open":
                source_y = (np.arange(ny) - CY[q]) % ny
                source_solid = self.state.solid[source_y, xs]
            else:
                source_solid = self.state.solid[ys, xs]
            self._stream_mask[q, yd, xd] = dest_fluid & ~source_solid
            self._bounce_mask[q, yd, xd] = dest_fluid & source_solid
            if self.boundary == "open":
                coeff = 6.0 * WEIGHTS[q] * (
                    CX[q] * self.state.wall_velocity[0, source_y, xs]
                    + CY[q] * self.state.wall_velocity[1, source_y, xs])
            else:
                coeff = 6.0 * WEIGHTS[q] * (
                    CX[q] * self.state.wall_velocity[0, ys, xs]
                    + CY[q] * self.state.wall_velocity[1, ys, xs])
            self._wall_coeff[q, yd, xd] = coeff

    def refresh_macroscopic(self):
        s = self.state
        np.copyto(s.rho, s.f[1])
        for q in range(2, 9):
            s.rho += s.f[q]
        s.rho += s.f[0]
        np.multiply(s.f[1] - s.f[3] + s.f[5] - s.f[6] - s.f[7] + s.f[8], 1.0, out=s.u[0])
        np.add(s.f[2] - s.f[4], s.f[5] + s.f[6] - s.f[7] - s.f[8], out=s.u[1])
        np.divide(s.u[0], s.rho, out=s.u[0], where=self._fluid)
        np.divide(s.u[1], s.rho, out=s.u[1], where=self._fluid)
        s.rho[s.solid] = 1
        s.u[:, s.solid] = 0
        s.macroscopic_valid = True

    def _collide(self):
        s = self.state
        self.refresh_macroscopic()
        np.multiply(s.u[0], s.u[0], out=self._u2)
        np.multiply(s.u[1], s.u[1], out=self._tmp)
        np.add(self._u2, self._tmp, out=self._u2)
        for q in range(9):
            np.multiply(s.u[0], CX[q], out=self._eu)
            if CY[q]:
                np.multiply(s.u[1], CY[q], out=self._tmp)
                np.add(self._eu, self._tmp, out=self._eu)
            np.multiply(self._eu, self._eu, out=self._tmp)
            self._tmp *= self.dtype.type(4.5)
            self._tmp += self.dtype.type(1.0)
            self._tmp += self.dtype.type(3.0) * self._eu
            self._tmp -= self.dtype.type(1.5) * self._u2
            self._tmp *= self.dtype.type(WEIGHTS[q])
            self._tmp *= s.rho
            np.subtract(self._tmp, s.f[q], out=s.f_post[q])
            s.f_post[q] *= self.omega
            s.f_post[q] += s.f[q]
            s.f_post[q, s.solid] = 0
        # Close storage-rounding error in the rest population.  This preserves
        # exact stored mass without changing momentum or the BGK continuum rule.
        np.copyto(s.f_post[0], s.f_post[1])
        for q in range(2, 9):
            s.f_post[0] += s.f_post[q]
        np.subtract(s.rho, s.f_post[0], out=s.f_post[0])
        s.f_post[0, s.solid] = 0

    def _stream(self):
        s = self.state
        ny, nx = s.solid.shape
        s.f_next.fill(0)
        for q in range(9):
            xd, xs = self._slices(int(CX[q]), nx)
            self._shifted.fill(0)
            if self.boundary == "open":
                self._xshifted.fill(0)
                self._xshifted[:, xd] = s.f_post[q, :, xs]
                if CY[q] == 1:
                    self._shifted[1:] = self._xshifted[:-1]
                    self._shifted[0] = self._xshifted[-1]
                elif CY[q] == -1:
                    self._shifted[:-1] = self._xshifted[1:]
                    self._shifted[-1] = self._xshifted[0]
                else:
                    self._shifted[:] = self._xshifted
            else:
                yd, ys = self._slices(int(CY[q]), ny)
                self._shifted[yd, xd] = s.f_post[q, ys, xs]
            np.copyto(s.f_next[q], self._shifted, where=self._stream_mask[q])
            np.multiply(s.rho, self._wall_coeff[q], out=self._tmp)
            self._tmp += s.f_post[OPPOSITE[q]]
            np.copyto(s.f_next[q], self._tmp, where=self._bounce_mask[q])
        if self.boundary == "open":
            from ..boundaries import zou_he_left, zou_he_right
            zou_he_left(s.f_next[:, :, 0], self.inlet_velocity, 0.0)
            inner = s.f_next[:, :, -2]
            inner_rho = np.sum(inner, axis=0)
            inner_uy = (inner[2] - inner[4] + inner[5] + inner[6] - inner[7] - inner[8]) / inner_rho
            zou_he_right(s.f_next[:, :, -1], self.outlet_density, inner_uy)

    def step(self):
        self._collide()
        self._stream()
        self.state.f, self.state.f_next = self.state.f_next, self.state.f
        self.state.current_step += 1
        self.state.macroscopic_valid = False

    def get_fields(self):
        if not self.state.macroscopic_valid:
            self.refresh_macroscopic()
        return self.state.rho, self.state.u

    def copy_current_f(self):
        return self.state.f.copy()

    def restore_f(self, checkpoint):
        np.copyto(self.state.f, checkpoint)
        self.state.macroscopic_valid = False
