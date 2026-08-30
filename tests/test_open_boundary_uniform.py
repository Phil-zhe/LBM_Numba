import numpy as np
import pytest
from numba_cuda_mlir import cuda

from lbm2d.backends.numpy_cpu import NumpyCPUSolver
from lbm2d.backends.numba_cuda_mlir import CudaMLIRSolver
from lbm2d.lattice import equilibrium
from lbm2d.state import HostState


def _uniform_state(dtype=np.float32):
    ny, nx = 12, 24
    rho = np.ones((ny, nx), dtype=dtype)
    u = np.zeros((2, ny, nx), dtype=dtype)
    u[0] = dtype(0.03)
    f = equilibrium(rho, u, dtype)
    solid = np.zeros((ny, nx), dtype=np.bool_)
    return HostState(f, np.zeros_like(f), np.zeros_like(f), rho, u, solid,
                     np.zeros((2, ny, nx), dtype=dtype))


def test_uniform_open_boundary_cpu():
    solver = NumpyCPUSolver(_uniform_state(), 1.2, boundary="open",
                            inlet_velocity=0.03, outlet_density=1.0)
    for _ in range(20): solver.step()
    rho, u = solver.get_fields()
    np.testing.assert_allclose(rho, 1, atol=2e-6)
    np.testing.assert_allclose(u[0], 0.03, atol=2e-6)
    np.testing.assert_allclose(u[1], 0, atol=2e-6)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA unavailable")
def test_uniform_open_boundary_gpu_matches_cpu():
    cpu = NumpyCPUSolver(_uniform_state(), 1.2, boundary="open",
                         inlet_velocity=0.03, outlet_density=1.0)
    gpu = CudaMLIRSolver(_uniform_state(), 1.2, boundary="open",
                         inlet_velocity=0.03, outlet_density=1.0)
    for _ in range(20):
        cpu.step(); gpu.step()
    np.testing.assert_allclose(gpu.copy_current_f(), cpu.copy_current_f(), rtol=2e-5, atol=2e-6)

