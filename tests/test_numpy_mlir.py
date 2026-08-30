import numpy as np
import pytest

from numba_cuda_mlir import cuda

from lbm2d.backends.numba_cuda_mlir import CudaMLIRSolver
from lbm2d.backends.numpy_cpu import NumpyCPUSolver
from lbm2d.config import CavityConfig
from lbm2d.state import initial_cavity_state


pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA device unavailable")


@pytest.mark.parametrize("steps", [1, 5, 20])
def test_cpu_gpu_short_time_consistency(steps):
    cfg = CavityConfig(nx=17, ny=17, max_steps=20, minimum_steps=0,
                       check_interval=1, diagnostic_interval=1)
    cpu_state = initial_cavity_state(cfg.nx, cfg.ny, cfg.lid_velocity, np.float32)
    gpu_state = initial_cavity_state(cfg.nx, cfg.ny, cfg.lid_velocity, np.float32)
    cpu = NumpyCPUSolver(cpu_state, cfg.nu_tau_omega[2])
    gpu = CudaMLIRSolver(gpu_state, cfg.nu_tau_omega[2])
    for _ in range(steps):
        cpu.step()
        gpu.step()
    gpu.synchronize()
    np.testing.assert_allclose(gpu.copy_current_f(), cpu.copy_current_f(), rtol=2e-5, atol=2e-6)
    cr, cu = cpu.get_fields()
    gr, gu = gpu.get_fields()
    np.testing.assert_allclose(gr, cr, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gu, cu, rtol=2e-5, atol=2e-6)


def test_repeated_launch_is_stable():
    cfg = CavityConfig(nx=17, ny=17, max_steps=500, minimum_steps=0,
                       check_interval=1, diagnostic_interval=1)
    gpu = CudaMLIRSolver(initial_cavity_state(17, 17, cfg.lid_velocity, np.float32), cfg.nu_tau_omega[2])
    for _ in range(500):
        gpu.step()
    rho, u = gpu.get_fields()
    assert np.isfinite(rho).all() and np.isfinite(u).all()
    assert np.all(rho[1:-1, 1:-1] > 0)

