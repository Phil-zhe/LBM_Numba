import numpy as np

from lbm2d.backends.numpy_cpu import NumpyCPUSolver
from lbm2d.backends.numpy_reference import reference_step
from lbm2d.config import CavityConfig
from lbm2d.state import initial_cavity_state


def test_cpu_matches_reference_moving_lid():
    cfg = CavityConfig(nx=17, ny=17, max_steps=10, minimum_steps=0,
                       check_interval=1, diagnostic_interval=1)
    _, _, omega = cfg.nu_tau_omega
    state = initial_cavity_state(cfg.nx, cfg.ny, cfg.lid_velocity, np.float64)
    expected = reference_step(state.f.copy(), state.solid, state.wall_velocity, omega)
    solver = NumpyCPUSolver(state, omega)
    solver.step()
    np.testing.assert_allclose(state.f, expected, rtol=1e-13, atol=1e-14)
    assert not state.macroscopic_valid
    _, u = solver.get_fields()
    assert state.macroscopic_valid
    assert np.mean(u[0, -2, 1:-1]) > 0
    assert np.all(state.f[:, state.solid] == 0)


def test_mass_is_conserved_in_closed_cavity():
    cfg = CavityConfig(nx=33, ny=33, max_steps=20, minimum_steps=0,
                       check_interval=1, diagnostic_interval=1)
    state = initial_cavity_state(cfg.nx, cfg.ny, cfg.lid_velocity, np.float32)
    solver = NumpyCPUSolver(state, cfg.nu_tau_omega[2])
    m0 = float(state.rho[~state.solid].sum(dtype=np.float64))
    for _ in range(20):
        solver.step()
    rho, _ = solver.get_fields()
    m1 = float(rho[~state.solid].sum(dtype=np.float64))
    assert abs(m1 - m0) / m0 < 1e-5

