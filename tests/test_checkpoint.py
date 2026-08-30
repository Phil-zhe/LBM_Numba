import numpy as np

from lbm2d.backends.numpy_cpu import NumpyCPUSolver
from lbm2d.checkpoint import load_checkpoint, save_checkpoint
from lbm2d.config import CavityConfig
from lbm2d.state import initial_cavity_state


def test_checkpoint_resume_matches_continuous(tmp_path):
    cfg = CavityConfig(nx=17, ny=17, max_steps=10, minimum_steps=0,
                       check_interval=1, diagnostic_interval=1)
    def make():
        return NumpyCPUSolver(initial_cavity_state(17, 17, 0.05, np.float64), cfg.nu_tau_omega[2])
    continuous = make()
    for _ in range(10): continuous.step()
    first = make()
    for _ in range(4): first.step()
    path = tmp_path / "state.npz"
    save_checkpoint(path, first, cfg.to_dict(), first.backend, {"consecutive": 2})
    resumed = make()
    state = load_checkpoint(path, resumed, cfg.to_dict(), resumed.backend)
    assert state == {"consecutive": 2}
    for _ in range(6): resumed.step()
    assert resumed.state.current_step == 10
    np.testing.assert_allclose(resumed.copy_current_f(), continuous.copy_current_f(), rtol=0, atol=0)

