import numpy as np

from lbm2d.config import CylinderConfig
from lbm2d.state import initial_cylinder_state


def test_default_cylinder_geometry_and_perturbation():
    cfg = CylinderConfig()
    assert (cfg.nx - 1) / cfg.cylinder_diameter == 20
    assert cfg.cylinder_center_x / cfg.cylinder_diameter == 5
    assert (cfg.nx - 1 - cfg.cylinder_center_x) / cfg.cylinder_diameter == 15
    assert cfg.ny / cfg.cylinder_diameter == 8
    assert (cfg.probe_x, cfg.probe_y) == (360, 180)
    state = initial_cylinder_state(cfg, np.float32)
    for x in range(cfg.nx):
        fluid = ~state.solid[:, x]
        assert abs(float(state.u[1, fluid, x].mean(dtype=np.float64))) < 1e-10
    assert np.all(state.f[:, state.solid] == 0)

