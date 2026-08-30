import numpy as np

from lbm2d.boundaries import zou_he_left, zou_he_right
from lbm2d.lattice import CX, CY, equilibrium


def _moments(f):
    return f.sum(), f @ CX, f @ CY


def test_left_velocity_reconstruction_moments():
    rho, ux, uy = 1.02, 0.03, -0.004
    f = equilibrium(np.array([[rho]]), np.array([[[ux]], [[uy]]]))[:, 0, 0]
    f[[1, 5, 8]] = np.nan
    got_rho = zou_he_left(f, ux, uy)
    r, mx, my = _moments(f)
    np.testing.assert_allclose([got_rho, r, mx, my], [rho, rho, rho * ux, rho * uy], atol=1e-14)


def test_right_density_reconstruction_moments():
    rho, ux, uy = 0.99, 0.02, 0.003
    f = equilibrium(np.array([[rho]]), np.array([[[ux]], [[uy]]]))[:, 0, 0]
    f[[3, 6, 7]] = np.nan
    got_ux = zou_he_right(f, rho, uy)
    r, mx, my = _moments(f)
    np.testing.assert_allclose([got_ux, r, mx, my], [ux, rho, rho * ux, rho * uy], atol=1e-14)

