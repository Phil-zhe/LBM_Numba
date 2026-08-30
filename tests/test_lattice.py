import numpy as np

from lbm2d.lattice import CS2, CX, CY, OPPOSITE, WEIGHTS, equilibrium


def test_d2q9_invariants():
    assert np.isclose(WEIGHTS.sum(), 1)
    assert np.isclose(WEIGHTS @ CX, 0)
    assert np.isclose(WEIGHTS @ CY, 0)
    tensor = np.array([[np.sum(WEIGHTS * CX * CX), np.sum(WEIGHTS * CX * CY)],
                       [np.sum(WEIGHTS * CY * CX), np.sum(WEIGHTS * CY * CY)]])
    np.testing.assert_allclose(tensor, np.eye(2) * CS2)
    np.testing.assert_array_equal(OPPOSITE[OPPOSITE], np.arange(9))


def test_equilibrium_moments():
    rng = np.random.default_rng(4)
    rho = 1 + rng.uniform(-1e-3, 1e-3, (5, 7))
    u = rng.uniform(-1e-3, 1e-3, (2, 5, 7))
    f = equilibrium(rho, u)
    np.testing.assert_allclose(f.sum(axis=0), rho, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.tensordot(CX, f, axes=(0, 0)), rho * u[0], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(np.tensordot(CY, f, axes=(0, 0)), rho * u[1], rtol=1e-13, atol=1e-14)

