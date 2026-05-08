"""Smoke test: just verify the wrapper imports and a trivial call returns the right shape.

This does NOT validate numerical parity yet — that comes in
test_parity_julia.py once the Julia reference fixture is wired up.
"""

import numpy as np
import pytest

import hmcode_py


def _build_dummy_inputs(nk=16, nz=3, nR=64):
    k = np.logspace(-3, 1, nk)
    z = np.array([2.0, 1.0, 0.0])[:nz]
    # smooth, positive proxy spectra
    Pk_lin = np.empty((nz, nk))
    for iz, _z in enumerate(z):
        D = 1.0 / (1.0 + _z)
        Pk_lin[iz] = (k ** 0.965) * np.exp(-0.18 * k) * D ** 2 + 1e-12
    R_grid = np.logspace(-3, 2, nR)
    sigma_R = np.empty((nz, nR))
    for iz, _z in enumerate(z):
        D = 1.0 / (1.0 + _z)
        sigma_R[iz] = 2.8 * D ** 0.9 * (R_grid ** -0.32) / (1.0 + (R_grid / 9.0) ** 1.8)
    return k, z, Pk_lin, sigma_R, R_grid


@pytest.mark.parametrize("T_AGN", [None, 10 ** 7.8])
def test_shape_and_finiteness(T_AGN):
    k, z, Pk_lin, sigma_R, R_grid = _build_dummy_inputs()
    cosmo = hmcode_py.HMcodeCosmology(
        Omega_m=0.30, Omega_b=0.05, h=0.7, n_s=0.96, sigma_8=0.8,
    )
    Pk = hmcode_py.hmcode_power_tabulated(
        k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
        cosmo=cosmo, T_AGN=T_AGN, nM=64,
    )
    assert Pk.shape == (z.size, k.size)
    assert np.all(np.isfinite(Pk))
    assert np.all(Pk > 0)
