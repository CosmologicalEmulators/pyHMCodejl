"""pyhmcode-jl: Python wrapper around HMcode.jl via juliacall.

Two-layer API:

* :func:`hmcode_power_tabulated` - low-level parity layer. Takes numpy
  tabulated ``Pk_lin[nz, nk]`` and ``sigma_R[nz, nR]``. This is the layer
  used for bit-comparable parity tests against native Julia.

* :func:`power` - high-level CAMB-driven layer mirroring the original
  Python ``hmcode.power(k, zs, CAMB_results, ...)`` signature.

Output is always shaped ``(nz, nk)`` to match the original Python convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import _bridge

__all__ = [
    "HMcodeCosmology",
    "hmcode_power_tabulated",
    "power",
]


@dataclass
class HMcodeCosmology:
    """Mirror of ``HMcode.HMcodeCosmology`` (9 Float64 fields)."""

    Omega_m: float
    Omega_b: float
    h: float
    n_s: float
    sigma_8: float
    w0: float = -1.0
    wa: float = 0.0
    Omega_nu: float = 0.0
    Omega_k: float = 0.0


def hmcode_power_tabulated(
    k,
    z,
    Pk_lin,
    sigma_R,
    R_grid,
    cosmo: HMcodeCosmology,
    *,
    T_AGN: float | None = 10 ** 7.8,
    Mmin: float = 1.0,
    Mmax: float = 1e18,
    nM: int = 256,
    threaded: bool = False,
    use_fast_specials: bool = True,
) -> np.ndarray:
    """Compute the HMcode2020 non-linear matter power spectrum.

    Parameters
    ----------
    k : array_like, shape (nk,)
        Comoving wavenumbers in h/Mpc.
    z : array_like, shape (nz,)
        Redshifts. The native Julia code expects them monotonically
        decreasing; we do not enforce that here so the user can match
        the native Julia behavior exactly.
    Pk_lin : array_like, shape (nz, nk)
        Linear matter power spectrum on the (z, k) grid, in (Mpc/h)^3.
        Strictly positive.
    sigma_R : array_like, shape (nz, nR)
        sigma(R, z) on the (z, R_grid) grid. Strictly positive.
    R_grid : array_like, shape (nR,)
        Lagrangian radii in Mpc/h.
    cosmo : HMcodeCosmology
    T_AGN : float or None, default 10**7.8
        AGN feedback temperature in K. ``None`` disables baryonic
        feedback (matches the ``T_AGN=None`` Julia default semantics).
    Mmin, Mmax, nM : halo mass grid (Msun/h, count).
    threaded, use_fast_specials : pass-through Julia kwargs.

    Returns
    -------
    Pk : ndarray, shape (nz, nk)
    """
    return _bridge.call_hmcode_power(
        k=k,
        z=z,
        Pk_table=Pk_lin,
        sigma_table=sigma_R,
        R_grid=R_grid,
        cosmo=asdict(cosmo),
        T_AGN=T_AGN,
        Mmin=Mmin,
        Mmax=Mmax,
        nM=nM,
        threaded=threaded,
        use_fast_specials=use_fast_specials,
    )


def power(
    k,
    zs,
    CAMB_results,
    *,
    T_AGN: float | None = None,
    Mmin: float = 1.0,
    Mmax: float = 1e18,
    nM: int = 256,
) -> np.ndarray:
    """High-level wrapper mirroring the original Python ``hmcode.power``.

    Pulls Pk_lin, cold sigma_R, and cosmology fields out of a
    ``camb.CAMBdata`` object, then dispatches to
    :func:`hmcode_power_tabulated`.

    Notes
    -----
    Uses ``var1='delta_nonu'`` for ``sigma_R`` to match the original
    Python HMcode (cold matter variance, not total).
    """
    import camb  # noqa: F401  - imported for type semantics; keeps optional dep optional

    k = np.ascontiguousarray(np.asarray(k, dtype=np.float64))
    zs = np.ascontiguousarray(np.asarray(zs, dtype=np.float64))

    # --- background cosmology at z=0 ---------------------------------
    Om_c = CAMB_results.get_Omega(var="cdm", z=0.0)
    Om_b = CAMB_results.get_Omega(var="baryon", z=0.0)
    Om_nu = CAMB_results.get_Omega(var="nu", z=0.0)
    params = CAMB_results.Params
    cosmo = HMcodeCosmology(
        Omega_m=Om_c + Om_b + Om_nu,
        Omega_b=Om_b,
        h=params.H0 / 100.0,
        n_s=params.InitPower.ns,
        sigma_8=float(CAMB_results.get_sigma8_0()),
        w0=getattr(params.DarkEnergy, "w", -1.0),
        wa=getattr(params.DarkEnergy, "wa", 0.0),
        Omega_nu=Om_nu,
        Omega_k=params.omk,
    )

    # --- linear power: (nz, nk) --------------------------------------
    Pk_interp, _, _ = CAMB_results.get_matter_power_interpolator(
        nonlinear=False, return_z_k=True, extrap_kmax=1e10
    )
    Pk_lin = np.empty((zs.size, k.size), dtype=np.float64)
    for iz, z in enumerate(zs):
        Pk_lin[iz, :] = Pk_interp.P(z, k)

    # --- sigma_R(z) on a fixed R grid: (nz, nR) ----------------------
    # Match the original Python: cold variance via var1='delta_nonu'.
    R_grid = np.logspace(-3.0, 2.0, 256)
    sigma_R = np.empty((zs.size, R_grid.size), dtype=np.float64)
    for iz in range(zs.size):
        sigma_R[iz, :] = CAMB_results.get_sigmaR(
            R_grid, z_indices=[iz], var1="delta_nonu", var2="delta_nonu"
        )[0]

    return hmcode_power_tabulated(
        k=k,
        z=zs,
        Pk_lin=Pk_lin,
        sigma_R=sigma_R,
        R_grid=R_grid,
        cosmo=cosmo,
        T_AGN=T_AGN,
        Mmin=Mmin,
        Mmax=Mmax,
        nM=nM,
        threaded=False,
        use_fast_specials=True,
    )
