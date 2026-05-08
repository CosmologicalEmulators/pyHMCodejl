"""
juliacall bridge for HMcode.jl.

Loads the HMcode Julia package into a private module so we don't pollute
``Main``, then exposes thin Python helpers that build Julia interpolants
and dispatch ``hmcode_power``.

All heavy lifting stays Julia-side. Numpy arrays cross the boundary
exactly twice per call: once on the way in (Pk_table, sigma_table, k, z),
once on the way out (Pk_out as a (nk, nz) Julia matrix that we transpose
to (nz, nk) for the Python convention).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

_INIT_LOCK = threading.Lock()
_JL = None
_INTERP_LOADED = False


def _juliapkg_dir() -> str:
    return str(Path(__file__).resolve().parent)


def get_julia():
    """Return the lazily-initialised juliacall module handle.

    First call triggers Julia startup + package precompilation. We do this
    behind a lock so concurrent imports from threads don't double-init.
    """
    global _JL, _INTERP_LOADED
    if _JL is not None and _INTERP_LOADED:
        return _JL

    with _INIT_LOCK:
        if _JL is None:
            # Point JuliaPkg at our shipped juliapkg.json before importing juliacall.
            os.environ.setdefault(
                "PYTHON_JULIAPKG_PROJECT", _juliapkg_dir()
            )
            import juliacall  # noqa: WPS433 (intentional lazy import)

            jl = juliacall.newmodule("HMcodePy")
            jl.seval("using HMcode")
            jl.seval("using Interpolations")
            _JL = jl

        if not _INTERP_LOADED:
            interp_jl = Path(__file__).resolve().parent / "_interp.jl"
            _JL.seval(f'include("{interp_jl.as_posix()}")')
            _INTERP_LOADED = True

    return _JL


def _as_f64_contig(arr, name: str, ndim: int) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D array, got shape {a.shape}")
    return a


def call_hmcode_power(
    k: np.ndarray,
    z: np.ndarray,
    Pk_table: np.ndarray,
    sigma_table: np.ndarray,
    R_grid: np.ndarray,
    cosmo: dict,
    *,
    T_AGN: float | None,
    Mmin: float,
    Mmax: float,
    nM: int,
    threaded: bool,
    use_fast_specials: bool,
) -> np.ndarray:
    """Dispatch HMcode.hmcode_power with tabulated linear-power and sigma_R inputs.

    Shapes
    ------
    k : (nk,)
    z : (nz,)
    Pk_table : (nz, nk)         — Python convention
    sigma_table : (nz, nR)      — Python convention
    R_grid : (nR,)

    Returns
    -------
    Pk : (nz, nk)
    """
    jl = get_julia()

    k = _as_f64_contig(k, "k", 1)
    z = _as_f64_contig(z, "z", 1)
    Pk_table = _as_f64_contig(Pk_table, "Pk_table", 2)
    sigma_table = _as_f64_contig(sigma_table, "sigma_table", 2)
    R_grid = _as_f64_contig(R_grid, "R_grid", 1)

    nk, nz, nR = k.size, z.size, R_grid.size
    if Pk_table.shape != (nz, nk):
        raise ValueError(f"Pk_table shape {Pk_table.shape} != (nz, nk) = ({nz}, {nk})")
    if sigma_table.shape != (nz, nR):
        raise ValueError(f"sigma_table shape {sigma_table.shape} != (nz, nR) = ({nz}, {nR})")

    # Transpose Python (nz, nk) -> Julia (nk, nz) so the Julia helper sees a
    # column-major-friendly layout indexable as Pk[ik, iz]. juliacall hands
    # numpy arrays to Julia as PyArray, but transposing here keeps the
    # interpretation explicit and avoids surprises in build_*_interp.
    Pk_jl = np.ascontiguousarray(Pk_table.T)        # (nk, nz)
    sigma_jl = np.ascontiguousarray(sigma_table.T)  # (nR, nz)

    Pk_lin = jl.build_Pk_lin_interp(k, z, Pk_jl)
    sigma_R = jl.build_sigma_R_interp(R_grid, z, sigma_jl)

    cosmo_jl = jl.HMcode.HMcodeCosmology(
        float(cosmo["Omega_m"]),
        float(cosmo["Omega_b"]),
        float(cosmo["h"]),
        float(cosmo["n_s"]),
        float(cosmo["sigma_8"]),
        float(cosmo["w0"]),
        float(cosmo["wa"]),
        float(cosmo["Omega_nu"]),
        float(cosmo["Omega_k"]),
    )

    # T_AGN=None disables baryonic feedback.
    T_AGN_jl = jl.nothing if T_AGN is None else float(T_AGN)

    Pk_out = jl.HMcode.hmcode_power(
        k, z, Pk_lin, sigma_R, cosmo_jl,
        T_AGN=T_AGN_jl,
        Mmin=float(Mmin),
        Mmax=float(Mmax),
        nM=int(nM),
        threaded=bool(threaded),
        use_fast_specials=bool(use_fast_specials),
    )

    # Pk_out is a Julia Matrix{Float64} of shape (nk, nz). Convert to numpy
    # and transpose to the Python (nz, nk) convention.
    Pk_np = np.asarray(Pk_out, dtype=np.float64)
    if Pk_np.shape != (nk, nz):
        raise RuntimeError(
            f"unexpected Pk_out shape from Julia: {Pk_np.shape}, expected ({nk}, {nz})"
        )
    return np.ascontiguousarray(Pk_np.T)
