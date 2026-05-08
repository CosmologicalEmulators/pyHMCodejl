"""Parity test: wrapper output must match a native-Julia reference at machine precision.

The reference is computed inside the *same* juliacall session by:
  1. Building Pk_lin(k,z) and sigma_R(R,z) closures Julia-side from the
     same tabulated data, but via an *independent* code path (not through
     ``_interp.jl``). Plain log-log linear interpolation with nearest-z.
  2. Calling ``HMcode.hmcode_power`` directly Julia-side with those closures.

Then we call ``hmcode_py.hmcode_power_tabulated`` from Python with the
same numpy arrays and assert near-bit-equality.

Two wrapper-only failure modes this catches:
  - shape/transpose mistakes when crossing numpy ↔ Julia
  - kwarg / cosmology-struct field mismatches
"""

from __future__ import annotations

import numpy as np
import pytest

import hmcode_py
from hmcode_py import _bridge


# ---------------------------------------------------------------------------
# Fixture: deterministic, smooth, positive (k, z, Pk_lin, sigma_R, R_grid).
# Matches the spirit of HMcode.jl/benchmark/common_setup.jl's regression case.
# ---------------------------------------------------------------------------

def _make_inputs(nk=64, nz=5, nR=128):
    k = np.logspace(-3.0, 1.0, nk)
    z = np.linspace(0.0, 3.0, nz)[::-1].copy()  # decreasing, as HMcode expects
    R_grid = np.logspace(-3.0, 2.0, nR)

    Pk_lin = np.empty((nz, nk))
    sigma_R = np.empty((nz, nR))
    for iz, _z in enumerate(z):
        D = 1.0 / (1.0 + _z)
        Pk_lin[iz] = (
            (k ** 0.965) * np.exp(-0.18 * k)
            * (1.0 + 0.04 * np.sin(5.0 * np.log(k + 1e-12)))
            * D ** 2 + 1e-12
        )
        sigma_R[iz] = 2.8 * D ** 0.9 * (R_grid ** -0.32) / (1.0 + (R_grid / 9.0) ** 1.8)
    return k, z, Pk_lin, sigma_R, R_grid


COSMO = hmcode_py.HMcodeCosmology(
    Omega_m=0.314885, Omega_b=0.049, h=0.674, n_s=0.965, sigma_8=0.8,
    w0=-1.0, wa=0.0, Omega_nu=0.00142, Omega_k=0.0,
)


# ---------------------------------------------------------------------------
# Native-Julia reference path: independent of _interp.jl and _bridge.py.
# Builds the same interpolant semantics (log-log linear, nearest-z) but via
# plain closures defined inside the juliacall session.
# ---------------------------------------------------------------------------

_NATIVE_PATH_LOADED = False


def _load_native_path() -> None:
    """Define a Julia function ``hmcode_native_ref`` that takes raw Julia
    arrays and returns Pk in (nk, nz) — without touching _interp.jl."""
    global _NATIVE_PATH_LOADED
    if _NATIVE_PATH_LOADED:
        return
    jl = _bridge.get_julia()
    jl.seval(r"""
        # Independent reference path: same math as _interp.jl but written
        # from scratch using only Base + HMcode. No Interpolations.jl.
        function _logloglin_nearest(xs::Vector{Float64}, ys::Vector{Float64}, x::Real)
            lx = log(x)
            n = length(xs)
            if lx <= xs[1]
                slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
                return exp(ys[1] + (lx - xs[1]) * slope)
            elseif lx >= xs[n]
                slope = (ys[n] - ys[n-1]) / (xs[n] - xs[n-1])
                return exp(ys[n] + (lx - xs[n]) * slope)
            end
            i = searchsortedlast(xs, lx)
            x0, x1 = xs[i], xs[i+1]
            y0, y1 = ys[i], ys[i+1]
            return exp(y0 + (lx - x0) * (y1 - y0) / (x1 - x0))
        end

        function _nearest_idx(zs::Vector{Float64}, z::Real)
            ibest = 1
            dbest = abs(zs[1] - z)
            @inbounds for i in 2:length(zs)
                d = abs(zs[i] - z)
                if d < dbest
                    dbest = d
                    ibest = i
                end
            end
            return ibest
        end

        function hmcode_native_ref(k_in, z_in, Pk_table_in,
                                   sigma_table_in, R_grid_in,
                                   cosmo::HMcode.HMcodeCosmology;
                                   T_AGN=nothing,
                                   Mmin::Float64=1.0, Mmax::Float64=1e18,
                                   nM::Int=256, threaded::Bool=false,
                                   use_fast_specials::Bool=true)
            # Convert juliacall PyArrays to plain Julia Arrays so the rest
            # of this routine can use Base indexing without surprises.
            k = collect(Float64, k_in)
            z = collect(Float64, z_in)
            R_grid = collect(Float64, R_grid_in)
            Pk_table = Matrix{Float64}(Pk_table_in)       # (nk, nz)
            sigma_table = Matrix{Float64}(sigma_table_in) # (nR, nz)

            logk = log.(k)
            logR = log.(R_grid)
            logP = log.(Pk_table)
            logS = log.(sigma_table)

            Pk_lin = (kv, zv) -> begin
                iz = _nearest_idx(z, zv)
                _logloglin_nearest(logk, logP[:, iz], kv)
            end
            sigma_R = (Rv, zv) -> begin
                iz = _nearest_idx(z, zv)
                _logloglin_nearest(logR, logS[:, iz], Rv)
            end

            return HMcode.hmcode_power(k, z, Pk_lin, sigma_R, cosmo;
                                       T_AGN=T_AGN, Mmin=Mmin, Mmax=Mmax,
                                       nM=nM, threaded=threaded,
                                       use_fast_specials=use_fast_specials)
        end
    """)
    _NATIVE_PATH_LOADED = True


def _native_reference(k, z, Pk_lin, sigma_R, R_grid, cosmo, *, T_AGN, nM):
    """Run the independent Julia path; return Pk in (nz, nk)."""
    _load_native_path()
    jl = _bridge.get_julia()

    Pk_jl = np.ascontiguousarray(np.asarray(Pk_lin, dtype=np.float64).T)        # (nk, nz)
    sigma_jl = np.ascontiguousarray(np.asarray(sigma_R, dtype=np.float64).T)    # (nR, nz)
    cosmo_jl = jl.HMcode.HMcodeCosmology(
        cosmo.Omega_m, cosmo.Omega_b, cosmo.h, cosmo.n_s, cosmo.sigma_8,
        cosmo.w0, cosmo.wa, cosmo.Omega_nu, cosmo.Omega_k,
    )
    T_AGN_jl = jl.nothing if T_AGN is None else float(T_AGN)

    Pk_out = jl.hmcode_native_ref(
        np.asarray(k, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        Pk_jl, sigma_jl,
        np.asarray(R_grid, dtype=np.float64),
        cosmo_jl,
        T_AGN=T_AGN_jl, Mmin=1.0, Mmax=1e18, nM=int(nM),
        threaded=False, use_fast_specials=True,
    )
    Pk_np = np.asarray(Pk_out, dtype=np.float64)
    return np.ascontiguousarray(Pk_np.T)  # (nz, nk)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_AGN", [None, 10 ** 7.8])
def test_parity_against_native_julia(T_AGN):
    k, z, Pk_lin, sigma_R, R_grid = _make_inputs()

    Pk_ref = _native_reference(k, z, Pk_lin, sigma_R, R_grid, COSMO,
                               T_AGN=T_AGN, nM=128)
    Pk_wrapper = hmcode_py.hmcode_power_tabulated(
        k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
        cosmo=COSMO, T_AGN=T_AGN, nM=128,
    )

    assert Pk_wrapper.shape == Pk_ref.shape == (z.size, k.size)
    # Two independent interpolant constructions of the same math; tiny
    # roundoff is allowed but should be at numerical-noise level.
    rel = np.abs(Pk_wrapper / Pk_ref - 1.0)
    assert np.max(rel) < 1e-10, (
        f"max relative error = {np.max(rel):.3e} (T_AGN={T_AGN}); "
        f"ref={Pk_ref[0,:3]}, wrapper={Pk_wrapper[0,:3]}"
    )


def test_parity_cosmology_sweep():
    """Sweep a few cosmologies; each must match native Julia at 1e-10."""
    k, z, Pk_lin, sigma_R, R_grid = _make_inputs(nk=32, nz=3)
    cosmos = [
        hmcode_py.HMcodeCosmology(0.30, 0.045, 0.70, 0.96, 0.81),
        hmcode_py.HMcodeCosmology(0.32, 0.050, 0.68, 0.97, 0.79,
                                  w0=-0.95, wa=-0.10),
        hmcode_py.HMcodeCosmology(0.28, 0.048, 0.72, 0.95, 0.83,
                                  Omega_nu=0.0014, Omega_k=0.005),
    ]
    for cosmo in cosmos:
        Pk_ref = _native_reference(k, z, Pk_lin, sigma_R, R_grid, cosmo,
                                   T_AGN=None, nM=64)
        Pk_w = hmcode_py.hmcode_power_tabulated(
            k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
            cosmo=cosmo, T_AGN=None, nM=64,
        )
        rel = np.abs(Pk_w / Pk_ref - 1.0)
        assert np.max(rel) < 1e-10, f"cosmo {cosmo}: max rel err {np.max(rel):.3e}"
