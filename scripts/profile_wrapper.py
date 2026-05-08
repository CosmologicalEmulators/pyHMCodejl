"""End-to-end profile of the pyhmcode-jl wrapper.

Reports:
  1. cold vs warm wall-time for hmcode_power_tabulated
  2. wrapper vs original Python hmcode.power (speedup)
  3. wrapper vs native Julia (overhead -- how much we're paying to be Python)
  4. CAMB-layer power(...) cold and warm
  5. cProfile breakdown of the hot path
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from statistics import mean, stdev

import numpy as np

import hmcode_py
from hmcode_py import _bridge

# Optional: original Python HMcode for speedup comparison
_ORIG_PY = "/home/marcobonici/Desktop/work/CosmologicalEmulators/test_halofit/HMcode-python"
if _ORIG_PY not in sys.path:
    sys.path.insert(0, _ORIG_PY)
try:
    import camb
    import hmcode as hmcode_orig
    import hmcode.camb_stuff as camb_stuff
    HAVE_CAMB = True
except Exception:
    HAVE_CAMB = False


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_inputs(nk=128, nz=5, nR=256):
    k = np.logspace(-3.0, 1.0, nk)
    z = np.array([3.0, 2.0, 1.0, 0.5, 0.0])[:nz]
    R_grid = np.logspace(-3.0, 2.0, nR)
    Pk_lin = np.empty((nz, nk))
    sigma_R = np.empty((nz, nR))
    for iz, _z in enumerate(z):
        D = 1.0 / (1.0 + _z)
        Pk_lin[iz] = (k ** 0.965) * np.exp(-0.18 * k) * D ** 2 + 1e-12
        sigma_R[iz] = 2.8 * D ** 0.9 * (R_grid ** -0.32) / (1.0 + (R_grid / 9.0) ** 1.8)
    return k, z, Pk_lin, sigma_R, R_grid


COSMO = hmcode_py.HMcodeCosmology(
    Omega_m=0.314885, Omega_b=0.049, h=0.674, n_s=0.965, sigma_8=0.8,
)


def time_call(fn, *args, repeats=5, **kwargs):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ts.append(time.perf_counter() - t0)
    return out, ts


def fmt(ts):
    if len(ts) > 1:
        return f"{mean(ts)*1000:7.1f} ± {stdev(ts)*1000:5.1f} ms  (min {min(ts)*1000:7.1f})"
    return f"{ts[0]*1000:7.1f} ms"


# ---------------------------------------------------------------------------
# 1. Cold vs warm wrapper
# ---------------------------------------------------------------------------

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


section("1. Wrapper hmcode_power_tabulated  (nk=128, nz=5, nR=256, nM=256)")

k, z, Pk_lin, sigma_R, R_grid = make_inputs()

t0 = time.perf_counter()
Pk1 = hmcode_py.hmcode_power_tabulated(
    k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
    cosmo=COSMO, T_AGN=10**7.8, nM=256,
)
cold = time.perf_counter() - t0
print(f"cold call (includes Julia init + JIT): {cold*1000:.1f} ms  shape={Pk1.shape}")

_, warm = time_call(
    hmcode_py.hmcode_power_tabulated, repeats=10,
    k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
    cosmo=COSMO, T_AGN=10**7.8, nM=256,
)
print(f"warm calls (n=10):                     {fmt(warm)}")

# T_AGN off
_, warm_off = time_call(
    hmcode_py.hmcode_power_tabulated, repeats=10,
    k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
    cosmo=COSMO, T_AGN=None, nM=256,
)
print(f"warm calls T_AGN=None (n=10):          {fmt(warm_off)}")


# ---------------------------------------------------------------------------
# 2. Wrapper vs native Julia (overhead)
# ---------------------------------------------------------------------------

section("2. Wrapper overhead vs native Julia hmcode_power")

jl = _bridge.get_julia()
jl.seval("""
function _native_bench(k, z, Pk_lin_fn, sigma_R_fn, cosmo; T_AGN=10^7.8, nM=256)
    HMcode.hmcode_power(k, z, Pk_lin_fn, sigma_R_fn, cosmo;
                         T_AGN=T_AGN, Mmin=1.0, Mmax=1e18, nM=nM,
                         threaded=false, use_fast_specials=true)
end
""")

# Pre-build Julia interpolants once so the native timing excludes the build cost
# (this is what a Julia user would do in a tight loop too).
Pk_jl = np.ascontiguousarray(Pk_lin.T)
sigma_jl = np.ascontiguousarray(sigma_R.T)
Pk_lin_fn = jl.build_Pk_lin_interp(k, z, Pk_jl)
sigma_R_fn = jl.build_sigma_R_interp(R_grid, z, sigma_jl)
cosmo_jl = jl.HMcode.HMcodeCosmology(
    COSMO.Omega_m, COSMO.Omega_b, COSMO.h, COSMO.n_s, COSMO.sigma_8,
    COSMO.w0, COSMO.wa, COSMO.Omega_nu, COSMO.Omega_k,
)

# warm up
jl._native_bench(k, z, Pk_lin_fn, sigma_R_fn, cosmo_jl, T_AGN=10**7.8, nM=256)

native_ts = []
for _ in range(10):
    t0 = time.perf_counter()
    jl._native_bench(k, z, Pk_lin_fn, sigma_R_fn, cosmo_jl, T_AGN=10**7.8, nM=256)
    native_ts.append(time.perf_counter() - t0)

print(f"native Julia (interpolants prebuilt):  {fmt(native_ts)}")
print(f"wrapper warm  (rebuilds interp + xfer):{fmt(warm)}")
overhead_ms = (mean(warm) - mean(native_ts)) * 1000
print(f"-> wrapper overhead per call:          {overhead_ms:+.1f} ms "
      f"({(mean(warm)/mean(native_ts) - 1)*100:+.1f}%)")


# ---------------------------------------------------------------------------
# 3. Compare to original Python HMcode (speedup)
# ---------------------------------------------------------------------------

if HAVE_CAMB:
    section("3. Wrapper.power(CAMB) vs original Python hmcode.power(CAMB)")

    _, results, _, _, _ = camb_stuff.run(
        np.array([3., 2., 1., 0.5, 0.]),
        0.265, 0.049, 0.0, 0.674, 0.965, 0.8,
    )
    K = np.logspace(-3, 1, 128)
    ZS = np.array([3., 2., 1., 0.5, 0.])

    # Warm both
    hmcode_py.power(K, ZS, results, T_AGN=None)
    hmcode_orig.power(K, ZS, results, T_AGN=None)

    _, t_jl = time_call(hmcode_py.power, K, ZS, results, repeats=5, T_AGN=None)
    _, t_orig = time_call(hmcode_orig.power, K, ZS, results, repeats=5, T_AGN=None)

    print(f"hmcode_py.power      (warm, n=5):    {fmt(t_jl)}")
    print(f"hmcode_orig.power    (warm, n=5):    {fmt(t_orig)}")
    print(f"-> speedup:                            {mean(t_orig)/mean(t_jl):.2f}x")
else:
    print("\n(skipping section 3: CAMB / original HMcode not available)")


# ---------------------------------------------------------------------------
# 4. cProfile breakdown of the warm hot path
# ---------------------------------------------------------------------------

section("4. cProfile of one warm hmcode_power_tabulated call")

prof = cProfile.Profile()
prof.enable()
for _ in range(5):
    hmcode_py.hmcode_power_tabulated(
        k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
        cosmo=COSMO, T_AGN=10**7.8, nM=256,
    )
prof.disable()

s = io.StringIO()
pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumulative").print_stats(15)
print(s.getvalue())
