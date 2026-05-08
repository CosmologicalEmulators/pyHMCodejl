"""Verify threaded=True works and measure speedup.

Re-execs itself with PYTHON_JULIACALL_THREADS in {1,2,4,8} (capped at
host cpu count) so each child runs in a fresh juliacall process with the
right Julia thread budget.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from statistics import mean, stdev

import numpy as np


def make_inputs(nk=128, nz=8, nR=256):
    k = np.logspace(-3.0, 1.0, nk)
    z = np.linspace(0.0, 4.0, nz)[::-1].copy()
    R_grid = np.logspace(-3.0, 2.0, nR)
    Pk_lin = np.empty((nz, nk))
    sigma_R = np.empty((nz, nR))
    for iz, _z in enumerate(z):
        D = 1.0 / (1.0 + _z)
        Pk_lin[iz] = (k ** 0.965) * np.exp(-0.18 * k) * D ** 2 + 1e-12
        sigma_R[iz] = 2.8 * D ** 0.9 * (R_grid ** -0.32) / (1.0 + (R_grid / 9.0) ** 1.8)
    return k, z, Pk_lin, sigma_R, R_grid


def child_main():
    import hmcode_py
    from hmcode_py import _bridge

    jl = _bridge.get_julia()
    nthreads = int(jl.seval("Threads.nthreads()"))

    cosmo = hmcode_py.HMcodeCosmology(
        Omega_m=0.314885, Omega_b=0.049, h=0.674, n_s=0.965, sigma_8=0.8,
    )
    k, z, Pk_lin, sigma_R, R_grid = make_inputs()

    def run(threaded, nM=256, repeats=8):
        out = hmcode_py.hmcode_power_tabulated(  # warmup
            k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
            cosmo=cosmo, T_AGN=10**7.8, nM=nM, threaded=threaded,
        )
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = hmcode_py.hmcode_power_tabulated(
                k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
                cosmo=cosmo, T_AGN=10**7.8, nM=nM, threaded=threaded,
            )
            ts.append(time.perf_counter() - t0)
        return out, ts

    Pk_serial, ts_serial = run(threaded=False)
    Pk_thread, ts_thread = run(threaded=True)

    rel = np.abs(Pk_thread / Pk_serial - 1.0)

    def fmt(ts):
        return (f"mean {mean(ts)*1000:7.2f} ms  "
                f"std {stdev(ts)*1000:5.2f} ms  "
                f"min {min(ts)*1000:7.2f} ms")

    print(f"[child] Julia thread count: {nthreads}")
    print(f"[child] threaded=False  {fmt(ts_serial)}")
    print(f"[child] threaded=True   {fmt(ts_thread)}")
    print(f"[child] speedup: {mean(ts_serial)/mean(ts_thread):.2f}x   "
          f"min-speedup: {min(ts_serial)/min(ts_thread):.2f}x")
    print(f"[child] threaded vs serial   max rel err: {np.max(rel):.3e}   "
          f"med rel err: {np.median(rel):.3e}")


def parent_main():
    cpu = os.cpu_count() or 1
    print(f"host CPU count: {cpu}")
    for n in (1, 2, 4, 8):
        if n > cpu:
            continue
        print()
        print(f"--- spawning child with PYTHON_JULIACALL_THREADS={n} ---")
        env = os.environ.copy()
        env["PYTHON_JULIACALL_THREADS"] = str(n)
        # juliacall warns that multi-threaded Julia from Python needs this
        # to avoid segfaults on signal delivery. Cost: Ctrl-C inside Python
        # won't raise KeyboardInterrupt while a Julia call is in flight.
        env["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        env["_HMCODE_CHILD"] = "1"
        subprocess.run([sys.executable, __file__], env=env, check=True)


if __name__ == "__main__":
    if os.environ.get("_HMCODE_CHILD") == "1":
        child_main()
    else:
        parent_main()
