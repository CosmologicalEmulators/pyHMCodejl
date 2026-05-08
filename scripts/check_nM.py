"""Effect of nM (halo-mass grid size) on accuracy and timing.

Reference: nM = 1024. Sweep nM in {16,32,64,128,256,512,1024}. Reports
max/median relative deviation from reference and warm wall-clock per
call. Runs the sweep with T_AGN=10^7.8 and again with T_AGN=None.
"""

from __future__ import annotations

import time
from statistics import mean, stdev

import numpy as np

import hmcode_py


def make_inputs(nk=128, nz=5, nR=256):
    k = np.logspace(-3.0, 1.0, nk)
    z = np.array([3.0, 2.0, 1.0, 0.5, 0.0])
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

NM_LIST = [16, 32, 64, 128, 256, 512, 1024]


def time_call(repeats=6, **kwargs):
    hmcode_py.hmcode_power_tabulated(**kwargs)  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = hmcode_py.hmcode_power_tabulated(**kwargs)
        ts.append(time.perf_counter() - t0)
    return out, ts


def sweep(label, T_AGN, k, z, Pk_lin, sigma_R, R_grid):
    print(f"\n--- {label}  (T_AGN = {T_AGN}) ---")
    print(f"computing reference at nM = 1024 ...")
    Pk_ref, ts_ref = time_call(
        repeats=4,
        k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
        cosmo=COSMO, T_AGN=T_AGN, nM=1024,
    )
    print(f"reference timing: mean {mean(ts_ref)*1000:.1f} ms")
    print()
    print(f"{'nM':>5} | {'time/call (ms)':>30} | {'median rel err':>16} | {'max rel err':>14}")
    print("-" * 78)
    for nM in NM_LIST:
        Pk, ts = time_call(
            repeats=6,
            k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R_grid,
            cosmo=COSMO, T_AGN=T_AGN, nM=nM,
        )
        rel = np.abs(Pk / Pk_ref - 1.0)
        t_str = (f"{mean(ts)*1000:7.2f} ± {stdev(ts)*1000:5.2f} "
                 f"(min {min(ts)*1000:6.2f})")
        print(f"{nM:>5} | {t_str:>30} | {np.median(rel):>16.3e} | "
              f"{np.max(rel):>14.3e}")


def main():
    k, z, Pk_lin, sigma_R, R_grid = make_inputs()
    sweep("Feedback ON",  10**7.8, k, z, Pk_lin, sigma_R, R_grid)
    sweep("Feedback OFF", None,    k, z, Pk_lin, sigma_R, R_grid)


if __name__ == "__main__":
    main()
