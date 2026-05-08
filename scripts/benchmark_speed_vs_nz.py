"""Benchmark end-to-end runtime vs number of redshifts.

Measures, for nz in {10, 30, 50, 100}:
  1) CAMB run time (build linear-theory results object)
  2) Linear-table extraction time (Pk_lin + sigma_R)
  3) HMCode wrapper time (hmcode_power_tabulated)
  4) End-to-end total time

"Computing everything" here means CAMB + table extraction + HMCode call.

Usage:
  python scripts/benchmark_speed_vs_nz.py
  python scripts/benchmark_speed_vs_nz.py --repeats 5 --output benchmark_vs_nz.png
  python scripts/benchmark_speed_vs_nz.py --T-AGN 63095734.448 --nM 256
"""

from __future__ import annotations

import argparse
import time
from statistics import mean, stdev

import camb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hmcode_py


def build_camb_results(
    zs: np.ndarray,
    *,
    h: float,
    omega_b: float,
    omega_cdm: float,
    mnu: float,
    omk: float,
    As: float,
    ns: float,
    w0: float,
    wa: float,
) -> camb.CAMBdata:
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * h,
        ombh2=omega_b * h**2,
        omch2=omega_cdm * h**2,
        mnu=mnu,
        omk=omk,
    )
    pars.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=zs.tolist(), kmax=200.0)
    return camb.get_results(pars)


def extract_linear_tables(
    results: camb.CAMBdata,
    k: np.ndarray,
    zs: np.ndarray,
    *,
    nR: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interp, _, _ = results.get_matter_power_interpolator(
        nonlinear=False,
        return_z_k=True,
        extrap_kmax=1e10,
    )

    Pk_lin = np.empty((zs.size, k.size), dtype=np.float64)
    for iz, z in enumerate(zs):
        Pk_lin[iz, :] = interp.P(float(z), k)

    R_grid = np.logspace(-3.0, 2.0, nR)
    sigma_R = np.empty((zs.size, nR), dtype=np.float64)
    for iz in range(zs.size):
        sigma_R[iz, :] = results.get_sigmaR(
            R_grid,
            z_indices=[iz],
            var1="delta_nonu",
            var2="delta_nonu",
        )[0]

    return Pk_lin, sigma_R, R_grid


def build_hmcode_cosmo(results: camb.CAMBdata) -> hmcode_py.HMcodeCosmology:
    Om_b = results.get_Omega(var="baryon", z=0.0)
    Om_c = results.get_Omega(var="cdm", z=0.0)
    Om_nu = results.get_Omega(var="nu", z=0.0)
    p = results.Params
    return hmcode_py.HMcodeCosmology(
        Omega_m=Om_b + Om_c + Om_nu,
        Omega_b=Om_b,
        h=p.H0 / 100.0,
        n_s=p.InitPower.ns,
        sigma_8=float(results.get_sigma8_0()),
        w0=getattr(p.DarkEnergy, "w", -1.0),
        wa=getattr(p.DarkEnergy, "wa", 0.0),
        Omega_nu=Om_nu,
        Omega_k=p.omk,
    )


def benchmark_one(
    *,
    nz: int,
    k: np.ndarray,
    nR: int,
    T_AGN: float | None,
    nM: int,
    cosmo_params: dict,
) -> dict:
    # Use lower z range so sigma_R extraction and feedback are stable for all nz.
    zs = np.linspace(2.0, 0.0, nz)

    t0 = time.perf_counter()
    results = build_camb_results(zs, **cosmo_params)
    t_camb = time.perf_counter() - t0

    t1 = time.perf_counter()
    Pk_lin, sigma_R, R_grid = extract_linear_tables(results, k, zs, nR=nR)
    t_extract = time.perf_counter() - t1

    cosmo = build_hmcode_cosmo(results)

    t2 = time.perf_counter()
    Pk_nl = hmcode_py.hmcode_power_tabulated(
        k=k,
        z=zs,
        Pk_lin=Pk_lin,
        sigma_R=sigma_R,
        R_grid=R_grid,
        cosmo=cosmo,
        T_AGN=T_AGN,
        nM=nM,
    )
    t_hmcode = time.perf_counter() - t2

    total = t_camb + t_extract + t_hmcode

    return {
        "nz": nz,
        "t_camb": t_camb,
        "t_extract": t_extract,
        "t_hmcode": t_hmcode,
        "t_total": total,
        "shape": Pk_nl.shape,
    }


def fmt(ms_vals: list[float]) -> str:
    if len(ms_vals) == 1:
        return f"{ms_vals[0]:8.1f}"
    return f"{mean(ms_vals):8.1f} ± {stdev(ms_vals):5.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="benchmark repeats per nz")
    parser.add_argument("--nk", type=int, default=128)
    parser.add_argument("--nR", type=int, default=256)
    parser.add_argument("--nM", type=int, default=256)
    parser.add_argument("--T-AGN", dest="T_AGN", type=float, default=None)
    parser.add_argument("--output", default="benchmark_vs_nz.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    k = np.logspace(-3.0, 1.0, args.nk)
    nz_values = [10, 30, 50, 100]

    cosmo_params = {
        "h": 0.674,
        "omega_b": 0.049,
        "omega_cdm": 0.265,
        "mnu": 0.06,
        "omk": 0.0,
        "As": 2.1e-9,
        "ns": 0.965,
        "w0": -0.95,
        "wa": -0.10,
    }

    # Warm-up once to avoid first-call Julia JIT cost polluting results.
    print("Warm-up call (JIT/precompile effects)...")
    _ = benchmark_one(
        nz=5,
        k=k,
        nR=args.nR,
        T_AGN=args.T_AGN,
        nM=args.nM,
        cosmo_params=cosmo_params,
    )

    rows: list[dict] = []
    for nz in nz_values:
        reps = []
        for _ in range(args.repeats):
            reps.append(
                benchmark_one(
                    nz=nz,
                    k=k,
                    nR=args.nR,
                    T_AGN=args.T_AGN,
                    nM=args.nM,
                    cosmo_params=cosmo_params,
                )
            )

        rows.append(
            {
                "nz": nz,
                "camb_ms": [r["t_camb"] * 1000 for r in reps],
                "extract_ms": [r["t_extract"] * 1000 for r in reps],
                "hmcode_ms": [r["t_hmcode"] * 1000 for r in reps],
                "total_ms": [r["t_total"] * 1000 for r in reps],
                "shape": reps[0]["shape"],
            }
        )

    print("\nBenchmark results (milliseconds)")
    print(f"T_AGN={args.T_AGN}, nM={args.nM}, nk={args.nk}, nR={args.nR}, repeats={args.repeats}")
    print("-" * 95)
    print(f"{'nz':>5} | {'CAMB':>16} | {'Extract':>16} | {'HMCode':>16} | {'Total':>16} | {'output shape':>12}")
    print("-" * 95)
    for row in rows:
        print(
            f"{row['nz']:5d} | {fmt(row['camb_ms']):>16} | {fmt(row['extract_ms']):>16} | "
            f"{fmt(row['hmcode_ms']):>16} | {fmt(row['total_ms']):>16} | {str(row['shape']):>12}"
        )

    # Plot mean stage runtimes vs nz
    xs = np.array([r["nz"] for r in rows], dtype=float)
    camb = np.array([mean(r["camb_ms"]) for r in rows])
    ext = np.array([mean(r["extract_ms"]) for r in rows])
    hmc = np.array([mean(r["hmcode_ms"]) for r in rows])
    tot = np.array([mean(r["total_ms"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, camb, marker="o", label="CAMB")
    ax.plot(xs, ext, marker="o", label="Extract linear tables")
    ax.plot(xs, hmc, marker="o", label="HMCode wrapper")
    ax.plot(xs, tot, marker="o", lw=2.5, label="Total")
    ax.set_xlabel("Number of redshifts (nz)")
    ax.set_ylabel("Runtime [ms]")
    ax.set_title(f"End-to-end runtime vs nz (T_AGN={args.T_AGN}, nM={args.nM})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=170)
    print(f"\nWrote: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

