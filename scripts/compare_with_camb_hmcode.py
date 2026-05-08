"""Compare wrapped HMCode.jl directly against CAMB's HMCode implementation.

This script compares:
  - hmcode_py.hmcode_power_tabulated(...)  [HMCode.jl backend]
  - CAMB nonlinear spectrum (halofit_version='mead2020' or 'mead2020_feedback')

Output plot shows percentage residuals:
  100 * (P_jl / P_camb - 1)

Examples
--------
Single cosmology (no feedback):
  python scripts/compare_with_camb_hmcode.py --output compare_camb.png

Suite of 4 cosmologies (includes w0, wa, mnu):
  python scripts/compare_with_camb_hmcode.py --suite --output compare_camb_suite.png

Enable feedback model explicitly:
  python scripts/compare_with_camb_hmcode.py --suite --T-AGN 63095734.448 --output compare_camb_feedback.png
"""

from __future__ import annotations

import argparse

import camb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hmcode_py


SUITE_CASES = [
    {
        "label": "LCDM",
        "h": 0.674,
        "omega_b": 0.049,
        "omega_cdm": 0.265,
        "mnu": 0.0,
        "omk": 0.0,
        "As": 2.1e-9,
        "ns": 0.965,
        "w0": -1.0,
        "wa": 0.0,
    },
    {
        "label": "LCDM + mnu",
        "h": 0.674,
        "omega_b": 0.049,
        "omega_cdm": 0.265,
        "mnu": 0.06,
        "omk": 0.0,
        "As": 2.1e-9,
        "ns": 0.965,
        "w0": -1.0,
        "wa": 0.0,
    },
    {
        "label": "w0wa",
        "h": 0.70,
        "omega_b": 0.050,
        "omega_cdm": 0.260,
        "mnu": 0.0,
        "omk": 0.0,
        "As": 2.1e-9,
        "ns": 0.960,
        "w0": -0.95,
        "wa": -0.10,
    },
    {
        "label": "w0wa + mnu",
        "h": 0.69,
        "omega_b": 0.050,
        "omega_cdm": 0.255,
        "mnu": 0.10,
        "omk": 0.0,
        "As": 2.0e-9,
        "ns": 0.970,
        "w0": -0.90,
        "wa": -0.25,
    },
]


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
    T_AGN: float | None,
) -> camb.CAMBdata:
    """Build CAMB result with HMCode nonlinear model enabled."""
    if T_AGN is None:
        nonlin_model = camb.nonlinear.Halofit(halofit_version="mead2020")
    else:
        nonlin_model = camb.nonlinear.Halofit(
            halofit_version="mead2020_feedback",
            HMCode_logT_AGN=float(np.log10(T_AGN)),
        )

    pars = camb.CAMBparams(NonLinearModel=nonlin_model)
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
    nR: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Pk_lin(z,k) and sigma_R(z,R) tables from CAMB."""
    interp_lin, _, _ = results.get_matter_power_interpolator(
        nonlinear=False,
        return_z_k=True,
        extrap_kmax=1e10,
    )
    Pk_lin = np.empty((zs.size, k.size), dtype=np.float64)
    for iz, z in enumerate(zs):
        Pk_lin[iz, :] = interp_lin.P(float(z), k)

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


def compute_case(
    case: dict,
    *,
    k: np.ndarray,
    zs: np.ndarray,
    nM: int,
    T_AGN: float | None,
) -> dict:
    """Compute HMCode.jl and CAMB HMCode spectra for one cosmology."""
    results = build_camb_results(
        zs,
        h=case["h"],
        omega_b=case["omega_b"],
        omega_cdm=case["omega_cdm"],
        mnu=case["mnu"],
        omk=case["omk"],
        As=case["As"],
        ns=case["ns"],
        w0=case["w0"],
        wa=case["wa"],
        T_AGN=T_AGN,
    )

    Pk_lin, sigma_R, R_grid = extract_linear_tables(results, k, zs)

    Om_b = results.get_Omega(var="baryon", z=0.0)
    Om_c = results.get_Omega(var="cdm", z=0.0)
    Om_nu = results.get_Omega(var="nu", z=0.0)
    params = results.Params

    cosmo = hmcode_py.HMcodeCosmology(
        Omega_m=Om_b + Om_c + Om_nu,
        Omega_b=Om_b,
        h=params.H0 / 100.0,
        n_s=params.InitPower.ns,
        sigma_8=float(results.get_sigma8_0()),
        w0=getattr(params.DarkEnergy, "w", -1.0),
        wa=getattr(params.DarkEnergy, "wa", 0.0),
        Omega_nu=Om_nu,
        Omega_k=params.omk,
    )

    Pk_jl = hmcode_py.hmcode_power_tabulated(
        k=k,
        z=zs,
        Pk_lin=Pk_lin,
        sigma_R=sigma_R,
        R_grid=R_grid,
        cosmo=cosmo,
        T_AGN=T_AGN,
        nM=nM,
    )

    interp_nl = results.get_matter_power_interpolator(nonlinear=True).P
    Pk_camb = np.empty_like(Pk_jl)
    for iz, z in enumerate(zs):
        Pk_camb[iz, :] = interp_nl(float(z), k)

    frac = Pk_jl / Pk_camb - 1.0
    frac_pct = 100.0 * frac

    return {
        "case": case,
        "k": k,
        "zs": zs,
        "Pk_jl": Pk_jl,
        "Pk_camb": Pk_camb,
        "frac": frac,
        "frac_pct": frac_pct,
        "max_abs": float(np.max(np.abs(frac))),
        "med_abs": float(np.median(np.abs(frac))),
        "max_abs_pct": float(np.max(np.abs(frac_pct))),
        "med_abs_pct": float(np.median(np.abs(frac_pct))),
    }


def run_single(args: argparse.Namespace) -> None:
    zs = np.array([3.0, 2.0, 1.0, 0.5, 0.0], dtype=np.float64)
    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)
    case = {
        "label": "custom",
        "h": args.h,
        "omega_b": args.omega_b,
        "omega_cdm": args.omega_cdm,
        "mnu": args.mnu,
        "omk": args.omk,
        "As": args.As,
        "ns": args.ns,
        "w0": args.w0,
        "wa": args.wa,
    }
    out = compute_case(case, k=k, zs=zs, nM=args.nM, T_AGN=args.T_AGN)

    print(f"Max |ΔP/P| = {out['max_abs']:.3e}  ({out['max_abs_pct']:.3e} %)" )
    print(f"Med |ΔP/P| = {out['med_abs']:.3e}  ({out['med_abs_pct']:.3e} %)" )

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for iz, z in enumerate(out["zs"]):
        ax_top.loglog(out["k"], out["Pk_jl"][iz], lw=2.0, label=f"HMCode.jl (z={z:g})")
        ax_top.loglog(out["k"], out["Pk_camb"][iz], ls="--", lw=1.5, label=f"CAMB HMCode (z={z:g})")
        ax_bot.semilogx(out["k"], out["frac_pct"][iz], lw=1.8)

    ax_top.set_ylabel(r"$P_{\rm nl}(k)$")
    ax_top.set_title(f"HMCode.jl vs CAMB HMCode (nM={args.nM}, T_AGN={args.T_AGN})")
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=8, ncol=2)

    ax_bot.axhline(0.0, color="k", lw=1.0)
    ax_bot.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax_bot.set_ylabel(r"$100\times(P_{\rm jl}/P_{\rm camb}-1)\,[\%]$")
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote: {args.output}")

    if args.show:
        plt.show()


def run_suite(args: argparse.Namespace) -> None:
    zs = np.array([2.0, 1.0, 0.5, 0.0], dtype=np.float64)
    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)

    print("Running CAMB-HMCode comparison suite (4 cosmologies)...")
    outputs = []
    for case in SUITE_CASES:
        out = compute_case(case, k=k, zs=zs, nM=args.nM, T_AGN=args.T_AGN)
        outputs.append(out)
        print(
            f"- {case['label']:<12}  "
            f"max |ΔP/P| = {out['max_abs']:.3e} ({out['max_abs_pct']:.3e} %)   "
            f"med |ΔP/P| = {out['med_abs']:.3e} ({out['med_abs_pct']:.3e} %)"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, out in enumerate(outputs):
        ax = axes[i]
        for iz, z in enumerate(out["zs"]):
            ax.semilogx(out["k"], out["frac_pct"][iz], lw=1.7, label=f"z={z:g}")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.grid(alpha=0.3)
        ax.set_title(
            f"{out['case']['label']}\n"
            f"max {out['max_abs_pct']:.2e} %, med {out['med_abs_pct']:.2e} %",
            fontsize=10,
        )

    for ax in axes[2:]:
        ax.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel(r"$100\times(P_{\rm jl}/P_{\rm camb}-1)\,[\%]$")

    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(f"HMCode.jl vs CAMB HMCode (nM={args.nM}, T_AGN={args.T_AGN})", y=0.99)
    fig.tight_layout()
    fig.savefig(args.output, dpi=170)
    print(f"Wrote: {args.output}")

    if args.show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="hmcode_vs_camb.png")
    parser.add_argument("--show", action="store_true", help="show figure interactively")
    parser.add_argument("--suite", action="store_true", help="run 4 preset cosmologies")

    parser.add_argument("--nk", type=int, default=128)
    parser.add_argument("--kmin", type=float, default=1e-3)
    parser.add_argument("--kmax", type=float, default=10.0)
    parser.add_argument("--nM", type=int, default=256)
    parser.add_argument(
        "--T-AGN",
        dest="T_AGN",
        type=float,
        default=None,
        help="AGN feedback temperature in K. Default: None (no feedback).",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="set T_AGN=None and compare no-feedback HMCode variants",
    )

    # single-case cosmology knobs
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-b", type=float, default=0.049)
    parser.add_argument("--omega-cdm", type=float, default=0.265)
    parser.add_argument("--mnu", type=float, default=0.0)
    parser.add_argument("--omk", type=float, default=0.0)
    parser.add_argument("--As", type=float, default=2.1e-9)
    parser.add_argument("--ns", type=float, default=0.965)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--wa", type=float, default=0.0)

    args = parser.parse_args()
    if args.no_feedback:
        args.T_AGN = None
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.suite:
        run_suite(args)
    else:
        run_single(args)
