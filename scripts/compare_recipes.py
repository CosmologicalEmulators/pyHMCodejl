"""Compare nonlinear P(k) from wrapped HMCode.jl vs original Python HMcode.

What this script does:
1) Uses CAMB to compute linear P(k, z).
2) Feeds that linear table + sigma(R, z) into hmcode_py.hmcode_power_tabulated.
3) Runs the original Python HMcode implementation on the same CAMB_results.
4) Produces a plot comparing both recipes and their fractional difference.

Usage example:
  python scripts/compare_recipes.py --output hmcode_recipe_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import camb
import matplotlib

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


def _import_original_hmcode() -> object:
    """Import reference Python HMcode.

    Preferred path: pip package (`pip install hmcode`).
    Fallback path: sibling checkout at ../HMcode-python.
    """
    try:
        import hmcode as hmcode_orig  # type: ignore
        return hmcode_orig
    except Exception:
        pass

    here = Path(__file__).resolve()
    pyhmcodejl_root = here.parents[1]
    hmcode_python_root = pyhmcodejl_root.parent / "HMcode-python"
    if hmcode_python_root.exists():
        sys.path.insert(0, str(hmcode_python_root))
        import hmcode as hmcode_orig  # type: ignore
        return hmcode_orig

    raise ModuleNotFoundError(
        "Could not import original Python hmcode. Install it with `pip install hmcode`, "
        "or place a sibling checkout at ../HMcode-python."
    )


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
    """Create CAMB results object with linear power available on requested z grid."""
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
    nR: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear P(k,z) and sigma(R,z) grids from CAMB.

    Returns
    -------
    Pk_lin : (nz, nk)
    sigma_R : (nz, nR)
    R_grid : (nR,)
    """
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


def compute_case(
    case: dict,
    *,
    hmcode_orig: object,
    k: np.ndarray,
    zs: np.ndarray,
    nM: int,
    T_AGN: float | None,
) -> dict:
    """Run one cosmology case through both implementations."""
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
    Pk_py = hmcode_orig.power(k, zs, results, T_AGN=T_AGN, nM=nM)
    frac = Pk_jl / Pk_py - 1.0

    return {
        "case": case,
        "k": k,
        "zs": zs,
        "Pk_jl": Pk_jl,
        "Pk_py": Pk_py,
        "frac": frac,
        "frac_pct": 100.0 * frac,
        "max_abs": float(np.max(np.abs(frac))),
        "med_abs": float(np.median(np.abs(frac))),
        "max_abs_pct": float(np.max(np.abs(100.0 * frac))),
        "med_abs_pct": float(np.median(np.abs(100.0 * frac))),
    }


def run_comparison(args: argparse.Namespace) -> None:
    hmcode_orig = _import_original_hmcode()

    # Redshifts must be decreasing for original Python HMcode.
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
    out = compute_case(
        case,
        hmcode_orig=hmcode_orig,
        k=k,
        zs=zs,
        nM=args.nM,
        T_AGN=args.T_AGN,
    )

    Pk_jl = out["Pk_jl"]
    Pk_py = out["Pk_py"]
    frac = out["frac"]
    frac_pct = out["frac_pct"]

    print(f"Max |ΔP/P| = {out['max_abs']:.3e}  ({out['max_abs_pct']:.3e} %)" )
    print(f"Med |ΔP/P| = {out['med_abs']:.3e}  ({out['med_abs_pct']:.3e} %)" )

    # 4) Plot comparison
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    for iz, z in enumerate(zs):
        ax_top.loglog(k, Pk_jl[iz], lw=2.0, label=f"hmcode_py (z={z:g})")
        ax_top.loglog(k, Pk_py[iz], ls="--", lw=1.5, label=f"orig py (z={z:g})")
        ax_bot.semilogx(k, frac_pct[iz], lw=1.8)

    ax_top.set_ylabel(r"$P_{\rm nl}(k)$")
    ax_top.set_title(
        f"HMCode recipe comparison (nM={args.nM}, T_AGN={args.T_AGN})"
    )
    ax_top.grid(alpha=0.3)
    ax_top.legend(fontsize=8, ncol=2)

    ax_bot.axhline(0.0, color="k", lw=1.0)
    ax_bot.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
    ax_bot.set_ylabel(r"$100\times(P_{\rm jl}/P_{\rm py}-1)\,[\%]$")
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote: {args.output}")

    if args.show:
        plt.show()


def run_suite(args: argparse.Namespace) -> None:
    """Run 4 preset cosmologies and plot fractional agreement panels."""
    hmcode_orig = _import_original_hmcode()
    zs = np.array([3.0, 2.0, 1.0, 0.5, 0.0], dtype=np.float64)
    k = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk)

    results = []
    print("Running recipe comparison suite (4 cosmologies)...")
    for case in SUITE_CASES:
        out = compute_case(
            case,
            hmcode_orig=hmcode_orig,
            k=k,
            zs=zs,
            nM=args.nM,
            T_AGN=args.T_AGN,
        )
        results.append(out)
        print(
            f"- {case['label']:<12}  "
            f"max |ΔP/P| = {out['max_abs']:.3e} ({out['max_abs_pct']:.3e} %)   "
            f"med |ΔP/P| = {out['med_abs']:.3e} ({out['med_abs_pct']:.3e} %)"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, out in enumerate(results):
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
        ax.set_ylabel(r"$100\times(P_{\rm jl}/P_{\rm py}-1)\,[\%]$")

    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(
        f"HMCode recipe comparison suite (nM={args.nM}, T_AGN={args.T_AGN})",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=170)
    print(f"Wrote: {args.output}")

    if args.show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="hmcode_recipe_comparison.png")
    parser.add_argument("--show", action="store_true", help="show figure interactively")
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run 4 preset cosmologies (LCDM, mnu, w0wa, w0wa+mnu)",
    )
    parser.add_argument("--nk", type=int, default=128)
    parser.add_argument("--kmin", type=float, default=1e-3)
    parser.add_argument("--kmax", type=float, default=10.0)
    parser.add_argument("--nM", type=int, default=256)
    parser.add_argument("--T-AGN", dest="T_AGN", type=float, default=10**7.8)

    # Cosmology
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-b", type=float, default=0.049)
    parser.add_argument("--omega-cdm", type=float, default=0.265)
    parser.add_argument("--mnu", type=float, default=0.0)
    parser.add_argument("--omk", type=float, default=0.0)
    parser.add_argument("--As", type=float, default=2.1e-9)
    parser.add_argument("--ns", type=float, default=0.965)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--wa", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.suite:
        run_suite(args)
    else:
        run_comparison(args)
