"""Sweep nM values: CAMB-comparison accuracy + runtime benchmarks.

This script does two things for each nM in a list (default: 118,96,64):

1) Accuracy against CAMB HMCode:
   - Uses the 4-case suite from compare_with_camb_hmcode.py
   - Reports worst-case max/median percentage residual

2) Runtime benchmark:
   - Uses nz in {10, 30, 50, 100}
   - Reports HMCode-stage and end-to-end total runtime

Outputs:
  - console tables
  - CSV files
  - two summary plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sibling scripts importable when running as:
#   python scripts/sweep_nm_camb_comparison_and_bench.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_speed_vs_nz import benchmark_one
from scripts.compare_with_camb_hmcode import SUITE_CASES, compute_case


def parse_nm_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty --nM-list")
    return vals


def parse_zs(s: str) -> np.ndarray:
    zs = np.array([float(x) for x in s.split(",") if x.strip()], dtype=float)
    if zs.size == 0:
        raise ValueError("empty --zs")
    if np.any(np.diff(zs) > 0):
        raise ValueError("--zs must be monotonically decreasing")
    return zs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nM-list", default="118,96,64")
    p.add_argument("--nk", type=int, default=128)
    p.add_argument("--nR", type=int, default=256)
    p.add_argument("--zs", type=str, default="2,1,0.5,0")
    p.add_argument("--T-AGN", dest="T_AGN", type=float, default=None)
    p.add_argument("--bench-repeats", type=int, default=2)
    p.add_argument("--out-prefix", default="nm_sweep")
    args = p.parse_args()

    nm_list = parse_nm_list(args.nM_list)
    zs = parse_zs(args.zs)
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

    # 1) Accuracy sweep vs CAMB
    acc_rows = []
    print("\nAccuracy vs CAMB HMCode (percentage residuals)")
    print("-" * 90)
    print(f"{'nM':>6} | {'worst max %':>12} | {'worst med %':>12} | {'worst case':>16}")
    print("-" * 90)
    for nM in nm_list:
        worst_max = -1.0
        worst_med = -1.0
        worst_label = ""
        for case in SUITE_CASES:
            out = compute_case(case, k=k, zs=zs, nM=nM, T_AGN=args.T_AGN)
            if out["max_abs_pct"] > worst_max:
                worst_max = out["max_abs_pct"]
                worst_label = case["label"]
            worst_med = max(worst_med, out["med_abs_pct"])
        acc_rows.append(
            {
                "nM": nM,
                "worst_max_pct": worst_max,
                "worst_med_pct": worst_med,
                "worst_case": worst_label,
            }
        )
        print(f"{nM:6d} | {worst_max:12.4e} | {worst_med:12.4e} | {worst_label:>16}")

    # 2) Benchmark sweep
    bench_rows = []
    print("\nRuntime benchmark (ms)")
    print("-" * 110)
    print(f"{'nM':>6} | {'nz':>6} | {'HMCode ms':>14} | {'Total ms':>14} | {'shape':>12}")
    print("-" * 110)
    for nM in nm_list:
        for nz in nz_values:
            reps = []
            for _ in range(args.bench_repeats):
                reps.append(
                    benchmark_one(
                        nz=nz,
                        k=k,
                        nR=args.nR,
                        T_AGN=args.T_AGN,
                        nM=nM,
                        cosmo_params=cosmo_params,
                    )
                )
            hm_ms = mean([r["t_hmcode"] * 1000 for r in reps])
            tot_ms = mean([r["t_total"] * 1000 for r in reps])
            shape = reps[0]["shape"]
            bench_rows.append(
                {
                    "nM": nM,
                    "nz": nz,
                    "hmcode_ms": hm_ms,
                    "total_ms": tot_ms,
                    "shape": shape,
                }
            )
            print(f"{nM:6d} | {nz:6d} | {hm_ms:14.2f} | {tot_ms:14.2f} | {str(shape):>12}")

    # Write CSVs
    out_prefix = Path(args.out_prefix)
    acc_csv = out_prefix.with_name(out_prefix.name + "_accuracy.csv")
    bench_csv = out_prefix.with_name(out_prefix.name + "_benchmark.csv")

    with acc_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["nM", "worst_max_pct", "worst_med_pct", "worst_case"])
        w.writeheader()
        w.writerows(acc_rows)

    with bench_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["nM", "nz", "hmcode_ms", "total_ms", "shape"])
        w.writeheader()
        w.writerows(bench_rows)

    # Plot A: accuracy vs nM
    x = np.array([r["nM"] for r in acc_rows])
    y_max = np.array([r["worst_max_pct"] for r in acc_rows])
    y_med = np.array([r["worst_med_pct"] for r in acc_rows])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(x, y_max, marker="o", label="Worst max residual [%]")
    ax.plot(x, y_med, marker="o", label="Worst median residual [%]")
    ax.set_xlabel("nM")
    ax.set_ylabel("Residual [%]")
    ax.set_title(f"CAMB comparison residual vs nM (T_AGN={args.T_AGN}, zs={list(zs)})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    acc_png = out_prefix.with_name(out_prefix.name + "_accuracy.png")
    fig.savefig(acc_png, dpi=170)

    # Plot B: HMCode runtime vs nz for each nM
    fig2, ax2 = plt.subplots(figsize=(7, 4.8))
    for nM in nm_list:
        rows = [r for r in bench_rows if r["nM"] == nM]
        rows = sorted(rows, key=lambda r: r["nz"])
        ax2.plot(
            [r["nz"] for r in rows],
            [r["hmcode_ms"] for r in rows],
            marker="o",
            label=f"nM={nM}",
        )
    ax2.set_xlabel("nz")
    ax2.set_ylabel("HMCode stage runtime [ms]")
    ax2.set_title(f"HMCode runtime vs nz (T_AGN={args.T_AGN})")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    bench_png = out_prefix.with_name(out_prefix.name + "_benchmark.png")
    fig2.savefig(bench_png, dpi=170)

    print("\nWrote:")
    print(f"- {acc_csv}")
    print(f"- {bench_csv}")
    print(f"- {acc_png}")
    print(f"- {bench_png}")


if __name__ == "__main__":
    main()
