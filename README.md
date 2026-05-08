# pyhmcode-jl

Python wrapper around [HMCode.jl](https://github.com/CosmologicalEmulators/HMCode.jl)
via [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/).

The Julia package does the work; this Python package is a thin bridge
that hands tabulated arrays to Julia and gets back the HMcode2020
non-linear matter power spectrum as a numpy array.

Why bother: ~24× faster than the original Python HMcode, with output
agreeing to a few parts in 10⁴ on every cosmology we've tested.

---

## 1. Install

### Prerequisites

- Python ≥ 3.10
- A working `pip`. Julia itself is **not** required up front — `juliacall` /
  `juliapkg` will download a private Julia 1.11 on first import if you
  don't already have one.

### Install the package

From the repository root:

```bash
# (recommended) make a clean venv first
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# editable install
pip install -e .
```

If you also want to run the CAMB-based parity tests:

```bash
pip install -e ".[test]"
```

That pulls in `pytest` and `camb`. (The `power(..., CAMB_results, ...)`
high-level API requires `camb` at runtime; the low-level
`hmcode_power_tabulated` API does not.)

### First import

The first time you run `import hmcode_py`, three things happen:

1. `juliapkg` ensures a compatible Julia is on disk (downloads one if
   missing).
2. It resolves `HMCode.jl` from GitHub at the pinned commit
   (`74f213a`) into a private Julia project under `.venv/julia_env/`.
3. It precompiles HMcode.jl and its dependencies.

Expect ~30–60 s for this one-time setup. Subsequent imports take a
couple of seconds, and individual calls take tens of milliseconds.

---

## 2. Run it

The package exposes a two-layer API.

### Layer 1 — `hmcode_power_tabulated` (low level)

You supply linear power and σ(R) on grids you control. No CAMB
dependency. Useful for emulator pipelines and tight loops.

```python
import numpy as np
from hmcode_py import HMcodeCosmology, hmcode_power_tabulated

# Wavenumbers and redshifts
k = np.logspace(-3.0, 1.0, 128)        # (nk,) [h/Mpc]
z = np.array([3.0, 2.0, 1.0, 0.5, 0.0])  # (nz,) — *monotonically decreasing*

# Tabulated linear matter power and sigma(R, z), both strictly positive.
# Shapes are (nz, nk) and (nz, nR), matching numpy's natural row-major layout.
R = np.logspace(-3.0, 2.0, 256)        # (nR,) [Mpc/h]
Pk_lin   = ...   # shape (nz, nk),  units (Mpc/h)^3
sigma_R  = ...   # shape (nz, nR)  — cold-matter sigma(R, z) is the right choice

cosmo = HMcodeCosmology(
    Omega_m=0.315, Omega_b=0.049, h=0.674, n_s=0.965, sigma_8=0.8,
    w0=-1.0, wa=0.0, Omega_nu=0.0, Omega_k=0.0,
)

Pk_nl = hmcode_power_tabulated(
    k=k, z=z, Pk_lin=Pk_lin, sigma_R=sigma_R, R_grid=R,
    cosmo=cosmo,
    T_AGN=10**7.8,   # set None to disable baryonic feedback
    nM=256,          # halo-mass grid size; see "Tuning" below
)
# Pk_nl shape: (nz, nk)
```

### Layer 2 — `power(k, zs, CAMB_results, ...)` (high level)

Drop-in replacement for the original Python `hmcode.power`. Pass a
`camb.CAMBdata` and the wrapper extracts everything it needs.

```python
import numpy as np
import camb
from hmcode_py import power

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.4, ombh2=0.022, omch2=0.12, mnu=0.0, omk=0.0)
pars.InitPower.set_params(As=2.1e-9, ns=0.965)
pars.set_matter_power(redshifts=[3.0, 2.0, 1.0, 0.5, 0.0], kmax=200.0)

results = camb.get_results(pars)

k = np.logspace(-3.0, 1.0, 128)
zs = np.array([3.0, 2.0, 1.0, 0.5, 0.0])

Pk_nl = power(k, zs, results, T_AGN=10**7.8)   # shape (nz, nk)
```

This signature mirrors the original Python HMcode, so existing scripts
should work with a single import change.

### Tests

```bash
pytest tests/                 # all 9 tests; ~50 s once Julia is precompiled
pytest tests/test_smoke.py    # fastest sanity check
```

The suite covers:

- **`test_smoke.py`** — shape and finiteness with synthetic inputs.
- **`test_parity_julia.py`** — wrapper output vs an independent native-Julia
  path inside the same juliacall session. Tolerance `< 1e-10`. Catches
  marshalling, transpose, and kwarg bugs.
- **`test_parity_camb.py`** — `power(..., CAMB_results, ...)` vs the original
  Python HMcode at `HMcode-python/`. Tolerance `< 5e-3` (median ~2×10⁻⁵).

### Scripts: validation, comparison, and benchmarking

These scripts are all runnable from the repository root.

#### Basic speed profile

```bash
python scripts/profile_wrapper.py
```

What it reports:

- cold vs warm wrapper call time
- wrapper overhead vs direct native Julia call
- speedup vs original Python HMcode

#### nM convergence check

```bash
python scripts/check_nM.py
```

Sweeps `nM` and reports runtime plus relative error against `nM=1024`.

#### Recipe comparison: wrapped HMCode.jl vs original Python HMcode

Single cosmology plot:

```bash
python scripts/compare_recipes.py --output hmcode_recipe_comparison.png
```

Suite mode (4 cosmologies, includes `w0`, `wa`, `mnu`):

```bash
python scripts/compare_recipes.py --suite --output hmcode_recipe_suite_pct.png
```

Residual panel is in percentage:
`100 * (P_jl / P_py - 1)`.

#### Direct comparison: wrapped HMCode.jl vs CAMB HMCode

Baseline (no feedback, recommended):

```bash
python scripts/compare_with_camb_hmcode.py --suite --output hmcode_vs_camb_suite_pct.png
```

Feedback test (stable low-z range):

```bash
python scripts/compare_with_camb_hmcode.py --suite --T-AGN 63095734.448 --zs 1,0.5,0 --output hmcode_vs_camb_suite_TAGN_lowz_pct.png
```

Notes:

- `--zs` must be monotonically decreasing.
- Residual panels are percentages: `100 * (P_jl / P_camb - 1)`.

#### End-to-end speed vs number of redshifts

```bash
python scripts/benchmark_speed_vs_nz.py --nM 128 --repeats 3 --output benchmark_vs_nz_nM128.png
```

Benchmarks are run at `nz = 10, 30, 50, 100` and split into:

- CAMB runtime
- linear-table extraction runtime
- HMCode wrapper runtime
- total runtime

#### Combined sweep for selected nM values

```bash
python scripts/sweep_nm_camb_comparison_and_bench.py --nM-list 128,96,64 --bench-repeats 2 --out-prefix nm_sweep_128_96_64
```

Produces:

- CAMB-comparison accuracy table and plot vs `nM`
- benchmark table and plot vs `nM` and `nz`
- CSV outputs for both tables

---

## Tuning

| knob | default | notes |
|---|---|---|
| `nM` | 256 | halo-mass grid size. `nM=128` ≈ 0.7 % max error vs `nM=1024` and is ~2× faster — fine for emulator training. Don't go below `nM=64`. |
| `T_AGN` | `10**7.8` (Layer 1) / `None` (Layer 2) | AGN feedback temperature in K. `None` disables baryonic feedback (~2.5× faster). |
| `Mmin`, `Mmax` | `1.0`, `1e18` Msun/h | Halo-mass integration range. Defaults match the original. |

---

## Layout

```
src/hmcode_py/
├── __init__.py          # public API: HMcodeCosmology, hmcode_power_tabulated, power
├── _bridge.py           # juliacall init + dispatch
├── _interp.jl           # Julia-side interpolant builders
└── juliapkg.json        # pins HMcode.jl by URL + UUID + commit SHA
```

## License

MIT.
