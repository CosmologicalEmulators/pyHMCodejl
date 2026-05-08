"""CAMB-layer parity: hmcode_py.power(k, zs, CAMB_results) vs original Python hmcode.power.

Tolerance: rtol < 5e-3. This is a *cross-implementation* check, not a bit
parity check. The original Python HMcode and HMcode.jl differ in:
  - linear-growth / sigma_R interpolation grids
  - integration tolerances inside halo-model loops
  - small algorithmic ordering differences

so a few-tenths-of-a-percent agreement is the realistic target. This
matches the ~5e-3 tolerance used by HMcode.jl's own Julia<->Python tests.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

import hmcode_py

# Make the original Python HMcode importable.
_ORIG_PY = "/home/marcobonici/Desktop/work/CosmologicalEmulators/test_halofit/HMcode-python"
if _ORIG_PY not in sys.path:
    sys.path.insert(0, _ORIG_PY)

camb = pytest.importorskip("camb")
try:
    import hmcode as hmcode_orig
    import hmcode.camb_stuff as camb_stuff
except Exception as exc:  # pragma: no cover
    pytest.skip(f"original Python HMcode not importable: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Cosmologies to sweep. Kept small so the test wall-clock stays manageable
# (each cosmology runs hmcode three times: orig-Python, our wrapper).
# ---------------------------------------------------------------------------

CASES = [
    # (label, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa, T_AGN)
    ("LCDM_vanilla", 0.265, 0.049, 0.0, 0.674, 0.965, 0.8, 0.0, -1.0, 0.0, None),
    ("LCDM_mnu",     0.265, 0.049, 0.0, 0.674, 0.965, 0.8, 0.06, -1.0, 0.0, None),
    ("w0wa",         0.260, 0.050, 0.0, 0.700, 0.960, 0.81, 0.0, -0.95, -0.10, None),
    ("LCDM_AGN",     0.265, 0.049, 0.0, 0.674, 0.965, 0.8, 0.0, -1.0, 0.0, 10 ** 7.8),
]

K = np.logspace(-3.0, 1.0, 64)
ZS = np.array([3.0, 2.0, 1.0, 0.5, 0.0])  # decreasing, as the original requires


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_camb_layer_parity(case):
    label, Om_c, Om_b, Om_k, h, ns, s8, m_nu, w0, wa, T_AGN = case

    # CAMB setup mirrors the comparison harness in HMcode-python/comparisons/CAMB.py.
    log10_T_AGN = None if T_AGN is None else float(np.log10(T_AGN))
    _, results, _, _, _ = camb_stuff.run(
        ZS, Om_c, Om_b, Om_k, h, ns, s8,
        m_nu=m_nu, w=w0, wa=wa, log10_T_AGN=log10_T_AGN,
    )

    Pk_orig = hmcode_orig.power(K, ZS, results, T_AGN=T_AGN)
    Pk_jl = hmcode_py.power(K, ZS, results, T_AGN=T_AGN)

    assert Pk_orig.shape == Pk_jl.shape == (ZS.size, K.size)
    assert np.all(np.isfinite(Pk_jl))
    assert np.all(Pk_jl > 0)

    rel = np.abs(Pk_jl / Pk_orig - 1.0)
    max_rel = float(np.max(rel))
    median_rel = float(np.median(rel))

    # Diagnostic — surfaces in pytest output via -v / on failure.
    print(f"\n[{label}] T_AGN={T_AGN}  median rel err {median_rel:.3e}  "
          f"max rel err {max_rel:.3e}")

    assert max_rel < 5e-3, (
        f"[{label}] max rel err {max_rel:.3e} exceeds 5e-3; "
        f"location {np.unravel_index(np.argmax(rel), rel.shape)}"
    )
