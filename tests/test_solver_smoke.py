"""Smoke tests for the angular spectrum solver.

These keep the solver's most important invariants pinned without trying to
re-derive the full physics validation in `validate_*.py`. The validate
scripts are still the authoritative correctness checks; this file only
guards regressions in the public API.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Run validate-style scripts from project root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source,
)


# ---------------------------------------------------------------------------
# Shared bowl-source fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def bowl_setup():
    nX, nY, nT = 32, 32, 96
    dX = dY = 0.5e-3; dT = 5e-8
    c0, f0 = 1500.0, 1e6
    xaxis = (np.arange(nX) - nX // 2) * dX
    yaxis = (np.arange(nY) - nY // 2) * dY
    taxis = np.arange(nT) * dT
    field = make_bowl_source(
        xaxis, yaxis, taxis, f0=f0, c0=c0, p0=1e6,
        radius=6e-3, roc=12e-3, ncycles=3)
    return dict(field=field, dX=dX, dY=dY, dT=dT, c0=c0, f0=f0,
                nX=nX, nY=nY, nT=nT)


def _make_params(b, **overrides):
    base = dict(
        dX=b['dX'], dY=b['dY'], dT=b['dT'],
        c0=b['c0'], rho0=1000.0, beta=3.5, alpha0=-1, f0=b['f0'],
        propDist=5e-3, useSplitStep=True, useAdaptiveFiltering=True,
        dZmin=1e-3,
    )
    base.update(overrides)
    return SolverParams(**base)


# ---------------------------------------------------------------------------
# CPU vs GPU-reductions equivalence
# ---------------------------------------------------------------------------
def test_gpu_reductions_equivalence(bowl_setup):
    """GPU-side reductions must match host reductions on every output.

    The pI / pIloss arrays may differ at float32 ULP level due to
    summation order (jnp.sum vs np.sum). Other outputs are bit-exact.
    """
    b = bowl_setup
    cpu = angular_spectrum_solve(
        b['field'], _make_params(b, useGPUReductions=False), verbose=False)
    gpu = angular_spectrum_solve(
        b['field'], _make_params(b, useGPUReductions=True), verbose=False)

    field_c, pnp_c, ppp_c, pI_c, pIloss_c, zaxis_c, pax_c = cpu
    field_g, pnp_g, ppp_g, pI_g, pIloss_g, zaxis_g, pax_g = gpu

    np.testing.assert_array_equal(field_c, field_g)
    np.testing.assert_array_equal(pnp_c, pnp_g)
    np.testing.assert_array_equal(ppp_c, ppp_g)
    np.testing.assert_array_equal(pax_c, pax_g)
    np.testing.assert_array_equal(zaxis_c, zaxis_g)

    # ULP-tolerant on the float32 sums (different summation order between
    # jnp.sum and np.sum). Empirical envelope on small bowl runs is rtol~1e-3.
    np.testing.assert_allclose(pI_c, pI_g, rtol=1e-3)
    np.testing.assert_allclose(pIloss_c, pIloss_g, rtol=1e-2, atol=1.0)


# ---------------------------------------------------------------------------
# Output shapes and basic sanity
# ---------------------------------------------------------------------------
def test_basic_shapes(bowl_setup):
    b = bowl_setup
    out = angular_spectrum_solve(b['field'], _make_params(b), verbose=False)
    field, pnp, ppp, pI, pIloss, zaxis, pax = out

    assert field.shape == (b['nX'], b['nY'], b['nT'])
    nZ = zaxis.size
    assert pnp.shape == (b['nX'], b['nY'], nZ)
    assert ppp.shape == (b['nX'], b['nY'], nZ)
    assert pI.shape == (b['nX'], b['nY'], nZ)
    assert pIloss.shape == (b['nX'], b['nY'], nZ)
    assert pax.shape == (b['nT'], nZ)

    assert nZ > 0
    assert np.all(np.isfinite(field))
    assert pI.max() > 0


def test_pnp_nonpositive_ppp_nonnegative(bowl_setup):
    b = bowl_setup
    _, pnp, ppp, *_ = angular_spectrum_solve(
        b['field'], _make_params(b), verbose=False)
    # Negative-pressure peaks should be ≤ 0 and positive-pressure peaks ≥ 0
    assert pnp.max() <= 1e-3, f'pnp.max() = {pnp.max()}'
    assert ppp.min() >= -1e-3, f'ppp.min() = {ppp.min()}'


# ---------------------------------------------------------------------------
# TOF return-shape sanity
# ---------------------------------------------------------------------------
def test_tof_envelope_returns_8_tuple(bowl_setup):
    b = bowl_setup
    taxis = np.arange(b['nT']) * b['dT']
    out = angular_spectrum_solve(
        b['field'], _make_params(b), verbose=False,
        tof_env_ratio=1.0, taxis=taxis)
    assert len(out) == 8
    tof = out[7]
    assert tof.shape[0] == b['nX']
    assert tof.shape[1] == b['nY']
    # TOF must be non-negative finite values
    assert np.all(tof >= 0)
    assert np.all(np.isfinite(tof))


# ---------------------------------------------------------------------------
# Pre-flight pipeline runs and produces panels + LaTeX
# ---------------------------------------------------------------------------
def test_preflight_produces_outputs(tmp_path, bowl_setup):
    from preflight import preflight_report
    b = bowl_setup
    params = _make_params(b)
    result = preflight_report(
        b['field'], params, output_dir=str(tmp_path),
        scenario='smoke', compile_pdf=False, verbose=False)

    assert os.path.isfile(result['panels_path'])
    assert os.path.isfile(result['tex_path'])
    assert isinstance(result['metrics'], dict)
    assert 'max_Isppa_W_per_cm2' not in result['metrics']  # only emitted post-run
    assert 'f0' in result['metrics']
    assert isinstance(result['warnings'], list)
