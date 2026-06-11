"""Physics regression tests.

Fast (seconds-scale) pins on the physics paths the smoke tests don't
touch: the cubic-Burgers KT kernel (including the sonic-point detector
on the compound-wave Riemann problem), quadratic-Burgers harmonic
generation, and the per-pixel attenuation-loss tracking of the
split-step KT march. The full-resolution authoritative checks remain
the validate_*.py scripts; tolerances here are ~1.5x the values
measured on the coarse grids used, so they fail on genuine regressions
(the pre-fix sonic-detector error was ~2x larger) without being flaky.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax.numpy as jnp

from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source, _kt_flux_cubic,
)


# ---------------------------------------------------------------------------
# Cubic-Burgers Riemann problems (KT scheme)
# ---------------------------------------------------------------------------
# Coarse version of validate_cubic_riemann.py: 801 samples instead of
# 4001. N3=1.0 calibrates the KT kernel to the canonical PDE
# du/dz = -d(-u^3/3)/dtau (see the march() docstring in that script).

def _riemann_march(u_left, u_right, nT=801):
    tau = np.linspace(-2.0, 2.0, nT)
    dT = tau[1] - tau[0]
    dZ = 0.4 * dT  # CFL-safe for max|u| <= 1
    n_steps = int(round(1.0 / dZ))
    z = n_steps * dZ
    field = jnp.array(
        np.where(tau < 0.0, u_left, u_right).astype(np.float32)[None, None, :])
    for _ in range(n_steps):
        field = _kt_flux_cubic(field, 1.0, dZ, dT)
    return tau, z, np.asarray(field)[0, 0, :]


def _l2_interior(u, exact):
    valid = slice(50, -50)  # exclude boundary cells, as in the validate script
    return float(np.sqrt(np.mean((u[valid] - exact[valid]) ** 2)))


def test_cubic_kt_shock():
    """Test A: u_L=0, u_R=+1 — Lax shock travelling at s = -1/3."""
    tau, z, u = _riemann_march(0.0, 1.0)
    exact = np.where(tau < -z / 3.0, 0.0, 1.0)
    assert np.all(np.isfinite(u))
    err = _l2_interior(u, exact)
    assert err < 0.05, f'KT cubic shock L2 error {err:.4f} (measured baseline 0.025)'


def test_cubic_kt_compound_wave_sonic_point():
    """Test C: u_L=+1, u_R=-1 — rarefaction + sonic shock through u=0.

    This crosses the sonic point and exercises the KT sonic-face
    detector (commit bd709e7); before that fix the full-grid error was
    ~3x larger, so the bound below catches a regression.
    """
    tau, z, u = _riemann_march(1.0, -1.0)
    exact = np.empty_like(tau)
    exact[tau < -z] = 1.0
    fan = (tau >= -z) & (tau <= -z / 4.0)
    exact[fan] = np.sqrt(-tau[fan] / z)
    exact[tau > -z / 4.0] = -1.0
    assert np.all(np.isfinite(u))
    err = _l2_interior(u, exact)
    assert err < 0.06, f'KT compound-wave L2 error {err:.4f} (measured baseline 0.038)'


# ---------------------------------------------------------------------------
# Quadratic-Burgers harmonic generation (full solver)
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def bowl_setup():
    nX, nY, nT = 32, 32, 96
    dX = 0.5e-3
    dT = 5e-8
    c0, f0 = 1500.0, 1e6
    xaxis = (np.arange(nX) - nX // 2) * dX
    taxis = np.arange(nT) * dT
    # p0 = 5 MPa so 5 mm of propagation produces clearly measurable
    # second-harmonic growth on this coarse grid.
    field = make_bowl_source(
        xaxis, xaxis, taxis, f0=f0, c0=c0, p0=5e6,
        radius=6e-3, roc=12e-3, ncycles=3)
    return dict(field=field, dX=dX, dT=dT, c0=c0, f0=f0,
                nX=nX, nY=nY, nT=nT)


def _make_params(b, **overrides):
    base = dict(
        dX=b['dX'], dY=b['dX'], dT=b['dT'],
        c0=b['c0'], rho0=1000.0, beta=3.5, alpha0=-1, f0=b['f0'],
        propDist=5e-3, useSplitStep=True, useAdaptiveFiltering=True,
        dZmin=1e-3, diagnostic=False, preflight=False,
    )
    base.update(overrides)
    return SolverParams(**base)


def _second_harmonic_ratio(trace, dT, f0):
    spec = np.abs(np.fft.rfft(trace))
    freqs = np.fft.rfftfreq(trace.size, dT)
    i1 = int(np.argmin(np.abs(freqs - f0)))
    i2 = int(np.argmin(np.abs(freqs - 2 * f0)))
    a1 = spec[max(i1 - 2, 0):i1 + 3].max()
    a2 = spec[max(i2 - 2, 0):i2 + 3].max()
    return a2 / a1


def test_nonlinearity_generates_second_harmonic(bowl_setup):
    """beta>0 must grow on-axis second-harmonic content vs beta=0.

    Guards the quadratic Burgers flux path: a no-op nonlinear step (or a
    sign/scaling error) collapses the measured ratio gap. Measured on
    this grid: linear 0.083, nonlinear 0.112 (ratio 1.36).
    """
    b = bowl_setup
    out_lin = angular_spectrum_solve(
        b['field'], _make_params(b, beta=0.0), verbose=False)
    out_nl = angular_spectrum_solve(
        b['field'], _make_params(b, beta=3.5), verbose=False)
    cx, cy = b['nX'] // 2, b['nY'] // 2
    h_lin = _second_harmonic_ratio(out_lin[0][cx, cy, :], b['dT'], b['f0'])
    h_nl = _second_harmonic_ratio(out_nl[0][cx, cy, :], b['dT'], b['f0'])
    assert h_nl > 1.2 * h_lin, (
        f'second-harmonic ratio nonlinear={h_nl:.4f} vs linear={h_lin:.4f}; '
        'expected >=1.2x growth')
    # And the fields themselves must differ by more than float noise.
    diff = np.max(np.abs(out_nl[0] - out_lin[0]))
    assert diff > 1e-2 * np.max(np.abs(out_nl[0]))


# ---------------------------------------------------------------------------
# Per-pixel attenuation-loss tracking (split-step KT march)
# ---------------------------------------------------------------------------
def test_atten_loss_tracking(bowl_setup):
    """useAttenLoss=True on the supported split-KT path (commit 94664aa).

    The tracked loss must be non-negative, finite, strictly positive in
    total, and monotonic in the attenuation coefficient.
    """
    b = bowl_setup

    def run(alpha0):
        params = _make_params(
            b, alpha0=alpha0, attenPow=1.0,
            fluxScheme='kt', useAttenLoss=True)
        return angular_spectrum_solve(b['field'], params, verbose=False)[4]

    loss_lo = run(0.5)
    loss_hi = run(2.0)
    for loss in (loss_lo, loss_hi):
        assert np.all(np.isfinite(loss))
        assert loss.min() >= 0.0, 'tracked attenuation loss must be >= 0 per pixel'
    assert loss_lo.sum() > 0.0
    assert loss_hi.sum() > loss_lo.sum(), (
        'higher alpha0 must record more attenuation loss '
        f'({loss_hi.sum():.3e} vs {loss_lo.sum():.3e})')
