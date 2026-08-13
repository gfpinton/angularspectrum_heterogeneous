"""Tests for 2-D (x, t) operation of the angular spectrum solver.

Two layers are covered:
  1. Degenerate nY = 1 operation of the 3-D solver — the singleton y axis
     must carry only the k_y = 0 mode (exact 2-D dispersion relation) and
     must not be damped or NaN'd by the boundary/filter constructions.
  2. The angular_spectrum_solve_2d wrapper — shape squeezing, screen
     promotion, and TOF plumbing.

The load-bearing physics check is equivalence against a y-invariant 3-D
run: with lateral boundary damping off, a y-uniform field stays exactly
y-uniform under periodic FFT propagation, and its dynamics are the 2-D
dynamics. The 3-D solver is the validated reference.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, angular_spectrum_solve_2d,
)


NX, NT = 64, 96
DX = DY = 0.5e-3
DT = 5e-8
C0, F0 = 1500.0, 1e6


@pytest.fixture(scope='module')
def line_source():
    """Cylindrically focused (line-focus) 2-D source, shape (NX, NT)."""
    xaxis = (np.arange(NX) - NX // 2) * DX
    taxis = np.arange(NT) * DT
    roc = 12e-3
    delay = (np.sqrt(roc ** 2 + xaxis ** 2) - roc) / C0
    field = np.zeros((NX, NT), dtype=np.float32)
    for i, d in enumerate(delay):
        if abs(xaxis[i]) <= 8e-3:
            t = taxis - 2e-6 + d
            env = np.exp(-(t * F0 / 3 * 2) ** 2)
            field[i, :] = 1e6 * env * np.sin(2 * np.pi * F0 * t)
    return field


def _make_params(**overrides):
    base = dict(
        dX=DX, dY=DY, dT=DT, c0=C0, rho0=1000.0, beta=3.5,
        alpha0=-1, f0=F0, propDist=5e-3, useSplitStep=True,
        useAdaptiveFiltering=True, dZmin=1e-3,
        diagnostic=False, preflight=False,
    )
    base.update(overrides)
    return SolverParams(**base)


# ---------------------------------------------------------------------------
# Physics: nY = 1 is exact 2-D (matches a y-invariant 3-D run)
# ---------------------------------------------------------------------------
def test_2d_matches_yinvariant_3d(line_source):
    # Odd nY: the legacy centered k-grid (linspace/(n-1), mean-subtracted)
    # only contains an exact k_y = 0 bin for odd n. With even n the DC-y
    # mode is propagated with a spurious half-bin tilt of pi/((n-1) dY) —
    # an O(1/n) grid-convention quirk that is negligible at production
    # sizes but visible at nY = 8. Odd n makes the comparison exact.
    nY = 9
    field_2d = line_source[:, np.newaxis, :]
    field_3d = np.repeat(field_2d, nY, axis=1)

    # Lateral damping off so the 3-D run stays exactly y-invariant
    params = _make_params(useBoundaryLayer=False)
    out2, pnp2, ppp2, *_ = angular_spectrum_solve(
        field_2d, params, verbose=False)
    out3, pnp3, ppp3, *_ = angular_spectrum_solve(
        field_3d, params, verbose=False)

    out2, out3 = np.asarray(out2), np.asarray(out3)
    assert np.all(np.isfinite(out2))

    # The 3-D run must have stayed y-invariant (pure ky=0 content)
    scale = np.abs(out3).max()
    assert np.abs(out3 - out3[:, :1, :]).max() < 1e-6 * scale

    # ...and the nY=1 run must reproduce it
    np.testing.assert_allclose(out2[:, 0, :], out3[:, nY // 2, :],
                               rtol=1e-5, atol=1e-5 * scale)
    np.testing.assert_allclose(np.asarray(pnp2)[:, 0, :],
                               np.asarray(pnp3)[:, nY // 2, :],
                               rtol=1e-5, atol=1e-5 * scale)


def test_2d_focuses(line_source):
    """A cylindrical lens must produce axial gain past the source plane."""
    _, _, ppp, *_ = angular_spectrum_solve_2d(
        line_source, _make_params(propDist=12e-3, useBoundaryLayer=False),
        verbose=False)
    ppp = np.asarray(ppp)
    assert ppp[NX // 2, 1:].max() > 1.3e6  # > 1.3x the 1 MPa source


# ---------------------------------------------------------------------------
# Boundary layer must be harmless (not fatal) on the singleton axis
# ---------------------------------------------------------------------------
def test_2d_default_boundary_runs(line_source):
    out, *_ = angular_spectrum_solve_2d(
        line_source, _make_params(useBoundaryLayer=True), verbose=False)
    out = np.asarray(out)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() > 1e3  # field not annihilated


@pytest.mark.parametrize('profile', ['quadratic', 'wendland'])
def test_2d_boundary_profiles(line_source, profile):
    out, *_ = angular_spectrum_solve_2d(
        line_source,
        _make_params(useBoundaryLayer=True, boundaryProfile=profile),
        verbose=False)
    assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Wrapper: shapes, exactness vs manual singleton run, input validation
# ---------------------------------------------------------------------------
def test_wrapper_matches_manual_singleton(line_source):
    params = _make_params()
    wrapped = angular_spectrum_solve_2d(line_source, params, verbose=False)
    manual = angular_spectrum_solve(
        line_source[:, np.newaxis, :], params, verbose=False)

    np.testing.assert_array_equal(np.asarray(wrapped[0]),
                                  np.asarray(manual[0])[:, 0, :])
    for w, m in zip(wrapped[1:5], manual[1:5]):
        np.testing.assert_array_equal(np.asarray(w), np.asarray(m)[:, 0, :])
    np.testing.assert_array_equal(wrapped[5], manual[5])  # zaxis
    np.testing.assert_array_equal(wrapped[6], manual[6])  # pax


def test_wrapper_shapes(line_source):
    out = angular_spectrum_solve_2d(line_source, _make_params(),
                                    verbose=False)
    field, pnp, ppp, pI, pIloss, zaxis, pax = out
    nZ = zaxis.size
    assert field.shape == (NX, NT)
    for arr in (pnp, ppp, pI, pIloss):
        assert arr.shape == (NX, nZ)
    assert pax.shape == (NT, nZ)
    assert nZ > 0


def test_wrapper_tof_8tuple(line_source):
    taxis = np.arange(NT) * DT
    out = angular_spectrum_solve_2d(
        line_source, _make_params(), verbose=False,
        tof_env_ratio=1.0, taxis=taxis)
    assert len(out) == 8
    tof = out[7]
    assert tof.shape[0] == NX
    assert tof.ndim == 2
    assert np.all(tof >= 0)
    assert np.all(np.isfinite(tof))


def test_wrapper_rejects_3d_input(line_source):
    with pytest.raises(ValueError, match='2-D'):
        angular_spectrum_solve_2d(
            line_source[:, np.newaxis, :], _make_params(), verbose=False)


# ---------------------------------------------------------------------------
# Phase screens with 1-D arrays through the 2-D wrapper
# ---------------------------------------------------------------------------
def test_wrapper_phase_screen_1d_kk(line_source):
    phase = np.zeros(NX, dtype=np.float32)
    amp = np.full(NX, 0.5, dtype=np.float32)
    screens = [(2.5e-3, phase, amp, 1.5)]

    def run(kk):
        params = _make_params(phaseScreens=screens, screenKKDispersion=kk)
        out, *_ = angular_spectrum_solve_2d(line_source, params,
                                            verbose=False)
        return np.asarray(out)

    out_off = run(False)
    out_on = run(True)
    assert np.all(np.isfinite(out_off))
    assert np.all(np.isfinite(out_on))
    # Screen must attenuate, and the K-K phase must reach the field
    unatten, *_ = angular_spectrum_solve_2d(line_source, _make_params(),
                                            verbose=False)
    assert np.abs(out_off).max() < np.abs(np.asarray(unatten)).max()
    assert np.max(np.abs(out_on - out_off)) > 0
