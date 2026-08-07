"""Tests for per-screen attenuation power laws and K-K dispersion.

Covers the (z, phase, amp, y) screen form and the screenKKDispersion
flag: legacy equivalence at y = 1, spectral scaling amp^((f/f0)^y),
Szabo phase consistency, per-pixel exponent maps, and end-to-end
plumbing through angular_spectrum_solve and skull_to_phase_screens.
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
    SolverParams, angular_spectrum_solve, make_bowl_source,
    _apply_phase_screen,
)


# ---------------------------------------------------------------------------
# Kernel-level fixtures
# ---------------------------------------------------------------------------
NX, NY, NT = 8, 8, 64
F0_BIN = 8.0  # f0 sits on rfft bin 8 of 33


@pytest.fixture(scope='module')
def rng_field():
    rng = np.random.default_rng(7)
    return rng.standard_normal((NX, NY, NT)).astype(np.float32)


def _rfft(a):
    return np.fft.rfft(np.asarray(a), axis=2)


def _screens(amp_value=0.7):
    phase = np.zeros((NX, NY), dtype=np.float32)
    amp = np.full((NX, NY), amp_value, dtype=np.float32)
    return phase, amp


# ---------------------------------------------------------------------------
# Legacy equivalence and power-law amplitude scaling
# ---------------------------------------------------------------------------
def test_pow1_matches_legacy(rng_field):
    phase, amp = _screens()
    legacy = np.asarray(_apply_phase_screen(rng_field, phase, F0_BIN, amp))
    pow1 = np.asarray(_apply_phase_screen(rng_field, phase, F0_BIN, amp,
                                          np.float32(1.0)))
    np.testing.assert_allclose(legacy, pow1, rtol=1e-5, atol=1e-6)


def test_power_law_spectral_scaling(rng_field):
    y = 1.5
    a = 0.7
    phase, amp = _screens(a)
    out = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(y))
    F_in = _rfft(rng_field)
    F_out = _rfft(out)
    fs = np.arange(F_in.shape[2]) / F0_BIN
    T = a ** (fs ** y)
    # Skip the Nyquist bin: irfft discards its imaginary part
    expected = F_in[:, :, :-1] * T[:-1]
    scale = np.abs(F_in).max()
    np.testing.assert_allclose(F_out[:, :, :-1], expected,
                               rtol=1e-4, atol=1e-4 * scale)


# ---------------------------------------------------------------------------
# Kramers-Kronig dispersion
# ---------------------------------------------------------------------------
def test_kk_preserves_magnitude(rng_field):
    phase, amp = _screens()
    off = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(1.5),
                              kk_dispersion=False)
    on = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(1.5),
                             kk_dispersion=True)
    F_off, F_on = _rfft(off), _rfft(on)
    scale = np.abs(F_off).max()
    np.testing.assert_allclose(np.abs(F_on[:, :, :-1]),
                               np.abs(F_off[:, :, :-1]),
                               rtol=1e-4, atol=1e-4 * scale)
    # ...but the fields themselves must differ (phase was added)
    assert np.max(np.abs(np.asarray(on) - np.asarray(off))) > 1e-4


def test_kk_y1_matches_szabo_phase(rng_field):
    a = 0.6
    phase, amp = _screens(a)
    out = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(1.0),
                              kk_dispersion=True)
    F_in = _rfft(rng_field)
    F_out = _rfft(out)
    n_freq = F_in.shape[2]
    fs = np.arange(n_freq) / F0_BIN
    A0 = -np.log(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = -(2.0 / np.pi) * A0 * fs * np.log(fs)
    phi[0] = 0.0
    T = (a ** fs) * np.exp(-1j * phi)
    expected = F_in[:, :, :-1] * T[:-1]
    scale = np.abs(F_in).max()
    np.testing.assert_allclose(F_out[:, :, :-1], expected,
                               rtol=1e-4, atol=1e-4 * scale)


def test_kk_default_y1_when_no_pow(rng_field):
    phase, amp = _screens()
    no_pow = _apply_phase_screen(rng_field, phase, F0_BIN, amp,
                                 kk_dispersion=True)
    pow1 = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(1.0),
                               kk_dispersion=True)
    np.testing.assert_allclose(np.asarray(no_pow), np.asarray(pow1),
                               rtol=1e-5, atol=1e-6)


def test_kk_y2_no_dispersion(rng_field):
    # tan(pi * 2/2) = 0: an f^2 law carries no Szabo dispersion
    phase, amp = _screens()
    off = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(2.0),
                              kk_dispersion=False)
    on = _apply_phase_screen(rng_field, phase, F0_BIN, amp, np.float32(2.0),
                             kk_dispersion=True)
    np.testing.assert_allclose(np.asarray(on), np.asarray(off),
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Per-pixel exponent maps
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kk', [False, True])
def test_per_pixel_pow_map(rng_field, kk):
    phase, amp = _screens()
    y_map = np.full((NX, NY), 1.2, dtype=np.float32)
    y_map[NX // 2:, :] = 1.8
    mapped = np.asarray(_apply_phase_screen(
        rng_field, phase, F0_BIN, amp, y_map, kk_dispersion=kk))
    lo = np.asarray(_apply_phase_screen(
        rng_field, phase, F0_BIN, amp, np.float32(1.2), kk_dispersion=kk))
    hi = np.asarray(_apply_phase_screen(
        rng_field, phase, F0_BIN, amp, np.float32(1.8), kk_dispersion=kk))
    np.testing.assert_allclose(mapped[:NX // 2], lo[:NX // 2],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mapped[NX // 2:], hi[NX // 2:],
                               rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: 4-tuple screens through the solver
# ---------------------------------------------------------------------------
def test_solver_4tuple_screen_kk_plumbing():
    nX, nY, nT = 32, 32, 96
    dX = dY = 0.5e-3
    dT = 5e-8
    c0, f0 = 1500.0, 1e6
    xaxis = (np.arange(nX) - nX // 2) * dX
    yaxis = (np.arange(nY) - nY // 2) * dY
    taxis = np.arange(nT) * dT
    field = make_bowl_source(
        xaxis, yaxis, taxis, f0=f0, c0=c0, p0=1e6,
        radius=6e-3, roc=12e-3, ncycles=3)

    phase = np.zeros((nX, nY), dtype=np.float32)
    amp = np.full((nX, nY), 0.5, dtype=np.float32)
    y_map = np.full((nX, nY), 1.5, dtype=np.float32)
    screens = [(2.5e-3, phase, amp, y_map)]

    def run(kk):
        params = SolverParams(
            dX=dX, dY=dY, dT=dT, c0=c0, rho0=1000.0, beta=3.5,
            alpha0=-1, f0=f0, propDist=5e-3, useSplitStep=True,
            useAdaptiveFiltering=True, dZmin=1e-3,
            diagnostic=False, preflight=False,
            phaseScreens=screens, screenKKDispersion=kk)
        out, *_ = angular_spectrum_solve(field, params, verbose=False)
        return np.asarray(out)

    out_off = run(False)
    out_on = run(True)
    assert np.all(np.isfinite(out_off))
    assert np.all(np.isfinite(out_on))
    # The K-K phase must actually reach the field
    assert np.max(np.abs(out_on - out_off)) > 0


# ---------------------------------------------------------------------------
# skull_to_phase_screens power-law emission
# ---------------------------------------------------------------------------
def test_skull_to_phase_screens_pow():
    from validate_transcranial import skull_to_phase_screens

    nx, ny, nz = 8, 8, 3
    c_map = np.full((nx, ny, nz), 1540.0, dtype=np.float32)
    c_map[2:6, 2:6, 1] = 2900.0  # bone blob in the middle slice
    dz = 1e-3
    c0, f0 = 1540.0, 1e6

    legacy = skull_to_phase_screens(c_map, dz, dz, c0, f0)
    assert all(len(s) == 3 for s in legacy)

    powed = skull_to_phase_screens(c_map, dz, dz, c0, f0,
                                   atten_pow_bone=1.2, atten_pow_tissue=1.0)
    assert all(len(s) == 4 for s in powed)
    for s in powed:
        y = s[3]
        assert y.shape == (nx, ny)
        assert np.all((y >= 1.0) & (y <= 1.2))

    # With y = 1 everywhere the amplitudes must reduce to the legacy ones
    unity = skull_to_phase_screens(c_map, dz, dz, c0, f0,
                                   atten_pow_bone=1.0, atten_pow_tissue=1.0)
    for s_leg, s_uni in zip(legacy, unity):
        np.testing.assert_allclose(s_uni[2], s_leg[2], rtol=1e-6)
        assert np.all(s_uni[3] == 1.0)
