"""
Kinematic-FWHM decomposition for shock-driven hyperfocusing.

Given a focal-plane time-history v(x, y, t) plus DT and the lateral
axis, computes derived rate-dependent fields and their lateral
concentration:

  a(x, y, t)  = ∂v/∂t          (particle acceleration, m/s²)
  ε̇_x        = ∂v/∂x           (strain rate in x, 1/s)
  ε̇_y        = ∂v/∂y           (strain rate in y, 1/s)
  |ε̇|        = √(ε̇_x² + ε̇_y²)  (strain-rate magnitude)
  γ(x, y, t) = ∫ ε̇_mag dt      (running strain time-integral, 1)

For each quantity reports at y=0:
  RMS-in-time profile      ⟨q²(x,y=0)⟩_t^(1/2)
  Time-peak profile        max_t |q(x,y=0,t)|
  Lateral FWHM             (linear-interp half-max crossings)
  FWHM ratio to pI         (the science number)

This module has no external dependencies beyond numpy/matplotlib.
"""
from __future__ import annotations
import json
import os
import warnings
import numpy as np


def fwhm(profile: np.ndarray, x: np.ndarray) -> float:
    """Linear-interp FWHM. Returns nan if profile never crosses half-max."""
    p = np.asarray(profile, dtype=np.float64)
    if p.max() <= 0:
        return float('nan')
    half = p.max() / 2.0
    above = np.where(p >= half)[0]
    if len(above) < 2:
        return float('nan')
    i_lo, i_hi = above[0], above[-1]

    def cross(i, side):
        if side == 'left' and i > 0:
            x0, x1, y0, y1 = x[i-1], x[i], p[i-1], p[i]
        elif side == 'right' and i < len(x) - 1:
            x0, x1, y0, y1 = x[i], x[i+1], p[i], p[i+1]
        else:
            return float(x[i])
        if y1 == y0:
            return float(x[i])
        return float(x0 + (half - y0) * (x1 - x0) / (y1 - y0))

    return cross(i_hi, 'right') - cross(i_lo, 'left')


def kinematic_fields(field: np.ndarray, dt: float, dx: float):
    """Return dict of derived kinematic fields, all shape (nX, nY, nT).

    Uses central differences via np.gradient (2nd-order accurate, falls
    back to one-sided at edges). All inputs/outputs are float32 to keep
    memory bounded.

    Quantities:
      v       — particle velocity (input, shear component for cubic regime)
      a       — particle acceleration ∂v/∂t (m/s²)
      eps_x   — strain rate component ∂v/∂x (1/s)
      eps_y   — strain rate component ∂v/∂y (1/s)
      eps_mag — strain rate magnitude √(ε̇_x²+ε̇_y²)
      u       — particle displacement ∫v·dt (m)
      gamma_x — shear strain component ∂u/∂x (1)
      gamma_y — shear strain component ∂u/∂y (1)
      gamma_vm — von Mises shear strain magnitude √(γ_x²+γ_y²) (1)
                 (rotationally-invariant scalar measure of lateral
                 deformation, in our 2D focal-plane reduction)
    """
    v = field.astype(np.float32, copy=False)
    a = np.gradient(v, dt, axis=2).astype(np.float32)
    eps_x = np.gradient(v, dx, axis=0).astype(np.float32)
    eps_y = np.gradient(v, dx, axis=1).astype(np.float32)
    eps_mag = np.sqrt(eps_x**2 + eps_y**2).astype(np.float32)
    # Displacement u = ∫v dt — cumulative trapezoid time-integral
    u = (np.cumsum(v, axis=2) * dt).astype(np.float32)
    # Subtract t=0 reference (already 0 since cumsum starts at 0)
    gamma_x = np.gradient(u, dx, axis=0).astype(np.float32)
    gamma_y = np.gradient(u, dx, axis=1).astype(np.float32)
    gamma_vm = np.sqrt(gamma_x**2 + gamma_y**2).astype(np.float32)
    return dict(v=v, a=a, eps_x=eps_x, eps_y=eps_y, eps_mag=eps_mag,
                u=u, gamma_x=gamma_x, gamma_y=gamma_y, gamma_vm=gamma_vm)


def lateral_2d_maps(field: np.ndarray, kf: dict):
    """Return dict of 2D (x,y) maps for FWHM and area-above-threshold.

    Critical: ALL maps here are dimensionally quadratic in v(x,y,t), so
    their lateral FWHMs / half-max areas can be compared directly to
    pI = ∫v²dt. Using a linear-in-v quantity (e.g. √⟨a²⟩, max_t|a|) as
    a map and FWHM-comparing it to pI introduces a spurious √2 broaden-
    ing factor that has nothing to do with shock physics.

    Reductions:
      *_meansq:  ⟨·²⟩_t  = mean-square in time (envelope-squared
                  proxy, suppresses windowing artifacts)
      *_peaksq:  max_t(·²) = (peak-in-time)² (sensitive to shock-front
                  spike; what the pivot doc calls "max_t |a|" but
                  squared for dimensional matching with pI)
    """
    return {
        'pI':              np.sum(field.astype(np.float64)**2, axis=-1),
        'v_peaksq':        np.max(field.astype(np.float64)**2, axis=-1),
        'a_meansq':        np.mean(kf['a']**2, axis=-1),
        'a_peaksq':        np.max(kf['a']**2, axis=-1),
        'eps_meansq':      np.mean(kf['eps_mag']**2, axis=-1),
        'eps_peaksq':      np.max(kf['eps_mag']**2, axis=-1),
        'gamma_vm_meansq': np.mean(kf['gamma_vm']**2, axis=-1),
        'gamma_vm_peaksq': np.max(kf['gamma_vm']**2, axis=-1),
    }


def lateral_profiles(field: np.ndarray, kf: dict, maps: dict):
    """Lateral profiles at y=0 — slice of the dim-matched 2D maps."""
    nY = field.shape[1]
    cy = nY // 2
    return {k: m[:, cy] for k, m in maps.items()}


def diagnostics(field: np.ndarray, dt: float, f0: float, n_max: int = 11):
    """Edge-steepness + odd-harmonic fraction (on-axis), same definition
    as the parent lossy-bowl validate script."""
    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    trace = field[cx, cy, :].astype(np.float64)
    spec = np.abs(np.fft.rfft(trace))
    freqs = np.fft.rfftfreq(nT, dt)
    bin_w = 0.4 * f0
    e_n = np.zeros(n_max)
    for n in range(1, n_max):
        m = (freqs >= n*f0 - bin_w) & (freqs <= n*f0 + bin_w)
        e_n[n] = float(np.sum(spec[m]**2))
    odd_high = e_n[3] + e_n[5] + e_n[7] + e_n[9]
    odd_frac = float(odd_high / max(e_n[1:n_max].sum(), 1e-30))
    dvdt = np.gradient(trace, dt)
    omega0 = 2 * np.pi * f0
    v_peak = float(max(trace.max(), -trace.min()))
    edge_steepness = float(np.max(np.abs(dvdt)) / max(v_peak * omega0, 1e-30))
    return dict(odd_frac=odd_frac, edge_steepness=edge_steepness,
                v_peak=v_peak)


def area_above_threshold(map_2d: np.ndarray,
                         dx_m: float, dy_m: float,
                         threshold_fraction: float = 0.5) -> float:
    """Focal-plane area where the 2D map exceeds threshold_fraction · peak.

    For a centred Gaussian I(r)=I₀·exp(-r²/σ²) at half-max this gives
    π·σ²·ln 2 = π·FWHM²/(4·ln 2) ≈ 1.133·FWHM². For annular/ring
    patterns it gives the area of the ring region — handles bimodal
    profiles unambiguously.
    """
    if map_2d.max() <= 0:
        return float('nan')
    threshold = threshold_fraction * map_2d.max()
    mask = map_2d >= threshold
    return float(np.sum(mask) * dx_m * dy_m)


def summarize(field, dt, dx, f0, xaxis):
    """One-call analysis. Returns a dict suitable for JSON dump + plotting."""
    kf = kinematic_fields(field, dt, dx)
    maps = lateral_2d_maps(field, kf)
    profiles = lateral_profiles(field, kf, maps)
    diag = diagnostics(field, dt, f0)

    x_mm = xaxis * 1e3
    fwhms = {k: fwhm(p, x_mm) for k, p in profiles.items()}
    pI_fwhm = fwhms['pI']
    ratios = {f'{k}/pI': (fwhms[k] / pI_fwhm) if (pI_fwhm and pI_fwhm == pI_fwhm)
              else float('nan')
              for k in fwhms if k != 'pI'}

    # Area-above-threshold: scale-free, robust to bimodal profiles.
    areas = {k: area_above_threshold(m, dx, dx) for k, m in maps.items()}
    area_pI = areas['pI']
    area_ratios = {
        f'{k}/pI':
            (areas[k] / area_pI) if (area_pI == area_pI and area_pI > 0)
            else float('nan')
        for k in areas if k != 'pI'
    }

    # NaN FWHMs/areas (degenerate profile or <2 half-max crossings) are
    # kept in the output so JSON schemas stay stable, but warn so a failed
    # extraction can't slip silently into downstream sweeps.
    bad = sorted([f'fwhm:{k}' for k, v in fwhms.items() if v != v]
                 + [f'area:{k}' for k, v in areas.items() if v != v])
    if bad:
        warnings.warn(f'summarize(): NaN metrics for {", ".join(bad)}',
                      RuntimeWarning, stacklevel=2)

    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    on_axis = {
        'v_peak':         float(np.max(np.abs(field[cx, cy, :]))),
        'a_peak':         float(np.max(np.abs(kf['a'][cx, cy, :]))),
        'a_rms':          float(np.sqrt(np.mean(kf['a'][cx, cy, :]**2))),
        'eps_peak':       float(np.max(kf['eps_mag'][cx, cy, :])),
        'eps_rms':        float(np.sqrt(np.mean(kf['eps_mag'][cx, cy, :]**2))),
        'u_peak':         float(np.max(np.abs(kf['u'][cx, cy, :]))),
        'gamma_vm_peak':  float(np.max(kf['gamma_vm'][cx, cy, :])),
    }
    return dict(fwhms_mm=fwhms, ratios=ratios,
                areas_mm2={k: v * 1e6 for k, v in areas.items()},
                area_ratios=area_ratios,
                on_axis=on_axis,
                diag=diag, profiles=profiles, maps=maps, x_mm=x_mm)
