"""
Retarded-frame -> laboratory-frame de-shear for the transient
angular-spectrum solver.

Physics
-------
The transient angular-spectrum march advances in ``z`` and stores the field
in a **retarded** time frame

    tau = t - z / c0

i.e. the bulk ``c0`` transit delay is subtracted at every march plane.  A
propagating pulse therefore looks nearly *frozen* in ``z`` (its retarded-time
peak sits at ~constant ``tau``) instead of sweeping across the domain.  To
visualise the field in conventional **laboratory** space-time we undo that
shear, sampling — for every depth ``z`` — the retarded-time axis at
``tau = t* - z / c0``:

    p_lab(x, z, t*) = p_ret(x, z, tau = t* - z / c0)

so a lab-frame snapshot at absolute time ``t*`` shows the pulse physically
propagating at ``c0`` from the transducer (small ``z``) through the focus and
diverging past it.  The field is **real signed** pressure ``p(x, y, t)``
(the solver reconstructs it via ``irfft`` / ``real``), so lab-frame imagery
uses a diverging colormap with a symmetric colour scale (red = compression,
blue = rarefaction).

This module is deliberately dependency-light (``numpy`` + ``matplotlib``);
no JAX is required for the transform.  It operates on an xz field-history
cube of shape ``(nX, nT_ret, nZ)`` — the signed field on the mid-``y`` plane
captured once per march step, with the retarded-time axis (``tau_axis``)
stride-subsampled to keep the cube small.

Public API
----------
``retarded_to_lab_snapshot(hist, tau_axis, zaxis, c0, t_star)``
    De-shear a single absolute time ``t*`` -> ``(nX, nZ)`` lab-frame slab.
``retarded_to_lab_cube(hist, tau_axis, zaxis, c0, t_lab)``
    De-shear a whole absolute-time grid -> ``(nX, nT_lab, nZ)`` lab cube.
``iter_lab_snapshots(hist, tau_axis, zaxis, c0, t_lab)``
    Generator yielding ``(t_star, slab)`` per absolute time (memory-light).
``pulse_retarded_support(hist, tau_axis, frac)``
    Retarded-time support ``[tau_a, tau_b]`` of the pulse.
``lab_time_grid(hist, tau_axis, zaxis, c0, nframes, frac)``
    Absolute-time grid spanning the pulse as it crosses the domain.
``linear_arrival_check(hist, tau_axis, zaxis, c0, energy_frac)``
    Quantitative validation: retarded peak slope (~0) and lab arrival slope
    (~1/c0) of ``tau_peak(z)``.
``plot_labframe_montage(...)`` / ``plot_labframe_snapshot(...)``
    Matplotlib rendering helpers (signed pressure, diverging colormap).
``plot_labframe_validation(...)``
    Composite validation figure (snapshot montage + arrival check) saved to
    a PNG; returns the ``linear_arrival_check`` dict.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


__all__ = [
    'pulse_retarded_support',
    'lab_time_grid',
    'retarded_to_lab_snapshot',
    'retarded_to_lab_cube',
    'iter_lab_snapshots',
    'linear_arrival_check',
    'plot_labframe_snapshot',
    'plot_labframe_montage',
    'plot_labframe_validation',
]


# ---------------------------------------------------------------------------
# Pulse support / lab-frame time window
# ---------------------------------------------------------------------------
def _is_signed(hist: np.ndarray) -> bool:
    """True if the cube holds signed pressure (vs a legacy |p| envelope)."""
    amax = float(np.abs(hist).max()) if hist.size else 0.0
    return bool(amax > 0 and hist.min() < -1e-3 * amax)


def pulse_retarded_support(hist: np.ndarray,
                           tau_axis: np.ndarray,
                           frac: float = 0.005) -> Tuple[float, float]:
    """Retarded-time support of the pulse.

    Returns ``(tau_a, tau_b)`` — the first/last retarded times whose summed
    (over x and z) energy exceeds ``frac`` of the peak.  Used to bound the
    lab-frame absolute-time window so snapshots only span the interval that
    actually contains the wavefront.
    """
    g = (np.asarray(hist, dtype=np.float64) ** 2).sum(axis=(0, 2))
    if not np.any(g):
        return float(tau_axis[0]), float(tau_axis[-1])
    supp = np.where(g > frac * g.max())[0]
    return float(tau_axis[supp[0]]), float(tau_axis[supp[-1]])


def lab_time_grid(hist: np.ndarray,
                  tau_axis: np.ndarray,
                  zaxis: np.ndarray,
                  c0: float,
                  nframes: int = 200,
                  frac: float = 0.005) -> np.ndarray:
    """Absolute-time grid spanning the pulse as it crosses the domain.

    The wavefront first appears at ``zaxis[0]/c0 + tau_a`` (pulse leading
    edge at the shallowest captured depth) and leaves at
    ``zaxis[-1]/c0 + tau_b`` (trailing edge at the deepest depth), where
    ``[tau_a, tau_b]`` is :func:`pulse_retarded_support`.
    """
    tau_a, tau_b = pulse_retarded_support(hist, tau_axis, frac=frac)
    t_lo = float(zaxis[0]) / c0 + tau_a
    t_hi = float(zaxis[-1]) / c0 + tau_b
    return np.linspace(t_lo, t_hi, int(nframes))


# ---------------------------------------------------------------------------
# Core de-shear transform
# ---------------------------------------------------------------------------
def retarded_to_lab_snapshot(hist: np.ndarray,
                             tau_axis: np.ndarray,
                             zaxis: np.ndarray,
                             c0: float,
                             t_star: float) -> np.ndarray:
    """De-shear one absolute time ``t*`` into a lab-frame ``(nX, nZ)`` slab.

    For every depth column ``z`` the retarded-time axis is sampled (with
    linear fractional-index interpolation) at ``tau = t* - z/c0``:

        f      = (t* - z/c0 - tau_axis[0]) / dTau
        i0     = floor(f);  frac = f - i0
        slab[:, z] = hist[:, i0, z]*(1-frac) + hist[:, i0+1, z]*frac

    Columns whose required ``tau`` falls outside the captured retarded
    window are zeroed (the pulse has not yet reached / has already left that
    depth at time ``t*``).
    """
    hist = np.asarray(hist)
    nX, nT, nZ = hist.shape
    tau0 = float(tau_axis[0])
    dTau = float(tau_axis[1] - tau_axis[0])
    zidx = np.arange(nZ)

    f = (t_star - np.asarray(zaxis, dtype=np.float64) / c0 - tau0) / dTau
    i0 = np.floor(f).astype(int)
    frac = f - i0
    valid = (i0 >= 0) & (i0 < nT - 1)
    i0c = np.clip(i0, 0, nT - 2)

    slab = (hist[:, i0c, zidx] * (1.0 - frac)
            + hist[:, i0c + 1, zidx] * frac)
    slab[:, ~valid] = 0.0
    return slab


def iter_lab_snapshots(hist: np.ndarray,
                       tau_axis: np.ndarray,
                       zaxis: np.ndarray,
                       c0: float,
                       t_lab: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield ``(t_star, slab)`` for each absolute time in ``t_lab``.

    Memory-light alternative to :func:`retarded_to_lab_cube` for movie
    rendering — never materialises the full lab cube.
    """
    for t_star in np.asarray(t_lab, dtype=np.float64):
        yield float(t_star), retarded_to_lab_snapshot(
            hist, tau_axis, zaxis, c0, float(t_star))


def retarded_to_lab_cube(hist: np.ndarray,
                         tau_axis: np.ndarray,
                         zaxis: np.ndarray,
                         c0: float,
                         t_lab: np.ndarray) -> np.ndarray:
    """De-shear the whole cube onto an absolute-time grid.

    Returns a lab-frame cube of shape ``(nX, nT_lab, nZ)`` where
    ``nT_lab = len(t_lab)`` and ``cube[:, k, :] = p_lab(x, z, t_lab[k])``.
    """
    hist = np.asarray(hist)
    nX, _, nZ = hist.shape
    t_lab = np.asarray(t_lab, dtype=np.float64)
    cube = np.empty((nX, t_lab.size, nZ), dtype=hist.dtype)
    for k, t_star in enumerate(t_lab):
        cube[:, k, :] = retarded_to_lab_snapshot(
            hist, tau_axis, zaxis, c0, float(t_star))
    return cube


# ---------------------------------------------------------------------------
# Quantitative validation: retarded vs lab arrival slope
# ---------------------------------------------------------------------------
def _fit_slope(z: np.ndarray, t: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares slope/intercept of ``t`` vs ``z`` and its R^2."""
    if z.size < 2:
        return float('nan'), float('nan'), float('nan')
    slope, intercept = np.polyfit(z, t, 1)
    resid = t - (slope * z + intercept)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), float(r2)


def linear_arrival_check(hist: np.ndarray,
                         tau_axis: np.ndarray,
                         zaxis: np.ndarray,
                         c0: float,
                         energy_frac: float = 0.1) -> dict:
    """Validate the retarded->lab shear against ``t = tau + z/c0``.

    For every depth ``z`` the pulse's retarded-time peak

        tau_peak(z) = argmax_tau  sum_x p(x, z, tau)^2

    is extracted (only over depths whose peak energy exceeds
    ``energy_frac`` of the global maximum, to avoid noise at the shallow /
    post-focus tails).  Two straight-line fits are reported:

    * ``slope_ret`` — ``tau_peak(z)`` vs ``z``.  The bulk delay has been
      removed, so this should be ~0 (pulse frozen in the retarded frame).
    * ``slope_lab`` — absolute arrival ``tau_peak(z) + z/c0`` vs ``z``.
      This should equal ``1/c0`` (the pulse physically travels at ``c0``).

    Slopes are returned both in SI (s/m) and in the intuitive us/mm, next
    to the physical ``1/c0`` for direct comparison.
    """
    hist = np.asarray(hist, dtype=np.float64)
    tau_axis = np.asarray(tau_axis, dtype=np.float64)
    zaxis = np.asarray(zaxis, dtype=np.float64)

    E = (hist ** 2).sum(axis=0)            # (nT_ret, nZ)  energy vs (tau, z)
    Ez = E.max(axis=0)                     # peak energy per depth
    tau_peak = tau_axis[np.argmax(E, axis=0)]
    t_abs = tau_peak + zaxis / c0

    sel = Ez > energy_frac * (Ez.max() if Ez.max() > 0 else 1.0)
    z_sel = zaxis[sel]
    slope_ret, _, r2_ret = _fit_slope(z_sel, tau_peak[sel])
    slope_lab, _, r2_lab = _fit_slope(z_sel, t_abs[sel])

    inv_c0 = 1.0 / c0
    to_us_mm = 1e3                          # s/m -> us/mm
    return dict(
        c0=float(c0),
        inv_c0_s_per_m=float(inv_c0),
        inv_c0_us_per_mm=float(inv_c0 * to_us_mm),
        slope_ret_s_per_m=slope_ret,
        slope_ret_us_per_mm=slope_ret * to_us_mm,
        slope_lab_s_per_m=slope_lab,
        slope_lab_us_per_mm=slope_lab * to_us_mm,
        rel_err_lab=(abs(slope_lab - inv_c0) / inv_c0
                     if np.isfinite(slope_lab) and inv_c0 > 0 else float('nan')),
        r2_ret=r2_ret,
        r2_lab=r2_lab,
        n_z_used=int(z_sel.size),
        n_z_total=int(zaxis.size),
        # arrays for plotting
        zaxis=zaxis,
        tau_peak=tau_peak,
        t_abs=t_abs,
        sel=sel,
    )


# ---------------------------------------------------------------------------
# Matplotlib rendering helpers
# ---------------------------------------------------------------------------
def _extent_and_axes(hist, xaxis, zaxis):
    nX, _, nZ = hist.shape
    if xaxis is None:
        xaxis = (np.arange(nX) - nX // 2)
        x_mm = xaxis.astype(float)
    else:
        x_mm = np.asarray(xaxis, dtype=float) * 1e3
    z_mm = np.asarray(zaxis, dtype=float) * 1e3
    extent = [z_mm[0], z_mm[-1], x_mm[0], x_mm[-1]]
    return extent, x_mm, z_mm


def _color_scale(hist, signed):
    vmax = float(np.percentile(np.abs(hist), 99.9)) or 1.0
    if signed:
        return 'RdBu_r', -vmax, vmax
    return 'inferno', 0.0, vmax


def plot_labframe_snapshot(hist: np.ndarray,
                           tau_axis: np.ndarray,
                           zaxis: np.ndarray,
                           c0: float,
                           t_star: float,
                           ax=None,
                           xaxis: Optional[np.ndarray] = None,
                           vmax: Optional[float] = None,
                           add_colorbar: bool = True,
                           title: Optional[str] = None):
    """Render a single lab-frame snapshot at absolute time ``t*``.

    Signed pressure on a diverging (``RdBu_r``), symmetric colour scale;
    horizontal axis = depth ``z`` (mm), vertical axis = lateral ``x`` (mm).
    Returns the matplotlib ``AxesImage``.
    """
    hist = np.asarray(hist)
    signed = _is_signed(hist)
    cmap, vmin, vmax_auto = _color_scale(hist, signed)
    if vmax is not None:
        vmax_auto = float(vmax)
        vmin = -vmax_auto if signed else 0.0
    extent, _, _ = _extent_and_axes(hist, xaxis, zaxis)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    slab = retarded_to_lab_snapshot(hist, tau_axis, zaxis, c0, t_star)
    im = ax.imshow(slab, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=vmax_auto)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    if title is None:
        title = f't = {t_star*1e6:6.2f} µs'
    ax.set_title(title, fontsize=9)
    if add_colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02,
                                label='p (Pa)' if signed else '|p| (Pa)')
        cb.outline.set_visible(False)
    return im


def plot_labframe_montage(hist: np.ndarray,
                          tau_axis: np.ndarray,
                          zaxis: np.ndarray,
                          c0: float,
                          ntiles: int = 5,
                          t_lab: Optional[np.ndarray] = None,
                          xaxis: Optional[np.ndarray] = None,
                          axes=None,
                          fig=None):
    """Render a row of ``ntiles`` lab-frame snapshots sharing one colour scale.

    The snapshot times span the pulse from just after emission (shallow
    ``z``) through the focus to past it (deep ``z``).  Returns
    ``(fig, axes, t_shots)``.
    """
    hist = np.asarray(hist)
    signed = _is_signed(hist)
    _, _, vmax = _color_scale(hist, signed)
    if t_lab is None:
        t_lab = lab_time_grid(hist, tau_axis, zaxis, c0, nframes=ntiles)
    t_shots = np.asarray(t_lab, dtype=np.float64)

    if axes is None:
        fig, axes = plt.subplots(1, ntiles, figsize=(3.0 * ntiles, 3.2),
                                 constrained_layout=True)
    axes = np.atleast_1d(axes)
    im = None
    for k, (ax, t_star) in enumerate(zip(axes, t_shots)):
        im = plot_labframe_snapshot(
            hist, tau_axis, zaxis, c0, float(t_star), ax=ax,
            xaxis=xaxis, vmax=vmax, add_colorbar=False)
        if k > 0:
            ax.set_ylabel('')
    if im is not None and fig is not None:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01,
                          label='p (Pa)' if signed else '|p| (Pa)')
        cb.outline.set_visible(False)
    return fig, axes, t_shots


def plot_labframe_validation(hist: np.ndarray,
                             tau_axis: np.ndarray,
                             zaxis: np.ndarray,
                             c0: float,
                             out_png: str,
                             ntiles: int = 5,
                             xaxis: Optional[np.ndarray] = None,
                             energy_frac: float = 0.1,
                             dpi: int = 130) -> dict:
    """Composite lab-frame validation figure, saved to ``out_png``.

    Top row: ``ntiles`` de-sheared lab-frame snapshots showing the pulse
    physically propagating at ``c0``.  Bottom row: the
    :func:`linear_arrival_check` — ``tau_peak(z)`` in the retarded frame
    (should be flat) and the absolute arrival ``tau_peak(z)+z/c0`` (should
    have slope ``1/c0``).  Returns the check dict.
    """
    hist = np.asarray(hist)
    check = linear_arrival_check(hist, tau_axis, zaxis, c0,
                                 energy_frac=energy_frac)
    signed = _is_signed(hist)
    _, _, vmax = _color_scale(hist, signed)
    t_shots = lab_time_grid(hist, tau_axis, zaxis, c0, nframes=ntiles)

    fig = plt.figure(figsize=(3.0 * ntiles, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, ntiles, height_ratios=[1.15, 0.85])

    snap_axes = [fig.add_subplot(gs[0, k]) for k in range(ntiles)]
    im = None
    for k, (ax, t_star) in enumerate(zip(snap_axes, t_shots)):
        im = plot_labframe_snapshot(
            hist, tau_axis, zaxis, c0, float(t_star), ax=ax,
            xaxis=xaxis, vmax=vmax, add_colorbar=False)
        if k > 0:
            ax.set_ylabel('')
    if im is not None:
        cb = fig.colorbar(im, ax=snap_axes, fraction=0.02, pad=0.01,
                          label='p (Pa)' if signed else '|p| (Pa)')
        cb.outline.set_visible(False)

    # arrival-slope check spanning the full bottom row
    axc = fig.add_subplot(gs[1, :])
    z_mm = check['zaxis'] * 1e3
    sel = check['sel']
    axc.plot(z_mm, check['tau_peak'] * 1e6, 'o', ms=3, color='0.6',
             alpha=0.5, label=r'$\tau_{peak}(z)$ (retarded, all z)')
    axc.plot(z_mm[sel], check['tau_peak'][sel] * 1e6, 'o', ms=4,
             color='tab:blue',
             label=(r'$\tau_{peak}$ retarded  '
                    f"(slope {check['slope_ret_us_per_mm']:+.3f} µs/mm)"))
    axc.plot(z_mm[sel], check['t_abs'][sel] * 1e6, 's', ms=4,
             color='tab:red',
             label=(r'$\tau_{peak}+z/c_0$ (lab)  '
                    f"(slope {check['slope_lab_us_per_mm']:.3f} vs "
                    f"1/c0={check['inv_c0_us_per_mm']:.3f} µs/mm)"))
    if np.isfinite(check['slope_lab_s_per_m']):
        zfit = z_mm[sel]
        tfit = (check['slope_lab_s_per_m'] * check['zaxis'][sel]
                + (check['t_abs'][sel]
                   - check['slope_lab_s_per_m'] * check['zaxis'][sel]).mean())
        axc.plot(zfit, tfit * 1e6, '-', color='tab:red', lw=1, alpha=0.7)
    axc.set_xlabel('z (mm)')
    axc.set_ylabel('arrival time (µs)')
    axc.set_title('Retarded peak (flat) vs lab arrival (slope $1/c_0$)',
                  fontsize=10)
    axc.grid(True, alpha=0.4)
    axc.legend(fontsize=7, loc='best')

    fig.suptitle('Conventional-space-time (lab-frame) wavefront   '
                 r'$p_{lab}(x,z,t)=p_{ret}(x,z,\tau=t-z/c_0)$',
                 fontsize=11)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return check
