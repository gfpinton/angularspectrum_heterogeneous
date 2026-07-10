"""
Runtime + summary diagnostic plotting for the angular-spectrum solver.

Mirrors the imagery emitted by the MATLAB reference (`plot_asr.m` and the
inline movie loop inside `angular_spectrum_solver.m`) so that a Python run
can be visually validated as it progresses.

Two entry points:

* ``plot_runtime_frame`` — called every ``diagnosticInterval`` march steps
  by the solver. Produces a 3x3 dashboard PNG (X-T / Y-T slices, axial
  waveform, XY intensity, XZ/YZ intensity, XZ/YZ MI, on-axis intensity)
  and saves it to ``<diagnosticDir>/frame_NNNN.png``.

* ``plot_summary_report`` — called once at the end of propagation. Emits
  the per-quantity panels that MATLAB's ``plot_asr.m`` produces (Isppa,
  Isppa-dB, MI, PNP, PPP cross-sections; on-axis traces) into
  ``<diagnosticDir>/summary/``.

Both functions accept partial / not-yet-fully-populated arrays so that the
runtime caller can pass ``pI[:, :, :cc+1]`` mid-simulation.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Analytic-signal envelope (FFT-based, no scipy)
# ---------------------------------------------------------------------------
def _envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert-transform envelope along the last axis."""
    nT = x.shape[-1]
    F = np.fft.fft(x, axis=-1)
    h = np.zeros(nT)
    if nT % 2 == 0:
        h[0] = h[nT // 2] = 1
        h[1:nT // 2] = 2
    else:
        h[0] = 1
        h[1:(nT + 1) // 2] = 2
    shape = [1] * (x.ndim - 1) + [nT]
    return np.abs(np.fft.ifft(F * h.reshape(shape), axis=-1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def estimate_pulse_duration(initial_field: np.ndarray, dT: float) -> float:
    """Estimate pulse duration (FWHM of analytic-signal envelope at center).

    Mirrors the MATLAB heuristic:
        icvec = initial_field(round(nX/2), round(nY/2), :)
        pdur  = #samples with |hilbert(icvec)| > max/2  *  dT
    """
    nX, nY, nT = initial_field.shape
    icvec = np.asarray(initial_field[nX // 2, nY // 2, :], dtype=np.float64)
    if not np.any(icvec):
        return float(nT) * dT  # degenerate: use full window
    envelope = _envelope(icvec)
    half_max = envelope.max() / 2.0
    n_above = int(np.sum(envelope > half_max))
    return max(n_above, 1) * dT


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _axes(nX: int, nY: int, nT: int, dX: float, dY: float, dT: float):
    xaxis = (np.arange(nX) * dX); xaxis -= xaxis.mean()
    yaxis = (np.arange(nY) * dY); yaxis -= yaxis.mean()
    taxis = np.arange(nT) * dT
    return xaxis, yaxis, taxis


# ---------------------------------------------------------------------------
# Runtime frame (mirrors movie loop in angular_spectrum_solver.m)
# ---------------------------------------------------------------------------
def plot_runtime_frame(
        frame_idx: int,
        cc: int,
        z_cumulative: float,
        field: np.ndarray,            # (nX, nY, nT)
        pI_partial: np.ndarray,       # (nX, nY, cc+1)
        pnp_partial: np.ndarray,      # (nX, nY, cc+1)
        zvec: np.ndarray,             # (cc+1,)  per-step dz
        dX: float, dY: float, dT: float,
        c0: float, rho0: float, f0: float,
        propDist: float, pdur: float,
        max_amplitude: float,
        diagnostic_dir: str) -> str:
    """Save one PNG dashboard frame and return its path."""
    ensure_dir(diagnostic_dir)

    nX, nY, nT = field.shape
    xaxis, yaxis, taxis = _axes(nX, nY, nT, dX, dY, dT)

    # ----- per-step intensities (W/cm^2) and MI -----
    Isppa_xy = pI_partial[:, :, cc] * dT / (c0 * rho0 * pdur) / 1e4
    MI_xy = -pnp_partial[:, :, cc] / 1e6 / np.sqrt(f0 / 1e6)

    # Build z-resampled volumes against a fixed propDist axis so that the
    # x-axis on XZ / YZ plots is stable across frames.
    est_steps = max(int(np.ceil(propDist / max(zvec[0], 1e-9))), cc + 1)
    full_zaxis = np.linspace(0.0, propDist, est_steps)
    zz = np.cumsum(zvec[:cc + 1])

    pI_x_full = np.zeros((nX, est_steps), dtype=np.float32)
    pI_y_full = np.zeros((nY, est_steps), dtype=np.float32)
    MI_x_full = np.zeros((nX, est_steps), dtype=np.float32)
    MI_y_full = np.zeros((nY, est_steps), dtype=np.float32)
    for i in range(cc + 1):
        idx = int(np.argmin(np.abs(full_zaxis - zz[i])))
        pI_x_full[:, idx] = pI_partial[:, nY // 2, i]
        pI_y_full[:, idx] = pI_partial[nX // 2, :, i]
        MI_x_full[:, idx] = pnp_partial[:, nY // 2, i]
        MI_y_full[:, idx] = pnp_partial[nX // 2, :, i]

    Isppa_xz = pI_x_full * dT / (c0 * rho0 * pdur) / 1e4
    Isppa_yz = pI_y_full * dT / (c0 * rho0 * pdur) / 1e4
    MI_xz = -MI_x_full / 1e6 / np.sqrt(f0 / 1e6)
    MI_yz = -MI_y_full / 1e6 / np.sqrt(f0 / 1e6)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))

    # X-T slice
    ax = axes[0, 0]
    im = ax.imshow(field[:, nY // 2, :], aspect='auto', origin='lower',
                   extent=[taxis[0]*1e6, taxis[-1]*1e6,
                           xaxis[0]*1e3, xaxis[-1]*1e3])
    ax.set_xlabel('t (μs)'); ax.set_ylabel('x (mm)')
    ax.set_title(f'X-T slice, z = {z_cumulative*100:.2f} cm')
    fig.colorbar(im, ax=ax)

    # Y-T slice
    ax = axes[0, 1]
    im = ax.imshow(field[nX // 2, :, :], aspect='auto', origin='lower',
                   extent=[taxis[0]*1e6, taxis[-1]*1e6,
                           yaxis[0]*1e3, yaxis[-1]*1e3])
    ax.set_xlabel('t (μs)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'Y-T slice, z = {z_cumulative*100:.2f} cm')
    fig.colorbar(im, ax=ax)

    # Axial waveform
    ax = axes[0, 2]
    ax.plot(taxis * 1e6, field[nX // 2, nY // 2, :])
    ax.set_xlabel('t (μs)'); ax.set_ylabel('Pressure (Pa)')
    ax.set_title(f'Axial waveform, z = {z_cumulative*100:.2f} cm')
    ax.grid(True)

    # XY intensity
    ax = axes[1, 0]
    im = ax.imshow(Isppa_xy.T, origin='lower', aspect='equal',
                   extent=[xaxis[0]*1e3, xaxis[-1]*1e3,
                           yaxis[0]*1e3, yaxis[-1]*1e3])
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title(f'XY Isppa (W/cm²), z = {z_cumulative*100:.2f} cm')
    fig.colorbar(im, ax=ax)

    # XZ intensity (z on y-axis, z=0 at top)
    ax = axes[1, 1]
    im = ax.imshow(Isppa_xz.T, aspect='auto', origin='upper',
                   extent=[xaxis[0]*1e3, xaxis[-1]*1e3, propDist*100, 0])
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (cm)')
    ax.set_title('XZ Isppa (W/cm²)')
    fig.colorbar(im, ax=ax)

    # YZ intensity
    ax = axes[1, 2]
    im = ax.imshow(Isppa_yz.T, aspect='auto', origin='upper',
                   extent=[yaxis[0]*1e3, yaxis[-1]*1e3, propDist*100, 0])
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (cm)')
    ax.set_title('YZ Isppa (W/cm²)')
    fig.colorbar(im, ax=ax)

    # XZ MI
    ax = axes[2, 0]
    im = ax.imshow(MI_xz.T, aspect='auto', origin='upper',
                   extent=[xaxis[0]*1e3, xaxis[-1]*1e3, propDist*100, 0])
    ax.set_xlabel('x (mm)'); ax.set_ylabel('z (cm)')
    ax.set_title('XZ Mechanical Index')
    fig.colorbar(im, ax=ax)

    # YZ MI
    ax = axes[2, 1]
    im = ax.imshow(MI_yz.T, aspect='auto', origin='upper',
                   extent=[yaxis[0]*1e3, yaxis[-1]*1e3, propDist*100, 0])
    ax.set_xlabel('y (mm)'); ax.set_ylabel('z (cm)')
    ax.set_title('YZ Mechanical Index')
    fig.colorbar(im, ax=ax)

    # On-axis intensity vs z (z on x-axis, independent variable)
    ax = axes[2, 2]
    axial_I = Isppa_xz[nX // 2, :]
    ax.plot(full_zaxis * 100, axial_I, '-b', linewidth=1.5)
    ax.set_xlabel('z (cm)'); ax.set_ylabel('Intensity (W/cm²)')
    ax.set_title('On-axis Isppa')
    ax.grid(True)
    ax.set_xlim(0, propDist * 100)

    fig.suptitle(
        f'f₀={f0/1e6:.2f} MHz · p₀={max_amplitude/1e6:.2f} MPa · '
        f'z={z_cumulative*100:.2f} cm · max Isppa={Isppa_xy.max():.2f} W/cm² · '
        f'max MI={MI_xy.max():.3f}',
        y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = os.path.join(diagnostic_dir, f'frame_{frame_idx:04d}.png')
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Final summary report (mirrors plot_asr.m output)
# ---------------------------------------------------------------------------
def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Lab-frame (conventional space-time) validation page
# ---------------------------------------------------------------------------
def _render_labframe_tex(png_basename: str, check: dict) -> str:
    """One-page LaTeX embedding the lab-frame wavefront figure + arrival check.

    Compiled to ``labframe_validation.pdf`` — the post-solve sibling of the
    pre-solve ``preflight.pdf`` (same pdflatex toolchain).
    """
    def _f(x, nd=3):
        return 'n/a' if not np.isfinite(x) else f'{x:.{nd}f}'

    rel = check.get('rel_err_lab', float('nan'))
    ok = np.isfinite(rel) and rel < 0.05
    status = (r'\textcolor{teal!70!black}{\textbf{PASS}}' if ok
              else r'\textcolor{red}{\textbf{CHECK}}')

    return rf"""\documentclass{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1.2cm,a4paper,landscape]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{amsmath,amssymb}}
\setlength{{\parindent}}{{0pt}}
\pagestyle{{empty}}
\begin{{document}}

{{\large\textbf{{Angular Spectrum Solver — Lab-Frame Wavefront Validation}}}}\\[2pt]
The march stores the field in a \emph{{retarded}} time frame $\tau=t-z/c_0$
(the bulk $c_0$ transit delay is removed each march plane, so the pulse looks
nearly frozen in $z$). The panels below undo that shear —
$p_{{\text{{lab}}}}(x,z,t)=p_{{\text{{ret}}}}(x,z,\ \tau=t-z/c_0)$ — so the
wavefront physically propagates at $c_0$ from the source through the focus and
diverges past it. Signed pressure, diverging scale (red = compression, blue =
rarefaction).

\vspace{{2pt}}
\begin{{center}}
\includegraphics[width=0.86\linewidth]{{{png_basename}}}
\end{{center}}

\vspace{{-2pt}}
\noindent\textbf{{Linear-arrival sanity check}} \quad {status}
\hfill{{\small(retarded peak should be flat; lab arrival slope should equal $1/c_0$)}}

\vspace{{2pt}}
{{\small
\begin{{tabular}}{{lrr}}
\toprule
 & slope ($\mu$s/mm) & $R^2$ \\
\midrule
Retarded peak $\tau_{{\text{{peak}}}}(z)$ (expect $\approx 0$) & {_f(check.get('slope_ret_us_per_mm', float('nan')))} & {_f(check.get('r2_ret', float('nan')), 4)} \\
Lab arrival $\tau_{{\text{{peak}}}}(z)+z/c_0$ (expect $1/c_0$) & {_f(check.get('slope_lab_us_per_mm', float('nan')))} & {_f(check.get('r2_lab', float('nan')), 4)} \\
Physical $1/c_0$ ($c_0={_f(check.get('c0', float('nan')), 1)}$ m/s) & {_f(check.get('inv_c0_us_per_mm', float('nan')))} & --- \\
\midrule
Lab-slope relative error & \multicolumn{{2}}{{r}}{{{_f(rel*100 if np.isfinite(rel) else rel, 2)}\% over {check.get('n_z_used', 0)}/{check.get('n_z_total', 0)} depths}} \\
\bottomrule
\end{{tabular}}
}}

\end{{document}}
"""


def _labframe_validation_page(field_history: np.ndarray,
                              tau_axis: np.ndarray,
                              zaxis: np.ndarray,
                              c0: float,
                              xaxis: np.ndarray,
                              summary_dir: str,
                              ntiles: int = 5,
                              compile_pdf: bool = True,
                              verbose: bool = True) -> Optional[dict]:
    """Render the lab-frame wavefront figure and (optionally) compile a PDF.

    ``field_history`` is the captured retarded-frame xz cube
    ``(nX, nT_ret, nZ)`` (signed mid-y pressure per march step). Returns a
    dict with the PNG/PDF paths and the arrival-check metrics, or ``None``
    when there is too little history to de-shear.
    """
    from labframe import plot_labframe_validation  # local: keep it optional

    hist = np.asarray(field_history)
    if hist.ndim != 3 or hist.shape[1] < 2 or hist.shape[2] < 2:
        if verbose:
            print('[labframe] insufficient xz history; skipping lab-frame page')
        return None

    png_path = os.path.join(summary_dir, 'labframe_wavefront.png')
    check = plot_labframe_validation(
        hist, tau_axis, zaxis, c0, png_path, ntiles=ntiles, xaxis=xaxis)

    pdf_path = None
    tex_path = os.path.join(summary_dir, 'labframe_validation.tex')
    with open(tex_path, 'w') as f:
        f.write(_render_labframe_tex(os.path.basename(png_path), check))
    if compile_pdf:
        from preflight import compile_latex_pdf   # local: avoids import cycle
        pdf_path = compile_latex_pdf(
            'labframe_validation.tex', summary_dir, verbose=verbose)

    if verbose:
        print(f'[labframe] wavefront figure -> {png_path}')
        if pdf_path:
            print(f'[labframe] validation PDF -> {pdf_path}')
        print(f"[labframe] lab arrival slope {check['slope_lab_us_per_mm']:.3f} "
              f"vs 1/c0 {check['inv_c0_us_per_mm']:.3f} µs/mm "
              f"(retarded peak slope {check['slope_ret_us_per_mm']:+.3f})")
    return {'png_path': png_path, 'tex_path': tex_path,
            'pdf_path': pdf_path, 'check': check}


def plot_summary_report(
        field: np.ndarray,             # (nX, nY, nT) final
        initial_field: np.ndarray,     # (nX, nY, nT)
        pnp: np.ndarray,               # (nX, nY, nZ)
        ppp: np.ndarray,               # (nX, nY, nZ)
        pI: np.ndarray,                # (nX, nY, nZ)
        pIloss: Optional[np.ndarray],  # (nX, nY, nZ) or None
        zaxis: np.ndarray,             # (nZ,)
        pax: np.ndarray,               # (nT, nZ)
        dX: float, dY: float, dT: float,
        c0: float, rho0: float, f0: float, pdur: float,
        diagnostic_dir: str,
        dB_levels=(-30, -20, -10, -6, -3),
        boundary_factor: float = 0.2,
        stab_history: Optional[np.ndarray] = None,
        dZ_history: Optional[np.ndarray] = None,
        restart_events: Optional[list] = None,
        stab_threshold: float = 0.2,
        elapsed_s: Optional[float] = None,
        xz_history: Optional[np.ndarray] = None,
        xz_tau_axis: Optional[np.ndarray] = None,
        labframe_compile_pdf: bool = True) -> None:
    """Emit the static summary panels into ``<diagnostic_dir>/summary/``.

    When ``xz_history`` (the captured retarded-frame xz cube, shape
    ``(nX, nT_ret, nZ)`` of signed mid-y pressure per march step) and its
    ``xz_tau_axis`` are supplied, a conventional-space-time (lab-frame)
    wavefront validation page is added (``labframe_wavefront.png`` plus, if
    ``labframe_compile_pdf`` and ``pdflatex`` are available,
    ``labframe_validation.pdf``).
    """
    summary_dir = os.path.join(diagnostic_dir, 'summary')
    ensure_dir(summary_dir)

    nX, nY, nT = field.shape
    xaxis, yaxis, taxis = _axes(nX, nY, nT, dX, dY, dT)

    # Index of max pI (3D argmax)
    flat_idx = int(np.argmax(pI))
    idx, idy, idz = np.unravel_index(flat_idx, pI.shape)

    # Isppa volumes
    Isppa = pI * dT / (c0 * rho0 * pdur) / 1e4              # W/cm^2
    pIdb = 10.0 * np.log10(np.maximum(Isppa, 1e-30)
                           / max(Isppa.max(), 1e-30))
    MI_vol = -pnp / 1e6 / np.sqrt(f0 / 1e6)

    # ------- IC vector -------
    icvec = initial_field[nX // 2, nY // 2, :]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(taxis, icvec)
    ax.set_xlabel('t (s)'); ax.set_ylabel('Pa')
    ax.set_title('Initial Condition (axial)')
    ax.grid(True)
    _save(fig, os.path.join(summary_dir, 'icvec.png'))

    # ------- IC vs peak axial waveform -------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(taxis, icvec, label='IC')
    ax.plot(taxis, pax[:, idz],
            label=f'Peak pressure, z={zaxis[idz]:.4f} m')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Pa'); ax.grid(True); ax.legend()
    _save(fig, os.path.join(summary_dir, 'paxpeak.png'))

    # ------- XY Isppa at peak z -------
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Isppa[:, :, idz].T, origin='lower', aspect='equal',
                   extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Isppa at z={zaxis[idz]:.4f} m')
    fig.colorbar(im, ax=ax, label='W/cm²')
    _save(fig, os.path.join(summary_dir, 'pIxy.png'))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pIdb[:, :, idz].T, origin='lower', aspect='equal', vmin=-40, vmax=0,
                   extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title(f'Isppa dB at z={zaxis[idz]:.4f} m')
    fig.colorbar(im, ax=ax, label='dB')
    _save(fig, os.path.join(summary_dir, 'pIxy_db.png'))

    # ------- XZ / YZ Isppa, Isppa dB, MI, PNP, PPP -------
    # Depth (z) plotted on the y-axis with z=0 at the top.
    z_top = zaxis[0]
    z_bot = zaxis[-1]

    def _imshow_zX(arr, xlab, title, cbar_label, fname,
                   vmin=None, vmax=None, axis_x=xaxis):
        # arr is shape (n_lateral, n_z); transpose so z is the first axis.
        fig, ax = plt.subplots(figsize=(6, 7))
        im = ax.imshow(arr.T, aspect='auto', origin='upper',
                       extent=[axis_x[0], axis_x[-1], z_bot, z_top],
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlab); ax.set_ylabel('z (m)'); ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar_label)
        _save(fig, os.path.join(summary_dir, fname))

    _imshow_zX(Isppa[:, nY // 2, :], 'x (m)', 'Isppa', 'W/cm²', 'pIxz.png')
    _imshow_zX(pIdb[:, nY // 2, :], 'x (m)', 'Isppa dB', 'dB', 'pIxz_db.png',
               vmin=-40, vmax=0)
    _imshow_zX(Isppa[nX // 2, :, :], 'y (m)', 'Isppa', 'W/cm²', 'pIyz.png',
               axis_x=yaxis)
    _imshow_zX(pIdb[nX // 2, :, :], 'y (m)', 'Isppa dB', 'dB', 'pIyz_db.png',
               vmin=-40, vmax=0, axis_x=yaxis)

    _imshow_zX(MI_vol[:, nY // 2, :], 'x (m)', 'MI', 'MI', 'MI_xz.png')
    _imshow_zX(MI_vol[nX // 2, :, :], 'y (m)', 'MI', 'MI', 'MI_yz.png',
               axis_x=yaxis)
    _imshow_zX(pnp[:, nY // 2, :], 'x (m)', 'PNP', 'Pa', 'pnp_xz.png')
    _imshow_zX(pnp[nX // 2, :, :], 'y (m)', 'PNP', 'Pa', 'pnp_yz.png',
               axis_x=yaxis)
    _imshow_zX(ppp[:, nY // 2, :], 'x (m)', 'PPP', 'Pa', 'ppp_xz.png')
    _imshow_zX(ppp[nX // 2, :, :], 'y (m)', 'PPP', 'Pa', 'ppp_yz.png',
               axis_x=yaxis)

    # ------- on-axis Isppa and dB (z on x-axis, independent variable) -------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(zaxis, Isppa[nX // 2, nY // 2, :])
    ax.set_xlabel('z (m)'); ax.set_ylabel('Isppa (W/cm²)'); ax.grid(True)
    _save(fig, os.path.join(summary_dir, 'Isppa_ax.png'))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(zaxis, pIdb[nX // 2, nY // 2, :])
    ax.set_xlabel('z (m)'); ax.set_ylabel('dB')
    ax.set_ylim(-40, 0); ax.grid(True)
    _save(fig, os.path.join(summary_dir, 'pIdb_ax.png'))

    # ------- combined dashboard like the runtime frame, but final -------
    # Depth on y-axis with z=0 on top for the depth panels.
    z_top_cm = zaxis[0] * 100
    z_bot_cm = zaxis[-1] * 100
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    im = axes[0, 0].imshow(Isppa[:, nY // 2, :].T, aspect='auto', origin='upper',
                           extent=[xaxis[0]*1e3, xaxis[-1]*1e3,
                                   z_bot_cm, z_top_cm])
    axes[0, 0].set_xlabel('x (mm)'); axes[0, 0].set_ylabel('z (cm)')
    axes[0, 0].set_title('Final XZ Isppa')
    fig.colorbar(im, ax=axes[0, 0], label='W/cm²')

    im = axes[0, 1].imshow(Isppa[nX // 2, :, :].T, aspect='auto', origin='upper',
                           extent=[yaxis[0]*1e3, yaxis[-1]*1e3,
                                   z_bot_cm, z_top_cm])
    axes[0, 1].set_xlabel('y (mm)'); axes[0, 1].set_ylabel('z (cm)')
    axes[0, 1].set_title('Final YZ Isppa')
    fig.colorbar(im, ax=axes[0, 1], label='W/cm²')

    axes[1, 0].plot(zaxis * 100, Isppa[nX // 2, nY // 2, :])
    axes[1, 0].set_xlabel('z (cm)'); axes[1, 0].set_ylabel('Isppa (W/cm²)')
    axes[1, 0].set_title('On-axis Isppa'); axes[1, 0].grid(True)

    axes[1, 1].plot(taxis * 1e6, icvec, label='IC')
    axes[1, 1].plot(taxis * 1e6, pax[:, idz],
                    label=f'Peak z={zaxis[idz]*100:.2f} cm')
    axes[1, 1].set_xlabel('t (μs)'); axes[1, 1].set_ylabel('Pa')
    axes[1, 1].set_title('Axial waveform: IC vs peak'); axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.suptitle(
        f'Final summary  |  max Isppa={Isppa.max():.2f} W/cm²  ·  '
        f'max MI={MI_vol.max():.3f}  ·  '
        f'peak z={zaxis[idz]*100:.2f} cm', y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(summary_dir, 'dashboard.png'))

    # ---------------------------------------------------------------------
    # Boundary leakage check (spatial + temporal)
    # ---------------------------------------------------------------------
    n_bx = max(int(round(nX * boundary_factor)), 1)
    n_by = max(int(round(nY * boundary_factor)), 1)
    bdy_mask = np.zeros((nX, nY), dtype=np.float32)
    bdy_mask[:n_bx, :] = 1; bdy_mask[-n_bx:, :] = 1
    bdy_mask[:, :n_by] = 1; bdy_mask[:, -n_by:] = 1

    nZ = pI.shape[2]
    e_total = pI.reshape(-1, nZ).sum(axis=0)
    e_bdy = (pI * bdy_mask[:, :, None]).reshape(-1, nZ).sum(axis=0)
    spatial_frac = e_bdy / np.maximum(e_total, 1e-30)

    # Temporal: envelope of central trace, edge fraction + centroid
    pax_env = _envelope(pax.T)  # (nZ, nT)
    n_edge_t = max(int(round(0.05 * nT)), 1)
    e_t_total = (pax_env ** 2).sum(axis=1)
    e_t_edge = ((pax_env[:, :n_edge_t] ** 2).sum(axis=1)
                + (pax_env[:, -n_edge_t:] ** 2).sum(axis=1))
    temp_edge_frac = e_t_edge / np.maximum(e_t_total, 1e-30)
    centroid = (taxis[None, :] * pax_env ** 2).sum(axis=1) / np.maximum(e_t_total, 1e-30)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    axes[0].plot(zaxis * 100, spatial_frac * 100)
    axes[0].axhline(5, color='r', linestyle='--', alpha=0.6, label='5% warn')
    axes[0].set_xlabel('z (cm)'); axes[0].set_ylabel('Spatial boundary energy (%)')
    axes[0].grid(True); axes[0].legend()
    axes[0].set_title('Energy in absorbing layer')

    axes[1].plot(zaxis * 100, temp_edge_frac * 100)
    axes[1].axhline(1, color='r', linestyle='--', alpha=0.6, label='1% warn')
    axes[1].set_xlabel('z (cm)')
    axes[1].set_ylabel('Temporal edge energy (%, first+last 5%)')
    axes[1].grid(True); axes[1].legend()
    axes[1].set_title('Pulse near time-window edges')

    axes[2].plot(zaxis * 100, centroid * 1e6)
    axes[2].axhline(taxis[n_edge_t] * 1e6, color='r', linestyle='--', alpha=0.6,
                    label='5% edge')
    axes[2].axhline(taxis[-n_edge_t] * 1e6, color='r', linestyle='--', alpha=0.6)
    axes[2].set_xlabel('z (cm)')
    axes[2].set_ylabel('Pulse-envelope centroid (μs)')
    axes[2].grid(True); axes[2].legend()
    axes[2].set_title('Temporal drift')

    fig.suptitle('Boundary leakage check '
                 f'(max spatial={spatial_frac.max()*100:.2f}% · '
                 f'max temporal edge={temp_edge_frac.max()*100:.2f}%)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(summary_dir, 'boundaries.png'))

    # ---------------------------------------------------------------------
    # Stability margin / step-size history
    # ---------------------------------------------------------------------
    if stab_history is not None and stab_history.size > 0:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        steps = np.arange(stab_history.size)

        axes[0].plot(steps, stab_history)
        axes[0].axhline(stab_threshold, color='r', linestyle='--', alpha=0.7,
                        label=f'threshold={stab_threshold}')
        axes[0].set_xlabel('step #')
        axes[0].set_ylabel('Stability margin  N·dZ/dT·max|p|')
        axes[0].grid(True); axes[0].legend()
        axes[0].set_title('Stability margin per step')

        if dZ_history is not None and dZ_history.size > 0:
            axes[1].plot(np.arange(dZ_history.size), dZ_history * 1e3)
            axes[1].set_xlabel('step #'); axes[1].set_ylabel('dZ (mm)')
            axes[1].grid(True)
            axes[1].set_title('Step size per step')

        n_restart = len(restart_events) if restart_events else 0
        fig.suptitle(f'Numerical stability '
                     f'(max margin={stab_history.max():.4f} · '
                     f'restarts={n_restart})',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        _save(fig, os.path.join(summary_dir, 'stability.png'))

    # ---------------------------------------------------------------------
    # Harmonic content along the axis (FFT of central trace per z)
    # ---------------------------------------------------------------------
    PAX_F = np.fft.rfft(pax, axis=0)            # (n_freq, nZ)
    freqs = np.fft.rfftfreq(nT, dT)
    mag = (np.abs(PAX_F) / nT) * 2.0            # peak amplitude per bin
    df = freqs[1] if freqs.size > 1 else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(1, 7):
        f_target = k * f0
        if f_target > freqs[-1]:
            break
        bin_idx = int(round(f_target / df))
        bin_idx = min(bin_idx, mag.shape[0] - 1)
        ax.plot(zaxis * 100, np.maximum(mag[bin_idx, :], 1e-30) / 1e3,
                label=f'{k}·f₀ ({k*f0/1e6:.2f} MHz)')
    ax.set_yscale('log')
    ax.set_xlabel('z (cm)'); ax.set_ylabel('|p̂| (kPa)')
    ax.grid(True, which='both', alpha=0.4); ax.legend()
    ax.set_title('Harmonic content along axis')
    _save(fig, os.path.join(summary_dir, 'harmonics.png'))

    # ---------------------------------------------------------------------
    # Lab-frame (conventional space-time) wavefront validation page
    # ---------------------------------------------------------------------
    labframe_metrics = None
    if xz_history is not None and np.asarray(xz_history).size and xz_tau_axis is not None:
        lf = _labframe_validation_page(
            np.asarray(xz_history), np.asarray(xz_tau_axis), zaxis, c0,
            xaxis, summary_dir,
            compile_pdf=labframe_compile_pdf, verbose=True)
        if lf is not None:
            c = lf['check']
            labframe_metrics = {
                'lab_arrival_slope_us_per_mm': c['slope_lab_us_per_mm'],
                'inv_c0_us_per_mm': c['inv_c0_us_per_mm'],
                'lab_slope_rel_err': c['rel_err_lab'],
                'retarded_peak_slope_us_per_mm': c['slope_ret_us_per_mm'],
                'lab_arrival_r2': c['r2_lab'],
                'n_z_used': c['n_z_used'],
                'pdf_path': lf['pdf_path'],
            }

    # ---------------------------------------------------------------------
    # metrics.json
    # ---------------------------------------------------------------------
    metrics = {
        'labframe': labframe_metrics,
        'max_Isppa_W_per_cm2': float(Isppa.max()),
        'max_MI': float(MI_vol.max()),
        'max_pnp_Pa': float(pnp.max()),
        'min_pnp_Pa': float(pnp.min()),
        'max_ppp_Pa': float(ppp.max()),
        'peak_z_m': float(zaxis[idz]),
        'peak_indices_xyz': [int(idx), int(idy), int(idz)],
        'n_z_steps': int(nZ),
        'propDist_m': float(zaxis[-1]),
        'pdur_s': float(pdur),
        'f0_Hz': float(f0), 'c0_m_per_s': float(c0), 'rho0_kg_per_m3': float(rho0),
        'boundary_spatial_frac_max': float(spatial_frac.max()),
        'boundary_spatial_frac_final': float(spatial_frac[-1]),
        'boundary_temporal_edge_frac_max': float(temp_edge_frac.max()),
        'boundary_temporal_edge_frac_final': float(temp_edge_frac[-1]),
        'pulse_centroid_min_us': float(centroid.min() * 1e6),
        'pulse_centroid_max_us': float(centroid.max() * 1e6),
        'time_window_us': float(taxis[-1] * 1e6),
        'stability_threshold': float(stab_threshold),
        'stability_margin_max': (float(stab_history.max())
                                 if stab_history is not None and stab_history.size
                                 else None),
        'mean_dZ_m': (float(dZ_history.mean())
                      if dZ_history is not None and dZ_history.size else None),
        'n_stability_restarts': len(restart_events) if restart_events else 0,
        'restart_events': ([{'cc': int(c), 'margin': float(m)}
                            for c, m in restart_events]
                           if restart_events else []),
        'elapsed_s': float(elapsed_s) if elapsed_s is not None else None,
    }
    with open(os.path.join(summary_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Diagnostic summary report written to {summary_dir}')


# ---------------------------------------------------------------------------
# Initial-condition snapshot (mirrors plot_initial_conditions.m)
# ---------------------------------------------------------------------------
def plot_initial_conditions(initial_field: np.ndarray,
                            dX: float, dY: float, dT: float,
                            diagnostic_dir: str) -> None:
    ensure_dir(diagnostic_dir)
    nX, nY, nT = initial_field.shape
    xaxis, yaxis, taxis = _axes(nX, nY, nT, dX, dY, dT)
    icvec = initial_field[nX // 2, nY // 2, :]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))
    axes[0].plot(taxis * 1e6, icvec)
    axes[0].set_xlabel('t (μs)'); axes[0].set_ylabel('Pa')
    axes[0].set_title('Initial Condition (axial)'); axes[0].grid(True)

    im = axes[1].imshow(initial_field[:, nY // 2, :], aspect='auto', origin='lower',
                        extent=[taxis[0]*1e6, taxis[-1]*1e6,
                                xaxis[0]*1e3, xaxis[-1]*1e3])
    axes[1].set_xlabel('t (μs)'); axes[1].set_ylabel('x (mm)')
    axes[1].set_title('Initial Field (X-slice)')
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(initial_field[nX // 2, :, :], aspect='auto', origin='lower',
                        extent=[taxis[0]*1e6, taxis[-1]*1e6,
                                yaxis[0]*1e3, yaxis[-1]*1e3])
    axes[2].set_xlabel('t (μs)'); axes[2].set_ylabel('y (mm)')
    axes[2].set_title('Initial Field (Y-slice)')
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    _save(fig, os.path.join(diagnostic_dir, 'initial_field.png'))
