"""
Pre-flight sanity report.

Generates a single-page LaTeX/PDF summary of an upcoming simulation:

* compact tables for grid, pulse, material, and numerics
* multi-panel figure showing the IC in space and time and its spectrum
* a list of automatic warnings (under-resolved grid, pulse touching the
  time-window edge, aperture spilling into the absorbing band, predicted
  full-shock formation, high CFL, etc.)

Designed to be invoked **before** ``angular_spectrum_solve`` so the user
can verify that the run is set up correctly. Output lands in
``<output_dir>/preflight_panels.png``, ``preflight.tex`` and (if
pdflatex is available) ``preflight.pdf``.
"""
from __future__ import annotations

import datetime
import os
import shutil
import socket
import subprocess
import sys
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diagnostic_plotting import _envelope, ensure_dir, estimate_pulse_duration


# ---------------------------------------------------------------------------
# LaTeX escaping for user-supplied strings
# ---------------------------------------------------------------------------
_TEX_ESC = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\^{}', '\\': r'\textbackslash{}'}

# Unicode → LaTeX. Applied before _tex_escape so we don't double-escape backslashes.
_UNICODE_TEX = {
    'λ': r'$\lambda$', 'μ': r'$\mu$', 'α': r'$\alpha$', 'β': r'$\beta$',
    'ρ': r'$\rho$', 'Ω': r'$\Omega$', 'ω': r'$\omega$', 'Γ': r'$\Gamma$',
    '·': r'$\cdot$', '×': r'$\times$', '→': r'$\to$',
    '≥': r'$\geq$', '≤': r'$\leq$', '≠': r'$\neq$', '≈': r'$\approx$',
    '∞': r'$\infty$', '²': r'$^2$', '³': r'$^3$', '°': r'$^\circ$',
    '√': r'$\sqrt{}$', '±': r'$\pm$', '∫': r'$\int$', '∂': r'$\partial$',
    'π': r'$\pi$',
}


def _tex_escape(s: str) -> str:
    """Escape LaTeX special characters and translate common unicode."""
    s = str(s)
    for u, t in _UNICODE_TEX.items():
        s = s.replace(u, '\x00' + t + '\x00')   # protect from char escaping
    out = []
    in_protected = False
    for ch in s:
        if ch == '\x00':
            in_protected = not in_protected
            continue
        if in_protected:
            out.append(ch)
        else:
            out.append(_TEX_ESC.get(ch, ch))
    return ''.join(out)


# ---------------------------------------------------------------------------
# Derived quantities (no plotting / no LaTeX yet)
# ---------------------------------------------------------------------------
def _compute_metrics(initial_field: np.ndarray, params) -> dict:
    nX, nY, nT = initial_field.shape
    f0 = float(params.f0); c0 = float(params.c0); rho0 = float(params.rho0)
    beta = float(params.beta)
    omega0 = 2 * np.pi * f0
    lam = c0 / f0
    N = beta / (2 * c0 ** 3 * rho0)

    p0 = float(np.max(np.abs(initial_field)))

    cfl = c0 * params.dT / params.dX
    pts_per_lambda_x = lam / params.dX
    pts_per_lambda_y = lam / params.dY
    pts_per_cycle = 1.0 / (f0 * params.dT) if f0 > 0 else float('inf')
    domain_x = nX * params.dX
    domain_y = nY * params.dY
    t_window = nT * params.dT

    pdur = (params.pdur if getattr(params, 'pdur', None) is not None
            else estimate_pulse_duration(initial_field, params.dT))

    # Pulse envelope of central trace
    icvec = np.asarray(initial_field[nX // 2, nY // 2, :], dtype=np.float64)
    env = _envelope(icvec)
    t = np.arange(nT) * params.dT
    e_tot = (env ** 2).sum()
    centroid_t = (t * env ** 2).sum() / max(e_tot, 1e-30) if e_tot > 0 else 0.0
    n_edge_t = max(int(round(0.05 * nT)), 1)
    edge_frac_t = (((env[:n_edge_t] ** 2).sum() + (env[-n_edge_t:] ** 2).sum())
                   / max(e_tot, 1e-30))

    # Spatial energy and boundary fraction
    px2 = (initial_field ** 2).sum(axis=2)  # (nX, nY)
    bdy_factor = float(getattr(params, 'boundaryFactor', 0.2))
    n_bx = max(int(round(nX * bdy_factor)), 1)
    n_by = max(int(round(nY * bdy_factor)), 1)
    bdy_mask = np.zeros((nX, nY), dtype=np.float32)
    bdy_mask[:n_bx, :] = 1; bdy_mask[-n_bx:, :] = 1
    bdy_mask[:, :n_by] = 1; bdy_mask[:, -n_by:] = 1
    spat_bdy_frac = float((px2 * bdy_mask).sum() / max(px2.sum(), 1e-30))

    # Aperture footprint (pixels with non-trivial energy)
    if px2.max() > 0:
        ap_mask = px2 > 1e-3 * px2.max()
        ap_pixels = int(ap_mask.sum())
        ap_area = ap_pixels * params.dX * params.dY
    else:
        ap_pixels = 0
        ap_area = 0.0

    # Stability estimate
    stab_init = N * params.dZmin / params.dT * p0
    stab_thr = float(getattr(params, 'stabilityThreshold', 0.2))

    # Attenuation length (Np/m at f0). alpha0<0 marks "water" => 2.17e-3 f^2.
    if params.alpha0 < 0:
        alpha_eff = 2.17e-3
        pw = 2.0
        alpha_label = '$-1$ (water proxy)'
    else:
        alpha_eff = float(params.alpha0)
        pw = float(getattr(params, 'attenPow', 1.0))
        alpha_label = f'{alpha_eff:.4g} dB/MHz$^{{{pw:g}}}$/cm'
    alpha_Np_m = (alpha_eff * (f0 / 1e6) ** pw) * 100.0 / 8.685889638
    l_atten = 1.0 / alpha_Np_m if alpha_Np_m > 0 else float('inf')

    # Plane-wave shock formation length: rho0 c0^3 / (beta omega0 p0)
    if p0 > 0 and beta > 0:
        l_shock = (rho0 * c0 ** 3) / (beta * omega0 * p0)
    else:
        l_shock = float('inf')
    goldberg = (l_atten / l_shock) if (l_shock > 0 and np.isfinite(l_shock)
                                       and np.isfinite(l_atten)) else None

    # Estimated step count
    est_steps = int(np.ceil(params.propDist / max(params.dZmin, 1e-30)))

    return dict(
        nX=nX, nY=nY, nT=nT,
        dX=params.dX, dY=params.dY, dT=params.dT,
        c0=c0, rho0=rho0, beta=beta, alpha0=params.alpha0,
        attenPow=getattr(params, 'attenPow', 1.0),
        f0=f0, omega0=omega0, lam=lam,
        domain_x=domain_x, domain_y=domain_y, t_window=t_window,
        pts_per_lambda_x=pts_per_lambda_x,
        pts_per_lambda_y=pts_per_lambda_y,
        pts_per_cycle=pts_per_cycle, cfl=cfl,
        p0=p0, pdur=pdur, centroid_t=centroid_t, edge_frac_t=edge_frac_t,
        spat_bdy_frac=spat_bdy_frac,
        ap_pixels=ap_pixels, ap_area=ap_area,
        N=N, alpha_eff=alpha_eff, attenPow_eff=pw,
        alpha_Np_m=alpha_Np_m, l_atten=l_atten, l_shock=l_shock,
        goldberg=goldberg, alpha_label=alpha_label,
        propDist=params.propDist, dZmin=params.dZmin, est_steps=est_steps,
        stab_init=stab_init, stab_thr=stab_thr,
        useSplitStep=getattr(params, 'useSplitStep', True),
        useTVD=getattr(params, 'useTVD', True),
        useAdaptiveFiltering=getattr(params, 'useAdaptiveFiltering', True),
        fluxScheme=getattr(params, 'fluxScheme', 'rusanov'),
        boundaryFactor=bdy_factor,
        boundaryProfile=getattr(params, 'boundaryProfile', 'quadratic'),
        useFreqWeightedBoundary=getattr(params, 'useFreqWeightedBoundary', False),
        useSuperAbsorbing=getattr(params, 'useSuperAbsorbing', False),
        useObliquityCorrection=getattr(params, 'useObliquityCorrection', True),
        useNonlinearityObliquity=getattr(params, 'useNonlinearityObliquity', False),
        n_phaseScreens=(0 if getattr(params, 'phaseScreens', None) is None
                        else len(params.phaseScreens)),
        n_sourcePlanes=(0 if getattr(params, 'sourcePlanes', None) is None
                        else len(params.sourcePlanes)),
    )


def _build_warnings(m: dict) -> list:
    w = []
    if m['pts_per_lambda_x'] < 4:
        w.append(f"Under-resolved laterally: {m['pts_per_lambda_x']:.2f} pts/λ "
                 f"in X (recommend ≥4)")
    if m['pts_per_lambda_y'] < 4:
        w.append(f"Under-resolved laterally: {m['pts_per_lambda_y']:.2f} pts/λ "
                 f"in Y (recommend ≥4)")
    if m['pts_per_cycle'] < 8:
        w.append(f"Few samples per cycle: {m['pts_per_cycle']:.2f} (recommend ≥8)")
    if m['pdur'] > 0.5 * m['t_window']:
        w.append(f"Pulse fills {m['pdur']/m['t_window']*100:.0f}% of t-window "
                 f"({m['pdur']*1e6:.2f}/{m['t_window']*1e6:.2f} μs)")
    if m['edge_frac_t'] > 0.01:
        w.append(f"IC has {m['edge_frac_t']*100:.2f}% energy in t-window "
                 f"first/last 5% (above 1% threshold)")
    if m['spat_bdy_frac'] > 0.05:
        w.append(f"IC has {m['spat_bdy_frac']*100:.2f}% energy inside the "
                 f"spatial absorbing band (above 5% threshold)")
    if m['stab_init'] > m['stab_thr']:
        w.append(f"Initial stability margin {m['stab_init']:.3f} above threshold "
                 f"{m['stab_thr']:.3f} → solver will reduce dZ on step 1")
    elif m['stab_init'] > 0.5 * m['stab_thr']:
        w.append(f"Initial stability margin {m['stab_init']:.3f} above 0.5×threshold "
                 f"({m['stab_thr']:.3f}) → solver will run near limit")
    if np.isfinite(m['l_shock']) and m['l_shock'] < m['propDist']:
        w.append(f"Shock distance {m['l_shock']*100:.2f} cm below propDist "
                 f"{m['propDist']*100:.2f} cm → shock-formation regime "
                 f"(verify TVD/k-filter)")
    if m['cfl'] > 1.0:
        w.append(f"CFL c0·dT/dX={m['cfl']:.2f} above 1 (TVD limiter typically "
                 f"requires below 1)")
    return w


# ---------------------------------------------------------------------------
# Phase / amplitude screens overview
# ---------------------------------------------------------------------------
def _screen_stats(name: str, arr: np.ndarray) -> dict:
    a = np.asarray(arr)
    return {
        'name': name, 'shape': a.shape,
        'min': float(a.min()), 'max': float(a.max()),
        'mean': float(a.mean()), 'std': float(a.std()),
        'rms': float(np.sqrt((a ** 2).mean())),
    }


def _plot_screens(params, output_dir: str) -> Optional[str]:
    """If phaseScreens are configured, render a per-screen summary figure.

    Each screen contributes one row showing (a) phase map in radians,
    (b) phase histogram, and (optionally) (c) amplitude transmission map.
    Returns the path of the saved PNG, or None when no screens exist.
    """
    screens = getattr(params, 'phaseScreens', None)
    if not screens:
        return None

    # Detect whether any screen carries an amplitude array
    has_amp = any(len(s) > 2 and s[2] is not None for s in screens)
    cols = 3 if has_amp else 2
    rows = len(screens)

    # Use the screen's own shape as the spatial grid
    nX, nY = np.asarray(screens[0][1]).shape
    xaxis = (np.arange(nX) - nX / 2) * params.dX
    yaxis = (np.arange(nY) - nY / 2) * params.dY

    fig, axes = plt.subplots(rows, cols,
                             figsize=(4.0 * cols, 3.4 * rows),
                             squeeze=False)

    stats = []  # collect per-screen stats for the LaTeX table
    for i, s in enumerate(screens):
        z = float(s[0])
        phase = np.asarray(s[1])
        amp = np.asarray(s[2]) if (has_amp and len(s) > 2 and s[2] is not None) else None

        # 1) Phase map (diverging colormap, symmetric range)
        ax = axes[i, 0]
        vmax = float(np.max(np.abs(phase))) or 1.0
        im = ax.imshow(phase.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[xaxis[0] * 1e3, xaxis[-1] * 1e3,
                               yaxis[0] * 1e3, yaxis[-1] * 1e3])
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ax.set_title(f'Screen #{i+1} phase  (z = {z*100:.2f} cm)')
        fig.colorbar(im, ax=ax, label='rad')

        # 2) Phase histogram
        ax = axes[i, 1]
        ax.hist(phase.ravel(), bins=40, color='steelblue', alpha=0.85)
        ax.set_xlabel('phase (rad)'); ax.set_ylabel('count')
        ax.set_title(f'std = {phase.std():.3f} rad')
        ax.grid(True, alpha=0.4)

        # 3) Amplitude map (sequential colormap, [0,1])
        if has_amp:
            ax = axes[i, 2]
            if amp is None:
                ax.text(0.5, 0.5, '(no amplitude screen)',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                vmin = max(0.0, float(amp.min()) - 0.05)
                vmax_a = min(1.0, float(amp.max()) + 0.05)
                im = ax.imshow(amp.T, origin='lower', cmap='viridis',
                               vmin=vmin, vmax=vmax_a,
                               extent=[xaxis[0] * 1e3, xaxis[-1] * 1e3,
                                       yaxis[0] * 1e3, yaxis[-1] * 1e3])
                ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
                ax.set_title(f'Screen #{i+1} amplitude transmission')
                fig.colorbar(im, ax=ax, label='|T|')

        stats.append({
            'idx': i + 1,
            'z_cm': z * 100,
            'phase': _screen_stats('phase', phase),
            'amp': _screen_stats('amp', amp) if amp is not None else None,
        })

    fig.tight_layout()
    out = os.path.join(output_dir, 'preflight_screens.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, stats


def _render_screens_block(stats: list, png_relpath: str) -> str:
    """LaTeX snippet placed on a fresh page after the main report."""
    rows = []
    has_amp = any(s['amp'] is not None for s in stats)
    if has_amp:
        rows.append(r'\toprule')
        rows.append(r'\# & $z$ (cm) & '
                    r'phase min/max (rad) & phase $\sigma$ (rad) & '
                    r'amp min/max & amp mean \\')
        rows.append(r'\midrule')
        for s in stats:
            ph = s['phase']
            am = s['amp']
            am_str = (f"{am['min']:.3f} / {am['max']:.3f}"
                      if am is not None else r'\textemdash')
            am_mean = f"{am['mean']:.3f}" if am is not None else r'\textemdash'
            rows.append(f"{s['idx']} & {s['z_cm']:.2f} & "
                        f"{ph['min']:.2f} / {ph['max']:.2f} & "
                        f"{ph['std']:.3f} & {am_str} & {am_mean} \\\\")
        rows.append(r'\bottomrule')
        col_spec = 'r r c c c c'
    else:
        rows.append(r'\toprule')
        rows.append(r'\# & $z$ (cm) & '
                    r'phase min/max (rad) & phase $\sigma$ (rad) & shape \\')
        rows.append(r'\midrule')
        for s in stats:
            ph = s['phase']
            shape_str = r'$' + r'\times'.join(str(d) for d in ph['shape']) + r'$'
            rows.append(f"{s['idx']} & {s['z_cm']:.2f} & "
                        f"{ph['min']:.2f} / {ph['max']:.2f} & "
                        f"{ph['std']:.3f} & {shape_str} \\\\")
        rows.append(r'\bottomrule')
        col_spec = 'r r c c c'

    table = ('\\begin{tabular}{' + col_spec + '}\n'
             + '\n'.join(rows) + '\n\\end{tabular}')

    return rf"""\clearpage
\section*{{Phase / amplitude screens ({len(stats)})}}

{table}

\vspace{{8pt}}
\begin{{center}}
\includegraphics[width=0.95\linewidth]{{{png_relpath}}}
\end{{center}}
"""


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------
def _plot_panels(initial_field: np.ndarray, m: dict, out_path: str) -> None:
    nX, nY, nT = initial_field.shape
    xaxis = (np.arange(nX) - nX // 2) * m['dX']
    yaxis = (np.arange(nY) - nY // 2) * m['dY']
    taxis = np.arange(nT) * m['dT']

    icvec = initial_field[nX // 2, nY // 2, :]
    env = _envelope(np.asarray(icvec, dtype=np.float64))

    # Spectrum of central trace (kPa, dB)
    spec = np.fft.rfft(icvec)
    freqs = np.fft.rfftfreq(nT, m['dT'])
    spec_kPa = (np.abs(spec) / nT) * 2.0 / 1e3
    spec_dB = 20 * np.log10(np.maximum(spec_kPa, 1e-30) /
                            max(spec_kPa.max(), 1e-30))

    # Spatial RMS map and lateral profile
    px2 = (initial_field ** 2).sum(axis=2)
    rms_xy = np.sqrt(px2 / nT)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    # 1) Pulse waveform (with envelope)
    ax = axes[0, 0]
    ax.plot(taxis * 1e6, icvec / 1e6, label='p(t)')
    ax.plot(taxis * 1e6, env / 1e6, '--', alpha=0.7, label='|envelope|')
    ax.axvline(m['centroid_t'] * 1e6, color='gray', linestyle=':',
               alpha=0.7, label=f'centroid {m["centroid_t"]*1e6:.2f} μs')
    n_edge_t = max(int(round(0.05 * nT)), 1)
    ax.axvspan(0, taxis[n_edge_t] * 1e6, color='red', alpha=0.08)
    ax.axvspan(taxis[-n_edge_t] * 1e6, taxis[-1] * 1e6, color='red', alpha=0.08)
    ax.set_xlabel('t (μs)'); ax.set_ylabel('p (MPa)')
    ax.set_title('IC pulse (centre)')
    ax.grid(True); ax.legend(loc='upper right', fontsize=8)

    # 2) Spectrum
    ax = axes[0, 1]
    ax.plot(freqs / 1e6, spec_dB)
    for k in range(1, 4):
        f_k = k * m['f0'] / 1e6
        if f_k < freqs[-1] / 1e6:
            ax.axvline(f_k, color='red', alpha=0.4, linestyle=':')
            ax.text(f_k, -3, f'{k}f₀', rotation=90, va='top', ha='right',
                    fontsize=8, color='red')
    ax.set_ylim(-80, 5); ax.set_xlim(0, min(6 * m['f0'] / 1e6, freqs[-1] / 1e6))
    ax.set_xlabel('f (MHz)'); ax.set_ylabel('|P(f)| (dB)')
    ax.set_title('IC spectrum (centre)')
    ax.grid(True, which='both', alpha=0.4)

    # 3) IC X-T slice
    ax = axes[1, 0]
    im = ax.imshow(initial_field[:, nY // 2, :], aspect='auto', origin='lower',
                   extent=[taxis[0] * 1e6, taxis[-1] * 1e6,
                           xaxis[0] * 1e3, xaxis[-1] * 1e3])
    ax.set_xlabel('t (μs)'); ax.set_ylabel('x (mm)')
    ax.set_title('IC X-T slice')
    fig.colorbar(im, ax=ax, label='Pa')

    # 4) Spatial RMS map
    ax = axes[1, 1]
    im = ax.imshow(rms_xy.T / 1e3, origin='lower', aspect='equal',
                   extent=[xaxis[0] * 1e3, xaxis[-1] * 1e3,
                           yaxis[0] * 1e3, yaxis[-1] * 1e3])
    # Mark spatial absorbing band
    bdy = m['boundaryFactor']
    x_in, x_out = xaxis[int(round(nX * bdy)) - 1] * 1e3, xaxis[-int(round(nX * bdy))] * 1e3
    y_in, y_out = yaxis[int(round(nY * bdy)) - 1] * 1e3, yaxis[-int(round(nY * bdy))] * 1e3
    ax.axvline(x_in, color='red', alpha=0.5, linestyle='--')
    ax.axvline(x_out, color='red', alpha=0.5, linestyle='--')
    ax.axhline(y_in, color='red', alpha=0.5, linestyle='--')
    ax.axhline(y_out, color='red', alpha=0.5, linestyle='--')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('IC spatial RMS (red = abs. layer edge)')
    fig.colorbar(im, ax=ax, label='kPa rms')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------
def _fmt_len(x):
    if not np.isfinite(x):
        return r'$\infty$'
    return f'{x*100:.2f} cm' if x >= 1e-2 else f'{x*1e3:.3f} mm'


def _yn(b):
    return r'\checkmark' if b else r'$\times$'


def _render_tex(m: dict, warnings: list, scenario: str,
                screens_block: str = '') -> str:
    sc_block = (f'\\noindent\\textit{{Scenario:}} '
                f'{_tex_escape(scenario)}\\\\' if scenario else '')

    if warnings:
        items = '\n'.join(r'  \item ' + _tex_escape(w) for w in warnings)
        warn_block = (
            r'\noindent\textbf{\textcolor{red}{Warnings (' f'{len(warnings)}'
            r'):}}' '\n'
            r'\begin{itemize}\setlength{\itemsep}{1pt}' '\n'
            f'{items}\n'
            r'\end{itemize}'
        )
    else:
        warn_block = (r'\noindent\textcolor{teal!70!black}{\textbf{No warnings.}'
                      r' Pre-flight checks passed.}')

    goldberg_str = (f"{m['goldberg']:.2f}" if m['goldberg'] is not None
                    else r'$\infty$')

    return rf"""\documentclass{{article}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage[margin=1.2cm,a4paper]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{xcolor}}
\usepackage{{amsmath,amssymb}}
\setlength{{\parindent}}{{0pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\pagestyle{{empty}}
\begin{{document}}

{{\Large\textbf{{Angular Spectrum Solver — Pre-flight Sanity Report}}}}\\[2pt]
{sc_block}
\textit{{Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
on {_tex_escape(socket.gethostname())} \textbar{{}} Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}}}\\
\vspace{{2pt}}\hrule\vspace{{4pt}}

\noindent
\begin{{minipage}}[t]{{0.49\linewidth}}
\textbf{{Grid}}\\[1pt]
\begin{{tabular}}{{lr}}
\toprule
$n_X \times n_Y \times n_T$ & {m['nX']} $\times$ {m['nY']} $\times$ {m['nT']}\\
$dX,\,dY$ & {m['dX']*1e3:.4f}, {m['dY']*1e3:.4f} mm\\
$dT$ & {m['dT']*1e9:.3f} ns\\
Domain $X \times Y$ & {m['domain_x']*100:.2f} $\times$ {m['domain_y']*100:.2f} cm\\
Time window & {m['t_window']*1e6:.2f} \textmu s\\
$\lambda = c_0/f_0$ & {m['lam']*1e3:.3f} mm\\
pts/$\lambda$ ($X$, $Y$) & {m['pts_per_lambda_x']:.2f}, {m['pts_per_lambda_y']:.2f}\\
samples/cycle & {m['pts_per_cycle']:.2f}\\
CFL ($c_0\,dT/dX$) & {m['cfl']:.4f}\\
\bottomrule
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}
\textbf{{Pulse \& source}}\\[1pt]
\begin{{tabular}}{{lr}}
\toprule
$f_0$ & {m['f0']/1e6:.4f} MHz\\
$\omega_0=2\pi f_0$ & {m['omega0']:.3e} rad/s\\
$p_0=\max\lvert p\rvert$ & {m['p0']/1e6:.4f} MPa\\
Pulse FWHM ($p_{{\text{{dur}}}}$) & {m['pdur']*1e6:.4f} \textmu s\\
Envelope centroid & {m['centroid_t']*1e6:.4f} \textmu s\\
Edge energy (first/last 5\%) & {m['edge_frac_t']*100:.3f} \%\\
Aperture pixels & {m['ap_pixels']}\\
Aperture area & {m['ap_area']*1e6:.3f} mm$^2$\\
Spatial bdy.\ energy frac.\ & {m['spat_bdy_frac']*100:.3f} \%\\
\bottomrule
\end{{tabular}}
\end{{minipage}}

\vspace{{6pt}}
\begin{{minipage}}[t]{{0.49\linewidth}}
\textbf{{Material}}\\[1pt]
\begin{{tabular}}{{lr}}
\toprule
$c_0$ & {m['c0']:.1f} m/s\\
$\rho_0$ & {m['rho0']:.1f} kg/m$^3$\\
$\beta$ (nonlinearity) & {m['beta']:.3f}\\
$\alpha_0$ & {m['alpha_label']}\\
$N=\beta/(2c_0^3\rho_0)$ & {m['N']:.3e}\\
$\alpha$ at $f_0$ & {m['alpha_Np_m']:.4g} Np/m\\
$\ell_{{\text{{atten}}}}=1/\alpha$ & {_fmt_len(m['l_atten'])}\\
$\ell_{{\text{{shock}}}}=\rho_0c_0^3/(\beta\omega_0 p_0)$ & {_fmt_len(m['l_shock'])}\\
Goldberg $\Gamma=\ell_{{\text{{atten}}}}/\ell_{{\text{{shock}}}}$ & {goldberg_str}\\
\bottomrule
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}
\textbf{{Numerics}}\\[1pt]
\begin{{tabular}}{{lr}}\toprule
propDist & {m['propDist']*100:.3f} cm\\
dZmin & {m['dZmin']*1e3:.3f} mm\\
Est.\ \# z-steps & {m['est_steps']}\\
useSplitStep & {_yn(m['useSplitStep'])}\\
useTVD / flux & {_yn(m['useTVD'])} / {_tex_escape(m['fluxScheme'])}\\
useAdaptiveFiltering & {_yn(m['useAdaptiveFiltering'])}\\
boundaryFactor / profile & {m['boundaryFactor']:.2f} / {_tex_escape(m['boundaryProfile'])}\\
useFreqWeightedBdy & {_yn(m['useFreqWeightedBoundary'])}\\
useSuperAbsorbing & {_yn(m['useSuperAbsorbing'])}\\
Obliquity (atten / NL) & {_yn(m['useObliquityCorrection'])} / {_yn(m['useNonlinearityObliquity'])}\\
phase screens / source planes & {m['n_phaseScreens']} / {m['n_sourcePlanes']}\\
init.\ stab.\ margin / thr.\ & {m['stab_init']:.4f} / {m['stab_thr']:.3f}\\
\bottomrule
\end{{tabular}}
\end{{minipage}}

\vspace{{8pt}}
\begin{{center}}
\includegraphics[width=0.96\linewidth]{{preflight_panels.png}}
\end{{center}}

\vspace{{4pt}}
{warn_block}

{screens_block}
\end{{document}}
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preflight_report(initial_field: np.ndarray,
                     params,
                     output_dir: str,
                     scenario: str = '',
                     compile_pdf: bool = True,
                     verbose: bool = True) -> dict:
    """Generate the pre-flight report and (optionally) compile to PDF.

    Returns a dict of the computed metrics and the warning list.
    """
    ensure_dir(output_dir)

    metrics = _compute_metrics(initial_field, params)
    warnings = _build_warnings(metrics)

    panels_path = os.path.join(output_dir, 'preflight_panels.png')
    _plot_panels(initial_field, metrics, panels_path)

    screens_path = None
    screens_block = ''
    screens_result = _plot_screens(params, output_dir)
    if screens_result is not None:
        screens_path, screens_stats = screens_result
        screens_block = _render_screens_block(
            screens_stats, png_relpath=os.path.basename(screens_path))

    tex_path = os.path.join(output_dir, 'preflight.tex')
    with open(tex_path, 'w') as f:
        f.write(_render_tex(metrics, warnings, scenario, screens_block))

    pdf_path = None
    if compile_pdf:
        pdflatex = shutil.which('pdflatex')
        if pdflatex is None:
            if verbose:
                print('[preflight] pdflatex not found; skipping PDF compile')
        else:
            try:
                subprocess.run(
                    [pdflatex, '-interaction=nonstopmode', '-halt-on-error',
                     '-output-directory', output_dir, 'preflight.tex'],
                    cwd=output_dir, capture_output=True, timeout=60, check=True)
                pdf_path = os.path.join(output_dir, 'preflight.pdf')
                # Tidy up auxiliary files
                for ext in ('aux', 'log', 'out'):
                    aux = os.path.join(output_dir, f'preflight.{ext}')
                    if os.path.exists(aux):
                        os.remove(aux)
            except subprocess.CalledProcessError as e:
                if verbose:
                    print('[preflight] pdflatex failed:')
                    print(e.stdout.decode(errors='ignore')[-2000:] if e.stdout else '')

    if verbose:
        print(f'[preflight] panels → {panels_path}')
        if screens_path:
            print(f'[preflight] screens → {screens_path}')
        print(f'[preflight] LaTeX  → {tex_path}')
        if pdf_path:
            print(f'[preflight] PDF    → {pdf_path}')
        print(f'[preflight] {len(warnings)} warning(s)')
        for w in warnings:
            print(f'  ! {w}')

    return {'metrics': metrics, 'warnings': warnings,
            'panels_path': panels_path, 'screens_path': screens_path,
            'tex_path': tex_path, 'pdf_path': pdf_path}
