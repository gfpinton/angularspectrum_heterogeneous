"""
Build publication-quality validation figures for hyperfocal_kinematic.tex.

Outputs PDFs into /celerina/gfp/mfs/hyperfocal_loss/, named:
  fig5_validation_perfharmonic.pdf — T1 central-lobe-rule σ_n test
  fig6_predicted_vs_measured.pdf   — T1 + T3 combined predicted/measured
  fig7_grid_convergence.pdf        — T2 grid stability
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Consistent publication style
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
from T1_validate_gaussian import F0, DT, V0_LIST as V0_LIST_T1
from kinematic_analyzer import fwhm

TV_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation',
)
# Manuscript figure output dir; override with $HYPERFOCAL_OUT.
OUT_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT_DIR, exist_ok=True)


def fig5_central_lobe():
    """Per-harmonic σ_n vs central-lobe-rule prediction σ_0/√n."""
    odd_n = [1, 3, 5, 7, 9, 11, 13]
    rows = []
    for v0 in V0_LIST_T1:
        p = os.path.join(TV_ROOT, 'T1_gaussian', f'v0_{v0:.3f}',
                         'nonlinear_focal.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        field = d['field_focal']
        xaxis = d['xaxis']
        nX, nY, nT = field.shape
        cy = nY // 2
        x_mm = xaxis * 1e3
        spec_xy = np.abs(np.fft.rfft(field, axis=-1))**2
        freqs = np.fft.rfftfreq(nT, DT)
        bin_w = 0.4 * F0
        sigmas = []
        for n in odd_n:
            m = (freqs >= n * F0 - bin_w) & (freqs <= n * F0 + bin_w)
            prof = np.sum(spec_xy[:, cy, :][:, m], axis=-1)
            if prof.max() <= 0:
                sigmas.append(float('nan')); continue
            f = fwhm(prof, x_mm)
            sigmas.append(f / np.sqrt(2*np.log(2)) if f == f else float('nan'))
        rows.append((v0, np.array(sigmas)))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    ax1, ax2 = axes

    colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
    for (v0, sigmas), c in zip(rows, colours):
        sig0 = sigmas[0]
        if not (sig0 == sig0): continue
        ratios = sigmas / (sig0 / np.sqrt(np.array(odd_n)))
        mask = np.isfinite(ratios) & (ratios > 0)
        ns = np.array(odd_n)[mask]; rr = ratios[mask]
        ax1.plot(ns, rr, 'o-', color=c, lw=1.3, ms=5,
                 label=f'V_0={v0:.3f}')
    ax1.axhline(1.0, color='k', ls='--', lw=1.0,
                label=r'$\sigma_0/\sqrt{n}$ rule')
    ax1.set(xlabel=r'Harmonic index $n$',
            ylabel=r'$\sigma_n\, /\, (\sigma_0/\sqrt{n})$',
            xscale='log', yscale='log',
            ylim=(0.4, 5),
            title=r'(a) per-harmonic $\sigma_n$ vs central-lobe rule')
    ax1.grid(alpha=0.3, which='both')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, ncol=2)

    # Panel b: σ_n vs n absolute, with theory line for lowest V₀
    for (v0, sigmas), c in zip(rows, colours):
        sig0 = sigmas[0]
        if not (sig0 == sig0): continue
        mask = np.isfinite(sigmas) & (sigmas > 0)
        ns = np.array(odd_n)[mask]; ss = sigmas[mask]
        ax2.plot(ns, ss, 'o-', color=c, lw=1.3, ms=5,
                 label=f'V_0={v0:.3f}')
    # Reference line: σ_0/√n based on the V₀=0.01 (linear) baseline
    sigma0_ref = rows[0][1][0]   # V₀=0.01, n=1
    if sigma0_ref == sigma0_ref:
        ns_smooth = np.linspace(1, 15, 60)
        ax2.plot(ns_smooth, sigma0_ref/np.sqrt(ns_smooth),
                 'k--', lw=1.0, label=r'$\sigma_0/\sqrt{n}$ theory')
    ax2.set(xlabel=r'Harmonic index $n$',
            ylabel=r'$\sigma_n$ (mm)',
            xscale='log', yscale='log',
            title=r'(b) measured $\sigma_n$ vs $n$')
    ax2.grid(alpha=0.3, which='both')
    ax2.legend(loc='lower left', frameon=True, framealpha=0.9, ncol=2)

    fig.savefig(os.path.join(OUT_DIR, 'fig5_validation_perfharmonic.pdf'))
    plt.close(fig)
    print(f'wrote fig5_validation_perfharmonic.pdf')


def fig6_predicted_vs_measured():
    """T1 + T3 combined: predicted vs measured ratios."""
    # T1 sweep_summary
    with open(os.path.join(TV_ROOT, 'T1_gaussian', 'sweep_summary.json')) as f:
        t1 = json.load(f)
    with open(os.path.join(TV_ROOT, 'T3_fnumber', 'sweep_summary.json')) as f:
        t3 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    ax1, ax2 = axes

    # Panel a: T1 — w_a/w_v predicted vs measured, label by V₀
    v0s = [r['v0'] for r in t1]
    p_wa = [r['predicted']['wa_over_wv'] for r in t1]
    m_wa = [r['measured']['wa_over_wv'] for r in t1]
    ax1.plot([0, 1.2], [0, 1.2], 'k--', lw=0.8,
             label=r'$y\,=\,x$')
    for v0, p, m in zip(v0s, p_wa, m_wa):
        ax1.plot(p, m, 'o', markersize=7,
                 color='C0' if v0 <= 0.03 else 'C3')
        ax1.annotate(f'V_0={v0:.3f}', (p, m),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    # T3 overlay
    p_t3 = [r['pred_wa_over_wv'] for r in t3]
    m_t3 = [r['meas_wa_over_wv'] for r in t3]
    fns  = [r['fn'] for r in t3]
    for fn, p, m in zip(fns, p_t3, m_t3):
        ax1.plot(p, m, 's', markersize=7,
                 color='C2' if fn >= 1.5 else 'C3',
                 markerfacecolor='none')
        ax1.annotate(f'F#={fn:.1f}', (p, m),
                     textcoords='offset points', xytext=(-25, 5), fontsize=7)
    ax1.set(xlabel=r'predicted $w_a/w_v$',
            ylabel=r'measured $w_a/w_v$',
            xlim=(0.4, 1.05), ylim=(0.4, 1.1),
            title='(a) $w_a/w_v$: T1 (circles), T3 (squares)')
    ax1.grid(alpha=0.3)
    # Custom legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], color='k', ls='--', lw=0.8, label='y=x'),
        Line2D([0],[0], marker='o', color='C0', lw=0, label='quasilinear (T1)'),
        Line2D([0],[0], marker='o', color='C3', lw=0, label='shocked (T1)'),
        Line2D([0],[0], marker='s', color='C2', lw=0,
               markerfacecolor='none', label='paraxial (T3)'),
        Line2D([0],[0], marker='s', color='C3', lw=0,
               markerfacecolor='none', label='tight F# (T3)'),
    ]
    ax1.legend(handles=legend_handles, loc='upper left', frameon=True)

    # Panel b: same for A_a²/A_v²
    p_A1 = [r['predicted']['Aa2_over_Av2'] for r in t1]
    m_A1 = [r['measured']['Aa2_over_Av2'] for r in t1]
    p_A3 = [r['pred_Aa_over_Av'] for r in t3]
    m_A3 = [r['meas_Aa_over_Av'] for r in t3]
    ax2.plot([0, 1.5], [0, 1.5], 'k--', lw=0.8)
    for v0, p, m in zip(v0s, p_A1, m_A1):
        ax2.plot(p, m, 'o', markersize=7,
                 color='C0' if v0 <= 0.03 else 'C3')
        ax2.annotate(f'V_0={v0:.3f}', (p, m),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    for fn, p, m in zip(fns, p_A3, m_A3):
        ax2.plot(p, m, 's', markersize=7,
                 color='C2' if fn >= 1.5 else 'C3',
                 markerfacecolor='none')
        ax2.annotate(f'F#={fn:.1f}', (p, m),
                     textcoords='offset points', xytext=(-25, 5), fontsize=7)
    ax2.set(xlabel=r'predicted $A_{a^2}/A_{v^2}$',
            ylabel=r'measured $A_{a^2}/A_{v^2}$',
            xlim=(0.2, 1.1), ylim=(0.2, 1.3),
            title=r'(b) half-max area ratio')
    ax2.grid(alpha=0.3)

    fig.savefig(os.path.join(OUT_DIR, 'fig6_predicted_vs_measured.pdf'))
    plt.close(fig)
    print('wrote fig6_predicted_vs_measured.pdf')


def fig7_grid_convergence():
    """T2 grid stability at V₀=0.025."""
    with open(os.path.join(TV_ROOT, 'T2_grid_gaussian',
                           'convergence_runs.json')) as f:
        runs = json.load(f)
    Rs = [r['R'] for r in runs]
    DX_mm = [r['DX']*1e3 for r in runs]
    DT_us = [r['DT']*1e6 for r in runs]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
    ax1, ax2 = axes

    # Panel a: w_a/w_v at each R
    ax1.plot(Rs, [r['pred_wa_over_wv'] for r in runs], 'b-o',
             lw=1.4, label='predicted', markersize=6)
    ax1.plot(Rs, [r['meas_wa_over_wv'] for r in runs], 'r-s',
             lw=1.4, label='measured', markersize=6)
    ax1.set(xlabel='Refinement R',
            ylabel=r'$w_a/w_v$',
            title=r'(a) $w_a/w_v$ vs grid refinement',
            ylim=(0.955, 0.975))
    ax1.grid(alpha=0.3); ax1.legend()

    # Panel b: A_a²/A_v²
    ax2.plot(Rs, [r['pred_Aa_over_Av'] for r in runs], 'b-o',
             lw=1.4, label='predicted', markersize=6)
    ax2.plot(Rs, [r['meas_Aa_over_Av'] for r in runs], 'r-s',
             lw=1.4, label='measured', markersize=6)
    ax2.set(xlabel='Refinement R',
            ylabel=r'$A_{a^2}/A_{v^2}$',
            title=r'(b) $A_{a^2}/A_{v^2}$ vs grid refinement',
            ylim=(0.92, 0.94))
    ax2.grid(alpha=0.3); ax2.legend()

    fig.savefig(os.path.join(OUT_DIR, 'fig7_grid_convergence.pdf'))
    plt.close(fig)
    print('wrote fig7_grid_convergence.pdf')


def main():
    print(f'Output dir: {OUT_DIR}')
    fig5_central_lobe()
    fig6_predicted_vs_measured()
    fig7_grid_convergence()


if __name__ == '__main__':
    main()
