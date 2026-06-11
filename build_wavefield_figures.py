"""
Wavefield reference figures for hyperfocal_kinematic.tex §2.

Shows the four focal-plane kinematic quantities directly so readers can
visualise what v, a, ε̇, γ_vm look like at the operating point of the
worked example (T3 F#=2, V₀=0.025 m/s).

Outputs:
  fig_wavefields_2D.pdf   — 5-panel 2D map: |v|_peak, ⟨v²⟩, ⟨a²⟩,
                              ⟨|ε̇|²⟩, ⟨γ_vm²⟩ at focal plane.
  fig_wavefields_traces.pdf — 4-panel time traces v(t), a(t),
                              |ε̇|(t), γ_vm(t) at on-axis and off-axis.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 7, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
from kinematic_analyzer import kinematic_fields, lateral_2d_maps
from T1_validate_gaussian import F0, DT, DX

# Figure output dir; override with $HYPERFOCAL_OUT.
OUT_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # Use T3 F#=2 nonlinear (worked-example operating point)
    d = np.load('validation_results/theory_validation/T3_fnumber/'
                'fn_2.0/nonlinear.npz')
    field = d['field_focal']
    xaxis = d['xaxis']
    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    x_mm = xaxis * 1e3
    t_ms = np.arange(nT) * DT * 1e3
    t_ms -= t_ms.mean()    # centre on 0

    kf = kinematic_fields(field, DT, DX)
    maps = lateral_2d_maps(field, kf)

    # ---------- (1) 2-D wavefield maps ----------
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), constrained_layout=True)
    quantities = [
        ('v_peaksq',         r'$(\max_t v)^2$',           'm$^2$/s$^2$'),
        ('pI',               r'$\langle v^2\rangle$',     'm$^2$/s$^2$'),
        ('a_peaksq',         r'$\langle a^2\rangle$',     'm$^2$/s$^4$'),
        ('eps_peaksq',       r'$\langle|\dot\varepsilon|^2\rangle$',
                             '1/s$^2$'),
        ('gamma_vm_peaksq',  r'$\langle\gamma_{\rm vm}^2\rangle$',
                             'dimensionless'),
    ]
    # Switch from peaksq to meansq for the time-mean-square ones (a, ε̇, γ).
    # The 'pI' key is already ∫v²dt → mean-square·T. Use the dim-matched maps:
    # use 'a_meansq', 'eps_meansq', 'gamma_vm_meansq' for the mean-square row,
    # which are dimensionally matched (quadratic-in-v) to pI.
    quantities = [
        ('v_peaksq',         r'(a) $(\max_t v)^2$'),
        ('pI',               r'(b) $\langle v^2\rangle$ (pI)'),
        ('a_meansq',         r'(c) $\langle a^2\rangle$'),
        ('eps_meansq',       r'(d) $\langle|\dot\varepsilon|^2\rangle$'),
        ('gamma_vm_meansq',  r'(e) $\langle\gamma_{\rm vm}^2\rangle$'),
    ]
    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    for ax, (key, title) in zip(axes, quantities):
        m = maps[key]
        mn = m / (m.max() + 1e-30)
        im = ax.imshow(10 * np.log10(np.maximum(mn, 1e-3)).T,
                       extent=ext, origin='lower', aspect='equal',
                       cmap='inferno', vmin=-30, vmax=0)
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=title,
               xlim=(-50, 50), ylim=(-50, 50))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='dB rel. peak')
    fig.suptitle('Focal-plane wavefield maps (dim-matched, quadratic in $v$): '
                 'T3 F\\#=2 bowl, $V_0=0.025$ m/s', fontsize=9, y=1.02)
    fig.savefig(os.path.join(OUT_DIR, 'fig_wavefields_2D.pdf'))
    plt.close(fig)
    print('wrote fig_wavefields_2D.pdf')

    # ---------- (2) On-axis vs off-axis time traces ----------
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.0),
                             constrained_layout=True, sharex=True)
    radii_mm = [0, 15, 30]
    colours = ['k', 'tab:blue', 'tab:red']
    quantities2 = [
        ('v',        kf['v'],         r'$v$ (m/s)',           '(a) velocity'),
        ('a',        kf['a'],         r'$a$ (m/s$^2$)',       '(b) acceleration'),
        ('eps_mag',  kf['eps_mag'],   r'$|\dot\varepsilon|$ (1/s)',
                                                              '(c) strain-rate magnitude'),
        ('gamma_vm', kf['gamma_vm'],  r'$\gamma_{\rm vm}$',   '(d) von Mises strain'),
    ]
    for ax, (_, arr, ylab, ttl) in zip(axes, quantities2):
        for r_mm, col in zip(radii_mm, colours):
            idx = int(np.argmin(np.abs(x_mm - r_mm)))
            ax.plot(t_ms, arr[idx, cy, :], color=col, lw=0.9,
                    label=f'$x={x_mm[idx]:+.1f}$ mm')
        ax.set(ylabel=ylab, title=ttl)
        ax.grid(alpha=0.3); ax.axhline(0, color='k', lw=0.3)
        ax.legend(loc='upper right', frameon=False, ncol=3)
    axes[-1].set_xlabel('$t$ (ms)')
    fig.suptitle('Time traces at the focal plane: T3 F\\#=2, $V_0=0.025$ m/s',
                 fontsize=10, y=1.005)
    fig.savefig(os.path.join(OUT_DIR, 'fig_wavefields_traces.pdf'))
    plt.close(fig)
    print('wrote fig_wavefields_traces.pdf')


if __name__ == '__main__':
    main()
