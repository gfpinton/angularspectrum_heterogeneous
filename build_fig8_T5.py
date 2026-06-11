"""
Publication-quality T5 figure: 1-D spherical cascade vs 3-D simulation.

Panel (a): per-V₀ harmonic spectra |V_n|² comparison (1-D vs 3-D).
Panel (b): cascade-moment ⟨n⟩_{a²} 1-D vs 3-D scatter.
Output: hyperfocal_loss/fig8_T5_spherical_1d.pdf
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
T5_ROOT = ('validation_results/theory_validation/T5_spherical_1d')

with open(os.path.join(T5_ROOT, 'sweep_summary.json')) as f:
    rows = json.load(f)

# Figure output dir; override with $HYPERFOCAL_OUT.
OUT_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'fig8_T5_spherical_1d.pdf')

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
ax1, ax2 = axes

# Panel (a): spectra of selected V₀
n_ax = np.arange(1, 16)
colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
for r, c in zip(rows, colours):
    ax1.semilogy(n_ax, np.array(r['Vn2_3d']), 'o-', color=c, lw=1.4,
                 markersize=4, label=f'$V_0={r["v0"]:.3f}$')
    ax1.semilogy(n_ax, np.array(r['Vn2_1d']), 's--', color=c, lw=1.0,
                 markersize=4, markerfacecolor='none')
ax1.set(xlabel='Harmonic $n$', ylabel='$|V_n|^2$ (normalised)',
        xlim=(0.5, 12), ylim=(1e-7, 2),
        title='(a) on-axis spectra: 3-D (solid) vs 1-D spherical (dashed)')
ax1.grid(alpha=0.3, which='both')
ax1.legend(loc='upper right', ncol=2, frameon=True, fontsize=7)

# Panel (b): ⟨n⟩_a² 1-D vs 3-D, log-log
n3d = np.array([r['n_a2_3d'] for r in rows])
n1d = np.array([r['n_a2_1d'] for r in rows])
v0s = [r['v0'] for r in rows]
ax2.loglog([1, 10], [1, 10], 'k--', lw=0.8, label='$y=x$')
for r, c in zip(rows, colours):
    ax2.loglog(r['n_a2_1d'], r['n_a2_3d'], 'o', color=c, markersize=8,
               label=f'$V_0={r["v0"]:.3f}$')
    ax2.annotate(f' $V_0={r["v0"]:.3f}$', (r['n_a2_1d'], r['n_a2_3d']),
                 fontsize=7, textcoords='offset points', xytext=(5, 3))
ax2.set(xlabel='$\\langle n\\rangle_{a^2}$ (1-D spherical)',
        ylabel='$\\langle n\\rangle_{a^2}$ (3-D simulation)',
        xlim=(0.9, 8), ylim=(0.9, 8),
        title='(b) cascade moment: 1-D vs 3-D')
ax2.grid(alpha=0.3, which='both')
ax2.legend(loc='lower right', fontsize=7)

fig.savefig(OUT)
plt.close(fig)
print(f'wrote {OUT}')
