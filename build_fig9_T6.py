"""
Publication figure for T6: clean shock-at-focus immersed-bowl run.

Panel (a): lateral profiles of max_t|u|, max_t|v|, max_t|a| at each
  quantity's z-peak, all normalised; FWHM markers shown.
Panel (b): lateral profiles of max_t|ε̇|, max_t|γ_vm| at their z-peak,
  normalised; ring-radius markers and central-lobe-rule prediction
  marked.
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
from kinematic_analyzer import fwhm

CAP = 'validation_results/clean_shock_at_focus/capture.npz'
# Figure output dir; override with $HYPERFOCAL_OUT.
OUT_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'fig9_T6_shock_at_focus.pdf')

d = np.load(CAP)
z = d['z_caps']
xaxis = d['xaxis']
x_mm = xaxis * 1e3
nX = len(xaxis); cx = cy = nX // 2
LAM_mm = 20.0      # λ = 2 cm = 20 mm
F_mm = 200.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
ax1, ax2 = axes

# Panel (a): u, v, a centred quantities
colours = {'u': 'tab:blue', 'v': 'tab:green', 'a': 'tab:red'}
labels = {'u': r'max$_t$|$u$|', 'v': r'max$_t$|$v$|', 'a': r'max$_t$|$a$|'}
for key, name in zip(['pk_u', 'pk_v', 'pk_a'], ['u', 'v', 'a']):
    M = d[key]
    on_axis = M[:, cx, cy]
    iz = int(np.argmax(on_axis))
    prof = M[iz, :, cy]
    pn = prof / prof.max()
    fw = fwhm(prof, x_mm)
    ax1.plot(x_mm, pn, color=colours[name], lw=1.6,
             label=f'{labels[name]} (FWHM={fw:.2f} mm = {fw/LAM_mm:.3f} λ, '
                   f'z={z[iz]*100:.2f} cm)')
ax1.axhline(0.5, color='gray', lw=0.6, ls=':')
ax1.set(xlabel='x (mm)', ylabel='Normalised',
        xlim=(-50, 50), ylim=(0, 1.05),
        title=r'(a) Centred kinematic quantities at their z-peak')
ax1.grid(alpha=0.3)
ax1.legend(loc='upper right', fontsize=7.5)

# Panel (b): annular quantities + theory ring radius
# Use w_0 from a Gaussian fit of pI at z_focus
pI = d['pI']
z_pI = d['zaxis']
on_axis_pI = pI[cx, cy, :]
i_focus = int(np.argmax(on_axis_pI))
pI_focal = pI[:, cy, i_focus]
fw_pI = fwhm(pI_focal, x_mm)
w0 = fw_pI / np.sqrt(2 * np.log(2))
print(f'pI FWHM = {fw_pI:.2f} mm, Gaussian w_0 = {w0:.2f} mm')

for key, name, sym in [('pk_eps', 'ε̇', r'$\dot\varepsilon$'),
                       ('pk_gam', 'γ_vm', r'$\gamma_{\rm vm}$')]:
    M = d[key]
    lat_max = M.max(axis=(1, 2))
    iz = int(np.argmax(lat_max))
    prof = M[iz, :, cy]
    pn = prof / prof.max()
    i_pos = np.where(xaxis > 0)[0]
    r_ring = float(xaxis[i_pos[int(np.argmax(prof[i_pos]))]]) * 1e3
    ax2.plot(x_mm, pn, lw=1.6,
             label=f'max$_t$|{sym}| (ring at r={r_ring:.2f} mm = '
                   f'{r_ring/LAM_mm:.3f} λ, z={z[iz]*100:.2f} cm)')
# Central-lobe prediction r_ring = w_0/√2 for single-harmonic
r_th = w0 / np.sqrt(2)
ax2.axvline(r_th, color='k', ls='--', lw=1.0,
            label=f'theory $w_0/\\sqrt{{2}}$ = {r_th:.2f} mm')
ax2.axvline(-r_th, color='k', ls='--', lw=1.0)
ax2.set(xlabel='x (mm)', ylabel='Normalised',
        xlim=(-50, 50), ylim=(0, 1.05),
        title=r'(b) Annular kinematic quantities + theory ring')
ax2.grid(alpha=0.3)
ax2.legend(loc='upper right', fontsize=7.5)

fig.savefig(OUT)
plt.close(fig)
print(f'wrote {OUT}')
