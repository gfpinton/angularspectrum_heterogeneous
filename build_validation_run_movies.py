"""
Validation-run movies.

For each cached AS-solver validation run, build:
  (1) A 4-panel focal-plane movie showing v, a, |ε̇|, γ_vm time-evolved.
  (2) A propagation movie showing pI(x, y=0, z) at the saved z-grid
       (single still per z, swept as a movie).

Operating points covered:
  T1 V₀ ∈ {0.01, 0.05, 0.10}  (Gaussian source, quasilinear → shocked sweep)
  T3 F#=1   (bowl, tight + shocked)

Outputs to /celerina/gfp/mfs/hyperfocal_loss/:
  movie_T1_v0_<V>.mp4               (focal-plane 4-panel)
  movie_T1_v0_<V>_axial.mp4         (propagation pI xz)
  movie_T3_fn_<F#>.mp4              (focal-plane 4-panel)
  movie_T3_fn_<F#>_axial.mp4        (propagation pI xz)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
from kinematic_analyzer import kinematic_fields
from T1_validate_gaussian import F0, DT, DX, FOCUS

# Movie output dir; override with $HYPERFOCAL_OUT.
OUT = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT, exist_ok=True)
FPS = 25
DPI = 100


def focal_panel_movie(npz_path, out_path, label, downsample=1):
    """4-panel focal-plane movie: v, a, |ε̇|, γ_vm."""
    d = np.load(npz_path)
    field = d['field_focal']
    xaxis = d['xaxis']
    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    x_mm = xaxis * 1e3
    t_ms = (np.arange(nT) - nT//2) * DT * 1e3

    kf = kinematic_fields(field, DT, DX)
    v       = kf['v']
    a       = kf['a']
    eps_mag = kf['eps_mag']
    gam_vm  = kf['gamma_vm']

    panels = [
        ('v',          v,         'RdBu_r', True,  'v (m/s)'),
        ('a',          a,         'RdBu_r', True,  'a (m/s²)'),
        ('|ε̇|',        eps_mag,   'inferno', False, '|ε̇| (1/s)'),
        ('γ_vm',       gam_vm,    'inferno', False, 'γ_vm'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6),
                             constrained_layout=True)
    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    ims = []
    for ax, (name, arr, cmap, sym, ylab) in zip(axes, panels):
        peak = float(np.max(np.abs(arr))) if sym else float(np.max(arr))
        vmin, vmax = (-peak, peak) if sym else (0, peak)
        im = ax.imshow(arr[:, :, 0].T, extent=ext, origin='lower',
                       aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(xlabel='x (mm)', xlim=(-50, 50), ylim=(-50, 50),
               title=name)
        if ax is axes[0]:
            ax.set_ylabel('y (mm)')
        plt.colorbar(im, ax=ax, label=ylab,
                     fraction=0.046, pad=0.04)
        ims.append((im, arr))
    suptitle = fig.suptitle(f'{label}   t = {t_ms[0]:+6.2f} ms',
                            fontsize=11, y=1.02)

    step = max(1, nT // 240)
    frames = list(range(0, nT, step))

    def update(i):
        for (im, arr), (_, _, _, _, _) in zip(ims, panels):
            im.set_data(arr[:, :, i].T)
        suptitle.set_text(f'{label}   t = {t_ms[i]:+6.2f} ms')
        return tuple(im for im, _ in ims) + (suptitle,)

    print(f'  writing {out_path} ({len(frames)} frames)')
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  blit=False, interval=1000/FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=DPI)
    plt.close(fig)


def axial_propagation_movie(npz_path, out_path, label):
    """Show pI(x, y=0, z) as a z-sweep (each frame is the pI slice
    at one z value)."""
    d = np.load(npz_path)
    pI = d['pI']
    xaxis = d['xaxis']
    zaxis = d['zaxis']
    nX, nY, nZ = pI.shape
    cy = nY // 2
    x_mm = xaxis * 1e3
    z_cm = zaxis * 100

    # Use a 2-panel: (a) on-axis pI(z) trace with current-z indicator,
    # (b) pI(x, y=0) slice at current z (1D lateral).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                             constrained_layout=True)
    ax1, ax2 = axes

    on_axis = pI[nX//2, cy, :]
    ax1.plot(z_cm, on_axis / on_axis.max(), 'b-', lw=1.2)
    ax1.set(xlabel='z (cm)', ylabel='on-axis pI / max',
            title=f'(a) on-axis intensity vs z', ylim=(0, 1.05))
    ax1.grid(alpha=0.3)
    cur_z, = ax1.plot([z_cm[0]], [on_axis[0]/on_axis.max()],
                       'ro', markersize=8)

    line, = ax2.plot(x_mm, pI[:, cy, 0]/on_axis.max(), 'k-', lw=1.4)
    ax2.set(xlabel='x (mm)', ylabel='pI(x, y=0) / on-axis peak',
            title=f'(b) lateral pI profile',
            xlim=(-50, 50), ylim=(0, 1.6))
    ax2.grid(alpha=0.3)
    txt = ax2.text(0.05, 0.92, f'z = {z_cm[0]:.2f} cm',
                   transform=ax2.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
    suptitle = fig.suptitle(label, fontsize=11, y=1.02)

    step = max(1, nZ // 200)
    frames = list(range(0, nZ, step))

    def update(i):
        cur_z.set_data([z_cm[i]], [on_axis[i]/on_axis.max()])
        line.set_ydata(pI[:, cy, i]/on_axis.max())
        txt.set_text(f'z = {z_cm[i]:.2f} cm')
        return cur_z, line, txt

    print(f'  writing {out_path} ({len(frames)} frames)')
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  blit=False, interval=1000/FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
    ani.save(out_path, writer=writer, dpi=DPI)
    plt.close(fig)


def main():
    print('Building AS-run validation movies...')

    T1 = os.path.join('validation_results', 'theory_validation',
                      'T1_gaussian')
    T3 = os.path.join('validation_results', 'theory_validation',
                      'T3_fnumber')

    # T1 cases at selected V₀
    for v0 in [0.01, 0.05, 0.10]:
        run_dir = os.path.join(T1, f'v0_{v0:.3f}')
        npz = os.path.join(run_dir, 'nonlinear_focal.npz')
        if not os.path.exists(npz):
            continue
        label = (f'T1 Gaussian source — $V_0={v0}$ m/s '
                 f'(F=10 cm, $w_{{\\rm src}}$=3 cm)')
        focal_panel_movie(npz,
                          os.path.join(OUT, f'movie_T1_v0_{v0:.3f}.mp4'),
                          label)
        axial_propagation_movie(npz,
            os.path.join(OUT, f'movie_T1_v0_{v0:.3f}_axial.mp4'),
            label + ' — pI(x, y=0, z) propagation')

    # T3 F#=1 case (tight + shocked, where central-lobe rule fails)
    npz = os.path.join(T3, 'fn_1.0', 'nonlinear.npz')
    if os.path.exists(npz):
        label = (f'T3 bowl — F\\#=1 ($V_0=0.025$ m/s)')
        focal_panel_movie(npz,
                          os.path.join(OUT, 'movie_T3_fn_1.0.mp4'),
                          label)
        axial_propagation_movie(npz,
            os.path.join(OUT, 'movie_T3_fn_1.0_axial.mp4'),
            label + ' — pI(x, y=0, z) propagation')

    print('Done.')


if __name__ == '__main__':
    main()
