"""
Wavefield movies + frame-sequence panels for hyperfocal_kinematic.tex.

Uses the T3 F#=2 cached nonlinear focal-plane field (V₀=0.025, the
worked-example operating point) and produces:

  movie_v.mp4         — v(x, y, t)
  movie_u.mp4         — u(x, y, t) = ∫_0^t v dt'        (displacement)
  movie_a.mp4         — a(x, y, t) = ∂_t v               (acceleration)
  movie_eps.mp4       — |ε̇|(x, y, t)                   (strain-rate magn)
  movie_gamma.mp4     — γ_vm(x, y, t)                  (von Mises strain)

Plus a static figure showing 6 frames of |ε̇| at chosen instants for
the .tex (LaTeX cannot embed movies):

  fig_ring_development.pdf  — |ε̇|(x, y, t) at t = -8, -4, 0, +4, +8 ms

All outputs go to /celerina/gfp/mfs/hyperfocal_loss/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
from kinematic_analyzer import kinematic_fields
from T1_validate_gaussian import F0, DT, DX

# Movie output dir; override with $HYPERFOCAL_OUT.
OUT_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(OUT_DIR, exist_ok=True)
FPS = 25
DPI = 110


def make_movie(arr3d, x_mm, t_ms, title, ylab, fname,
               cmap='RdBu_r', symmetric=True, db=False):
    """Animate a (nX, nY, nT) field as a 2-D map over t.

    symmetric=True → diverging colour map about 0.
    db=True → 10 log10(|·|² / peak), useful for strictly positive maps.
    """
    nT = arr3d.shape[-1]
    if db:
        peak = np.max(arr3d**2)
        data = lambda i: 10*np.log10(np.maximum(arr3d[:,:,i]**2, peak*1e-3)/peak)
        vmin, vmax = -25, 0
        cbar_label = 'dB rel. peak'
    elif symmetric:
        peak = np.max(np.abs(arr3d))
        data = lambda i: arr3d[:,:,i]
        vmin, vmax = -peak, peak
        cbar_label = ylab
    else:
        peak = np.max(arr3d)
        data = lambda i: arr3d[:,:,i]
        vmin, vmax = 0, peak
        cbar_label = ylab

    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    im = ax.imshow(data(0).T, extent=ext, origin='lower',
                   aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel='x (mm)', ylabel='y (mm)', xlim=(-50, 50), ylim=(-50, 50))
    title_obj = ax.set_title(f'{title}   t = {t_ms[0]:+6.2f} ms', fontsize=10)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    # Downsample frames for reasonable movie length (~200 frames)
    step = max(1, nT // 240)
    frames = list(range(0, nT, step))

    def update(i):
        im.set_data(data(i).T)
        title_obj.set_text(f'{title}   t = {t_ms[i]:+6.2f} ms')
        return im, title_obj

    print(f'  writing {fname} ({len(frames)} frames)...')
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  blit=False, interval=1000/FPS)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2400)
    ani.save(os.path.join(OUT_DIR, fname), writer=writer, dpi=DPI)
    plt.close(fig)


def make_frame_sequence(arr3d, x_mm, t_ms, t_choices_ms, title, fname,
                        cmap='RdBu_r', symmetric=True, db=False,
                        zoom_mm=50):
    """Static figure: arr3d sampled at several t values, side by side."""
    nT = arr3d.shape[-1]
    indices = [int(np.argmin(np.abs(t_ms - t))) for t in t_choices_ms]
    if db:
        peak = np.max(arr3d**2)
        scale = lambda x: 10*np.log10(np.maximum(x**2, peak*1e-3)/peak)
        vmin, vmax = -25, 0
        cbar_label = 'dB rel. peak'
    elif symmetric:
        peak = np.max(np.abs(arr3d))
        scale = lambda x: x
        vmin, vmax = -peak, peak
        cbar_label = 'amplitude'
    else:
        peak = np.max(arr3d)
        scale = lambda x: x
        vmin, vmax = 0, peak
        cbar_label = 'amplitude'

    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    n_panels = len(indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.4*n_panels, 2.7),
                             constrained_layout=True)
    if n_panels == 1: axes = [axes]
    for ax, i in zip(axes, indices):
        im = ax.imshow(scale(arr3d[:,:,i]).T, extent=ext, origin='lower',
                       aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(xlabel='x (mm)',
               xlim=(-zoom_mm, zoom_mm), ylim=(-zoom_mm, zoom_mm))
        ax.set_title(f't = {t_ms[i]:+5.1f} ms', fontsize=9)
        ax.set_aspect('equal')
    axes[0].set_ylabel('y (mm)')
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                        label=cbar_label)
    fig.suptitle(title, fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    print(f'Loading T3 F#=2 nonlinear field...')
    d = np.load('validation_results/theory_validation/T3_fnumber/'
                'fn_2.0/nonlinear.npz')
    field = d['field_focal']
    xaxis = d['xaxis']
    nX, nY, nT = field.shape
    x_mm = xaxis * 1e3
    # Time axis centred on 0
    t_ms = (np.arange(nT) - nT//2) * DT * 1e3
    print(f'  field shape: {field.shape}, t_range: '
          f'[{t_ms[0]:.1f}, {t_ms[-1]:.1f}] ms')

    kf = kinematic_fields(field, DT, DX)
    v       = kf['v']                # already = field
    a       = kf['a']                # ∂v/∂t
    u       = kf['u']                # ∫v dt
    eps_mag = kf['eps_mag']          # |ε̇|
    gam_vm  = kf['gamma_vm']         # γ_vm

    # MOVIES
    print(f'\nGenerating MP4 movies (fps={FPS}, dpi={DPI})...')
    make_movie(v, x_mm, t_ms,
               r'$v(x, y, t)$ — velocity (m/s)',
               'v (m/s)', 'movie_v.mp4', symmetric=True)
    make_movie(u, x_mm, t_ms,
               r'$u(x, y, t)$ — displacement (m)',
               'u (m)', 'movie_u.mp4', symmetric=True)
    make_movie(a, x_mm, t_ms,
               r'$a(x, y, t)$ — acceleration (m/s$^2$)',
               'a (m/s²)', 'movie_a.mp4', symmetric=True)
    make_movie(eps_mag, x_mm, t_ms,
               r'$|\dot\varepsilon|(x, y, t)$ — strain-rate magnitude (1/s)',
               '|ε̇| (1/s)', 'movie_eps.mp4', symmetric=False, cmap='inferno')
    make_movie(gam_vm, x_mm, t_ms,
               r'$\gamma_{\rm vm}(x, y, t)$ — von Mises shear strain',
               'γ_vm', 'movie_gamma.mp4', symmetric=False, cmap='inferno')

    # STATIC PANEL — ring development for the .tex
    # Pick instants spanning the pulse: t = -10, -5, 0, +5, +10, +15 ms
    t_choices = [-15.0, -8.0, -3.0, 0.0, +5.0, +12.0]
    print(f'\nGenerating static frame-sequence panel...')
    make_frame_sequence(eps_mag, x_mm, t_ms, t_choices,
                        title=r'$|\dot\varepsilon|(x, y, t)$ — annular ring '
                              r'development at focal plane '
                              r'(T3 F\#=2, $V_0=0.025$ m/s)',
                        fname='fig_ring_development.pdf',
                        cmap='inferno', symmetric=False)
    make_frame_sequence(v, x_mm, t_ms, t_choices,
                        title=r'$v(x, y, t)$ — velocity time-evolution at '
                              r'focal plane (T3 F\#=2, $V_0=0.025$ m/s)',
                        fname='fig_velocity_evolution.pdf',
                        cmap='RdBu_r', symmetric=True)

    print(f'\nOutputs in {OUT_DIR}/movie_*.mp4 and fig_ring_development.pdf')


if __name__ == '__main__':
    main()
