"""
Clean immersed-bowl simulation with shock formation at focus.

Setup
-----
  Brain shear regime: F0=100 Hz, c=2 m/s, ρ=1000 kg/m³, β₃=5×10⁸,
                      α₀=0.5 (linear-in-f shear loss).
  Focal length F = 10·λ = 20 cm.
  Bowl half-angle 45° → aperture radius a = F·sin(45°) ≈ 14.14 cm,
                       F# = F/(2a) ≈ 0.707,
                       cap depth = F·(1−cos 45°) ≈ 5.86 cm.
  Source: IMMERSED bowl via make_bowl_source_planes (true curved
          source injected as z-slices, not flat-projection).
  Source amplitude V₀ chosen so the cumulative shock parameter
          σ_f ≈ 1 at z = F (wave starts shocking at the focal plane).

Diagnostics
-----------
  AS solver: diagnostic=True → produces per-z-step frames and the
             summary panels (dashboard, harmonics, Isppa_ax, MI,
             pIxz, stability, etc.) in `as_diagnostic_frames/`.

  Per-step callback: every K z-steps, materialise the field and
             reduce to max_t |·| 2-D maps for each kinematic quantity:
                  pI(x,y,z)            (from solver's pI volume)
                  max_t |v|(x,y,z)
                  max_t |a|(x,y,z)
                  max_t |u|(x,y,z)       u = ∫v dt (displacement)
                  max_t |ε̇|(x,y,z)      |∇v|
                  max_t |γ_vm|(x,y,z)   |∇u|

Movies
------
  One 2-D z-sweep movie per quantity (6 total).  Each frame is the
  2-D map at one captured z; the movie sweeps through z showing
  the kinematic focus formation.

  Output: /celerina/gfp/mfs/hyperfocal_loss/
    clean_shock_pI.mp4
    clean_shock_v_peak.mp4
    clean_shock_a_peak.mp4
    clean_shock_u_peak.mp4
    clean_shock_eps_peak.mp4
    clean_shock_gamma_peak.mp4
"""
import os
import sys
import time
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'mathtext.fontset': 'cm',
})

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source_planes,
)

# ---- Physical regime ----
F0      = 100.0
C0      = 2.0
RHO0    = 1000.0
BETA3   = 5.0e8
ALPHA0  = 0.5
Y_ABS   = 1.0
NCYCLES = 4

# ---- Geometry ----
LAM      = C0 / F0                        # 2 cm
FOCUS    = 10 * LAM                       # 20 cm
HALF_ANG = np.deg2rad(45.0)
OUTER_R  = FOCUS * np.sin(HALF_ANG)       # 14.14 cm
ROC      = FOCUS                          # bowl radius of curvature = focal length
BOWL_DEPTH_GEOM = FOCUS * (1.0 - np.cos(HALF_ANG))  # 5.86 cm
F_NUMBER = FOCUS / (2.0 * OUTER_R)        # 0.707

# ---- Numerics ----
DX        = LAM / 15                      # 1.33 mm
DT        = 1.0 / (60.0 * F0)             # 0.167 ms
# Domain must accommodate aperture diameter 2a ≈ 28.3 cm with buffer.
# Round up to 16·λ = 32 cm. With DX=λ/15, nX = 240.
DOMAIN    = 16 * LAM
PROP_DIST = 1.10 * FOCUS                  # propagate just past focus

# ---- V₀ tuning: target shock at focus ----
# Rough estimate (paraxial): focal_gain = k₀·a²/(2F),
#   V_focal = V₀·focal_gain, σ_f = (β_3/(c⁵ρ²))·V_focal²·k₀·z_R
# For σ_f ≈ 1 at focus: V₀ ≈ 0.015 m/s (refine empirically)
V0_DEFAULT = 0.015

# Capture frequency for kinematic snapshots
CAPTURE_INTERVAL = 25  # capture every 25 z-steps

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'clean_shock_at_focus',
)
os.makedirs(OUT_ROOT, exist_ok=True)
# Movie output dir; override with $HYPERFOCAL_OUT.
MOVIE_DIR = os.environ.get('HYPERFOCAL_OUT', os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'hyperfocal_loss')))
os.makedirs(MOVIE_DIR, exist_ok=True)


class KinematicCapture:
    """Callback that materialises v(x,y,t) at every Kth z-step and
    reduces it to max_t |·| 2D maps for each kinematic quantity.

    Stores in memory; ~5 MB per capture × ~50 captures = ~250 MB.
    """
    def __init__(self, capture_every: int, dt: float, dx: float):
        self.capture_every = int(capture_every)
        self.dt = float(dt)
        self.dx = float(dx)
        self.z_list  = []
        self.pk_v    = []     # max_t |v|
        self.pk_a    = []     # max_t |a|
        self.pk_u    = []     # max_t |u|
        self.pk_eps  = []     # max_t |ε̇|
        self.pk_gam  = []     # max_t |γ_vm|

    def __call__(self, cc, z_cum, field):
        if int(cc) % self.capture_every != 0:
            return
        # Materialise field to numpy
        v = np.asarray(field, dtype=np.float32)   # (nX, nY, nT)
        # max_t |v|
        pk_v = np.max(np.abs(v), axis=-1)
        # acceleration a = ∂v/∂t
        a = np.gradient(v, self.dt, axis=2).astype(np.float32)
        pk_a = np.max(np.abs(a), axis=-1)
        del a
        # displacement u = ∫v dt (running)
        u = (np.cumsum(v, axis=2) * self.dt).astype(np.float32)
        pk_u = np.max(np.abs(u), axis=-1)
        # strain rate magnitude |∇v|
        ex = np.gradient(v, self.dx, axis=0).astype(np.float32)
        ey = np.gradient(v, self.dx, axis=1).astype(np.float32)
        eps_mag = np.sqrt(ex*ex + ey*ey)
        del ex, ey
        pk_eps = np.max(eps_mag, axis=-1)
        del eps_mag
        # von Mises shear strain |∇u|
        gx = np.gradient(u, self.dx, axis=0).astype(np.float32)
        gy = np.gradient(u, self.dx, axis=1).astype(np.float32)
        gam_vm = np.sqrt(gx*gx + gy*gy)
        del gx, gy, u
        pk_gam = np.max(gam_vm, axis=-1)
        del gam_vm

        self.z_list.append(float(z_cum))
        self.pk_v.append(pk_v)
        self.pk_a.append(pk_a)
        self.pk_u.append(pk_u)
        self.pk_eps.append(pk_eps)
        self.pk_gam.append(pk_gam)


def run(v0=V0_DEFAULT):
    diag_dir = os.path.join(OUT_ROOT, 'as_diagnostic_frames')
    if os.path.exists(diag_dir):
        shutil.rmtree(diag_dir)
    os.makedirs(diag_dir, exist_ok=True)

    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT

    print(f'Geometry:')
    print(f'  F          = {FOCUS*100:.2f} cm = {FOCUS/LAM:.1f}·λ')
    print(f'  half-angle = {np.rad2deg(HALF_ANG):.1f}°')
    print(f'  aperture a = {OUTER_R*100:.2f} cm = {OUTER_R/LAM:.2f}·λ')
    print(f'  ROC = F    = {ROC*100:.2f} cm')
    print(f'  bowl depth = {BOWL_DEPTH_GEOM*100:.2f} cm = '
          f'{BOWL_DEPTH_GEOM/FOCUS:.3f}·F')
    print(f'  F#         = {F_NUMBER:.3f}')
    print(f'  V₀         = {v0} m/s')
    print(f'Grid: nX={nX} × {nX} × nT={nT}, DX={DX*1e3:.2f} mm, '
          f'DT={DT*1e3:.3f} ms, domain={DOMAIN*100:.0f} cm')
    print(f'PROP_DIST = {PROP_DIST*100:.2f} cm')

    # ---- Immersed bowl source (plane-by-plane) ----
    dZ_slice = LAM / 10.0     # 2 mm slice thickness
    print(f'\nBuilding immersed bowl source (dZ_slice = {dZ_slice*1e3:.2f} mm)...')
    source_planes, bowl_depth = make_bowl_source_planes(
        xaxis, xaxis, taxis, F0, C0, v0,
        radius=OUTER_R, roc=ROC, dZ=dZ_slice, focus=FOCUS,
        ncycles=NCYCLES, dur=4,
    )
    print(f'  bowl depth (computed): {bowl_depth*100:.2f} cm, '
          f'{len(source_planes)} slices')

    initial_field = source_planes[0][1]
    inject_planes = source_planes[1:]

    # ---- Solver params ----
    N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * OUTER_R**2 / (2 * FOCUS)
    expected_peak = max(v0 * focus_gain, v0)
    safe_dZ = 0.15 * DT / max(expected_peak**2 * N3, 1e-30)
    dZmin = max(0.25 * safe_dZ, DT * C0 / 4.0)
    dZmin = min(dZmin, LAM / 10.0)
    print(f'\ndZmin = {dZmin*1e3:.3f} mm  (expected V_focal ≈ '
          f'{expected_peak:.4f} m/s)')

    sp = SolverParams(
        dX=DX, dY=DX, dT=DT, c0=C0, rho0=RHO0,
        beta=0.0, beta3=BETA3, nonlinearityOrder=3,
        alpha0=ALPHA0, attenPow=Y_ABS, f0=F0,
        propDist=PROP_DIST,
        useSplitStep=True, fluxScheme='kt', useTVD=True,
        useAdaptiveFiltering=True,
        boundaryProfile='quadratic',
        useFreqWeightedBoundary=False,
        useSuperAbsorbing=False, boundaryFactor=0.15,
        dZmin=dZmin, useGPUReductions=False, useAttenLoss=True,
        sourcePlanes=inject_planes,
        # AS validation flag ON
        diagnostic=True,
        diagnosticInterval=50,
        diagnosticDir=diag_dir,
        diagnosticSummary=True,
        diagnosticInitialConditions=True,
    )

    print(f'\nLaunching solve with per-step callback (capture every '
          f'{CAPTURE_INTERVAL} z-steps)...')
    cap = KinematicCapture(CAPTURE_INTERVAL, DT, DX)
    t0 = time.time()
    field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
        initial_field, sp, verbose=False, per_step_callback=cap)
    wall = time.time() - t0
    print(f'  solve done in {wall:.0f}s, {len(zaxis)} z-steps, '
          f'{len(cap.z_list)} kinematic snapshots')

    # ---- Diagnostics ----
    # On-axis pI peak
    cx = cy = nX // 2
    on_axis_pI = pI[cx, cy, :]
    z_focus_idx = int(np.argmax(on_axis_pI))
    z_focus = float(zaxis[z_focus_idx])
    print(f'\nz_focus (on-axis pI max) = {z_focus*100:.2f} cm '
          f'(geometric F = {FOCUS*100:.0f} cm)')
    # Focal trace
    focal_trace = field[cx, cy, :]
    v_peak = float(np.max(np.abs(focal_trace)))
    a_peak = float(np.max(np.abs(np.gradient(focal_trace, DT))))
    edge_steep = a_peak / (2 * np.pi * F0 * max(v_peak, 1e-30))
    print(f'On-axis at z_end: v_peak = {v_peak:.4f} m/s, '
          f'a_peak = {a_peak:.3e} m/s², edge_steepness = {edge_steep:.2f}')

    # Persist captures and pI volume
    save_path = os.path.join(OUT_ROOT, 'capture.npz')
    np.savez(save_path,
             z_caps=np.array(cap.z_list),
             pk_v=np.stack(cap.pk_v),
             pk_a=np.stack(cap.pk_a),
             pk_u=np.stack(cap.pk_u),
             pk_eps=np.stack(cap.pk_eps),
             pk_gam=np.stack(cap.pk_gam),
             pI=pI.astype(np.float32),
             zaxis=zaxis.astype(np.float32),
             xaxis=xaxis.astype(np.float32),
             v0=v0,
             v_peak=v_peak, a_peak=a_peak, edge_steep=edge_steep,
             z_focus_m=z_focus,
             bowl_depth=bowl_depth,
             wall_s=wall)
    print(f'\nWrote captures to {save_path}')

    with open(os.path.join(OUT_ROOT, 'run_metrics.json'), 'w') as f:
        json.dump(dict(
            v0=v0, focus_cm=FOCUS*100, half_angle_deg=45.0,
            aperture_cm=OUTER_R*100, F_number=F_NUMBER,
            bowl_depth_cm=bowl_depth*100,
            n_slices=len(source_planes),
            n_z_steps=len(zaxis), n_captures=len(cap.z_list),
            wall_s=wall,
            z_focus_cm=z_focus*100,
            v_peak=v_peak, a_peak=a_peak, edge_steep=edge_steep,
        ), f, indent=2)
    return save_path, xaxis


def make_zsweep_movie(arr_zxy, z_array, xaxis, title, cmap, fname,
                      symmetric=False, log_scale=False, z_focus=None):
    """arr_zxy shape (nZ_caps, nX, nY). Each frame is one 2D map."""
    nZ = arr_zxy.shape[0]
    x_mm = xaxis * 1e3
    z_cm = z_array * 100

    if log_scale:
        peak = float(np.max(arr_zxy))
        data = lambda i: 10 * np.log10(
            np.maximum(arr_zxy[i], peak * 1e-3) / peak)
        vmin, vmax = -30, 0
        clab = 'dB rel. global peak'
    elif symmetric:
        peak = float(np.max(np.abs(arr_zxy)))
        data = lambda i: arr_zxy[i]
        vmin, vmax = -peak, peak
        clab = 'amplitude'
    else:
        peak = float(np.max(arr_zxy))
        data = lambda i: arr_zxy[i]
        vmin, vmax = 0, peak
        clab = 'amplitude'

    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    im = ax.imshow(data(0).T, extent=ext, origin='lower',
                   aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           xlim=(-100, 100), ylim=(-100, 100))
    t_title = ax.set_title(f'{title}   z = {z_cm[0]:6.2f} cm', fontsize=10)
    plt.colorbar(im, ax=ax, label=clab, fraction=0.046, pad=0.04)

    # Optionally mark focal plane location
    if z_focus is not None:
        focal_marker = ax.text(0.02, 0.97,
                               f'F = {z_focus*100:.1f} cm',
                               transform=ax.transAxes, fontsize=8,
                               color='cyan', verticalalignment='top',
                               bbox=dict(facecolor='black', alpha=0.5))

    def update(i):
        im.set_data(data(i).T)
        t_title.set_text(f'{title}   z = {z_cm[i]:6.2f} cm')
        return im, t_title

    print(f'  writing {fname} ({nZ} frames)')
    ani = animation.FuncAnimation(fig, update, frames=nZ,
                                  interval=80, blit=False)
    writer = animation.FFMpegWriter(fps=15, bitrate=2400)
    ani.save(os.path.join(MOVIE_DIR, fname), writer=writer, dpi=110)
    plt.close(fig)


def make_pI_movie(pI_volume, zaxis, xaxis, fname, z_focus=None):
    """pI volume has shape (nX, nY, nZ). Sweep over z."""
    nX, nY, nZ = pI_volume.shape
    x_mm = xaxis * 1e3
    z_cm = zaxis * 100
    cy = nY // 2

    peak = float(pI_volume.max())
    ext = [x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]]
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    im = ax.imshow(
        10 * np.log10(np.maximum(pI_volume[:, :, 0], peak * 1e-3) / peak).T,
        extent=ext, origin='lower', aspect='equal',
        cmap='inferno', vmin=-30, vmax=0)
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           xlim=(-100, 100), ylim=(-100, 100))
    t_title = ax.set_title(f'pI(x, y, z)   z = {z_cm[0]:6.2f} cm', fontsize=10)
    plt.colorbar(im, ax=ax, label='dB rel. global peak',
                 fraction=0.046, pad=0.04)
    if z_focus is not None:
        ax.text(0.02, 0.97, f'F = {z_focus*100:.1f} cm',
                transform=ax.transAxes, fontsize=8, color='cyan',
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.5))

    # Subsample to ~200 frames
    step = max(1, nZ // 200)
    frames = list(range(0, nZ, step))

    def update(i):
        im.set_data(
            10 * np.log10(np.maximum(pI_volume[:, :, i],
                                     peak * 1e-3) / peak).T)
        t_title.set_text(f'pI(x, y, z)   z = {z_cm[i]:6.2f} cm')
        return im, t_title

    print(f'  writing {fname} ({len(frames)} frames)')
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  interval=80, blit=False)
    writer = animation.FFMpegWriter(fps=15, bitrate=2400)
    ani.save(os.path.join(MOVIE_DIR, fname), writer=writer, dpi=110)
    plt.close(fig)


def build_movies(save_path):
    d = np.load(save_path)
    z_caps = d['z_caps']
    xaxis  = d['xaxis']
    pI     = d['pI']
    zaxis  = d['zaxis']
    z_focus = float(d['z_focus_m'])

    print(f'\nGenerating movies (z spans {z_caps[0]*100:.1f}–{z_caps[-1]*100:.1f} cm '
          f'in {len(z_caps)} snapshots)...')

    # Intensity pI (uses full pI volume, log-dB scale)
    make_pI_movie(pI, zaxis, xaxis, 'clean_shock_pI.mp4',
                  z_focus=z_focus)

    # Per-snapshot max_t reductions
    make_zsweep_movie(d['pk_v'], z_caps, xaxis,
                      'max$_t$|v(x,y,t,z)|',
                      'magma', 'clean_shock_v_peak.mp4',
                      symmetric=False, log_scale=False,
                      z_focus=z_focus)
    make_zsweep_movie(d['pk_a'], z_caps, xaxis,
                      'max$_t$|a(x,y,t,z)|',
                      'magma', 'clean_shock_a_peak.mp4',
                      symmetric=False, log_scale=False,
                      z_focus=z_focus)
    make_zsweep_movie(d['pk_u'], z_caps, xaxis,
                      'max$_t$|u(x,y,t,z)|',
                      'magma', 'clean_shock_u_peak.mp4',
                      symmetric=False, log_scale=False,
                      z_focus=z_focus)
    make_zsweep_movie(d['pk_eps'], z_caps, xaxis,
                      r'max$_t$|$\dot\varepsilon$(x,y,t,z)|',
                      'magma', 'clean_shock_eps_peak.mp4',
                      symmetric=False, log_scale=False,
                      z_focus=z_focus)
    make_zsweep_movie(d['pk_gam'], z_caps, xaxis,
                      r'max$_t$|$\gamma_{\rm vm}$(x,y,t,z)|',
                      'magma', 'clean_shock_gamma_peak.mp4',
                      symmetric=False, log_scale=False,
                      z_focus=z_focus)

    print('\nMovies saved to', MOVIE_DIR)


def main():
    save_path, xaxis = run()
    build_movies(save_path)


if __name__ == '__main__':
    main()
