"""
3D cubic shear-shock — HEMISPHERICAL focused bowl (F# = 0.5).

Same brain-shear regime as validate_cubic_shear_bowl_lossy.py but
ROC = focus = outer_r so the aperture is a full hemisphere — the most
strongly focused geometry that the angular-spectrum source can encode
as a flat-plane time-delay distribution.

  outer_r = ROC = focus = 5 cm = 2.5 λ
  F# = 0.5
  Source curvature delay = ROC/c = 5cm / 2 m/s = 25 ms

Outputs in validation_results/cubic_shear_hemi/:
  initial.png, on_axis.png, focal_xy.png, xz_slice.png,
  intensity_vs_loss.png, summary.txt
  + diagnostic_frames/ (per-step propagation snapshots)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source,
)

OUT = os.path.join(os.path.dirname(__file__),
                   'validation_results', 'cubic_shear_hemi')
os.makedirs(OUT, exist_ok=True)

# physical regime — same brain shear as the F#=1 run
F0      = 100.0
C0      = 2.0
RHO0    = 1000.0
BETA3   = 5.0e8
LAM     = C0 / F0          # 2 cm
OUTER_R = 2.5 * LAM        # 5 cm
ROC     = OUTER_R          # HEMISPHERE: ROC = outer_r
FOCUS   = ROC              # focus at the apex of the hemisphere
V0      = 0.15
NCYCLES = 4
DUR     = 4

# numerics
DX        = LAM / 15
DT        = 1.0 / (60.0 * F0)
DOMAIN    = 6 * LAM
PROP_DIST = 1.20 * FOCUS   # propagate 20% past focus

# shear loss (same as F#=1 lossy run)
ALPHA0    = 0.5
Y         = 1.0


def main():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    # Time window must contain pulse + curvature delay (ROC/c).
    curvature_delay = ROC / C0
    pulse_only = 2.5 * NCYCLES / F0
    pulse_dur = pulse_only + 1.5 * curvature_delay
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    print(f'Grid: {nX}×{nX}×{nT}, DX={DX*1e3:.2f} mm, DT={DT*1e3:.2f} ms')
    print(f'HEMI: outer_r = ROC = focus = {ROC*100:.1f} cm = {ROC/LAM:.1f} λ')
    print(f'F# = {FOCUS/(2*OUTER_R):.2f}, curvature delay = {curvature_delay*1e3:.1f} ms')
    print(f'pulse window = {pulse_dur*1e3:.1f} ms (= {pulse_only*1e3:.1f} ms pulse + '
          f'{1.5*curvature_delay*1e3:.1f} ms slack)')
    print(f'Source V₀ = {V0} m/s, M = {V0/C0:.3f}')

    init = make_bowl_source(
        xaxis, xaxis, taxis, F0, C0, V0,
        radius=OUTER_R, roc=ROC, focus=FOCUS,
        ncycles=NCYCLES, dur=DUR,
    )
    print(f'Init: shape={init.shape}, peak={np.max(np.abs(init)):.4f}')

    # initial conditions plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cy = nX // 2
    ext = [taxis[0]*1e3, taxis[-1]*1e3, xaxis[0]*100, xaxis[-1]*100]
    axes[0].imshow(init[:, cy, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r', vmin=-V0, vmax=V0)
    axes[0].set(xlabel='t (ms)', ylabel='x (cm)', title='Initial X-T (y=0)')
    axes[1].imshow(init[cy, :, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r', vmin=-V0, vmax=V0)
    axes[1].set(xlabel='t (ms)', ylabel='y (cm)', title='Initial Y-T (x=0)')
    axes[2].plot(taxis*1e3, init[cy, cy, :])
    axes[2].set(xlabel='t (ms)', ylabel='v (m/s)', title='On-axis source')
    axes[2].grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, 'initial.png'), dpi=160); plt.close(fig)

    # CFL
    N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * OUTER_R**2 / (2 * FOCUS)
    expected_peak = max(V0 * focus_gain, V0)
    safe_dZ = 0.15 * DT / max(expected_peak**2 * N3, 1e-30)
    dZmin = max(0.25 * safe_dZ, DT * C0 / 4.0)
    dZmin = min(dZmin, LAM / 10.0)
    print(f'N₃={N3:.3e}, focus_gain≈{focus_gain:.2f}, dZmin={dZmin*1e3:.2f} mm')

    diag_dir = os.path.join(OUT, 'diagnostic_frames')
    os.makedirs(diag_dir, exist_ok=True)
    sp = SolverParams(
        dX=DX, dY=DX, dT=DT, c0=C0, rho0=RHO0,
        beta=0.0, beta3=BETA3, nonlinearityOrder=3,
        alpha0=ALPHA0, attenPow=Y, f0=F0,
        propDist=PROP_DIST,
        useSplitStep=True, fluxScheme='kt', useTVD=True,
        useAdaptiveFiltering=True,
        boundaryProfile='quadratic',
        useFreqWeightedBoundary=False,
        useSuperAbsorbing=False, boundaryFactor=0.15,
        dZmin=dZmin,
        useGPUReductions=False,
        useAttenLoss=True,
        diagnostic=True,
        diagnosticInterval=50,
        diagnosticDir=diag_dir,
        diagnosticSummary=True,
        diagnosticInitialConditions=True,
    )

    print('Running 3D cubic-Burgers hemispherical solve...')
    t0 = time.time()
    field, pnp, ppp, pI, pIloss, zaxis, _ = angular_spectrum_solve(
        init, sp, verbose=True)
    wall = time.time() - t0
    print(f'\nDone in {wall:.0f}s ({len(zaxis)} z-steps)')
    print(f'Volume peak: ppp={ppp.max():.4f}, pnp={pnp.min():.4f}')

    on_axis_pI = pI[nX//2, cy, :]
    z_focus_idx = int(np.argmax(on_axis_pI))
    z_focus = float(zaxis[z_focus_idx])
    print(f'z_focus = {z_focus*100:.2f} cm (geometric: {FOCUS*100:.0f} cm)')

    focal_trace = field[nX//2, cy, :]
    spec = np.abs(np.fft.rfft(focal_trace))
    freqs = np.fft.rfftfreq(nT, DT)
    bin_w = 0.4 * F0
    e_n = np.zeros(11)
    for n in range(1, 11):
        m = (freqs >= n*F0 - bin_w) & (freqs <= n*F0 + bin_w)
        e_n[n] = np.sum(spec[m]**2)
    e_odd_high = e_n[3] + e_n[5] + e_n[7] + e_n[9]
    odd_frac = e_odd_high / max(e_n[1:11].sum(), 1e-30)
    dvdt = np.gradient(focal_trace, DT)
    v_peak = max(focal_trace.max(), -focal_trace.min())
    edge_steepness = np.max(np.abs(dvdt)) / max(v_peak * 2*np.pi*F0, 1e-30)
    print(f'Focal: ppp={focal_trace.max():.3f}, pnp={focal_trace.min():.3f}')
    print(f'  edge={edge_steepness:.2f}, odd-high={odd_frac:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(taxis*1e3, focal_trace, 'b-', lw=1.2)
    axes[0].set(xlabel='t (ms)', ylabel='v (m/s)',
                title=f'On-axis at focus (z={z_focus*100:.2f} cm)\n'
                      f'ppp={focal_trace.max():.3f}, pnp={focal_trace.min():.3f}, '
                      f'edge/lin={edge_steepness:.1f}, odd={odd_frac:.2f}')
    axes[0].axhline(0, color='k', lw=0.4); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(freqs, spec / spec.max(), 'k-', lw=0.9)
    for n in range(1, 11):
        color = 'r' if n % 2 == 1 else 'b'
        axes[1].axvline(n * F0, color=color, ls=':', lw=0.4, alpha=0.6)
    axes[1].set(xlabel='Frequency (Hz)', ylabel='|V| (norm)',
                title=f'Spectrum (odd-high={odd_frac:.2f})',
                xlim=[0, 1500], ylim=[1e-5, 2])
    axes[1].grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, 'on_axis.png'), dpi=160); plt.close(fig)

    # focal lateral
    pI_lat  = pI[:, cy, z_focus_idx]
    ppp_lat = ppp[:, cy, z_focus_idx]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xaxis*100, pI_lat / pI_lat.max(),  'b-',  lw=1.6, label='pI')
    ax.plot(xaxis*100, ppp_lat / ppp_lat.max(),'r--', lw=1.4, label='ppp')
    half = pI_lat.max() / 2
    above = np.where(pI_lat >= half)[0]
    fwhm_cm = ((xaxis[above[-1]] - xaxis[above[0]]) * 100
               if len(above) >= 2 else float('nan'))
    ax.set_title(f'Focal-plane lateral (z={z_focus*100:.2f} cm)\n'
                 f'pI FWHM = {fwhm_cm:.2f} cm = {fwhm_cm/(LAM*100):.2f} λ')
    ax.set(xlabel='x (cm)', ylabel='Normalised', xlim=[-10, 10])
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, 'focal_xy.png'), dpi=160); plt.close(fig)

    # xz dB slice
    pI_xz = pI[:, cy, :]
    pI_db = 10*np.log10(np.maximum(pI_xz, pI_xz.max()*1e-4) / pI_xz.max())
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ext_xz = [zaxis[0]*100, zaxis[-1]*100, xaxis[0]*100, xaxis[-1]*100]
    im = ax.imshow(pI_db, extent=ext_xz, aspect='auto', origin='lower',
                   cmap='inferno', vmin=-30, vmax=0)
    ax.set(xlabel='z (cm)', ylabel='x (cm)',
           title='pI(x, y=0, z) [dB rel. max]', ylim=[-7, 7])
    ax.axvline(z_focus*100, color='cyan', ls=':', lw=1,
               label=f'focus={z_focus*100:.1f} cm')
    ax.axvline(FOCUS*100, color='white', ls='--', lw=0.8, alpha=0.6,
               label=f'geom={FOCUS*100:.0f} cm')
    plt.colorbar(im, ax=ax, label='dB')
    ax.legend()
    fig.savefig(os.path.join(OUT, 'xz_slice.png'), dpi=160); plt.close(fig)

    # intensity vs loss-rate at focus
    dZ_at = float(zaxis[z_focus_idx] - zaxis[max(z_focus_idx-1, 0)]) \
            or float(zaxis[1] - zaxis[0])
    loss_lat = pIloss[:, cy, z_focus_idx] / max(dZ_at, 1e-30)
    pI_n = pI_lat / (pI_lat.max() + 1e-30)
    L_n  = loss_lat / (loss_lat.max() + 1e-30)
    def fwhm_of(p, ax_mm):
        h = p.max() / 2; ab = np.where(p >= h)[0]
        return float(ax_mm[ab[-1]] - ax_mm[ab[0]]) if len(ab) >= 2 else float('nan')
    pI_fwhm = fwhm_of(pI_lat,   xaxis*1e3)
    L_fwhm  = fwhm_of(loss_lat, xaxis*1e3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(xaxis*1e3, pI_n, 'b-', lw=1.6,  label=f'pI FWHM={pI_fwhm:.1f} mm')
    axes[0].plot(xaxis*1e3, L_n,  'r--', lw=1.6, label=f'loss FWHM={L_fwhm:.1f} mm')
    axes[0].set(xlabel='x (mm)', ylabel='Normalised', xlim=[-50, 50],
                title='Linear scale')
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].plot(xaxis*1e3, 10*np.log10(np.maximum(pI_n, 1e-6)), 'b-', lw=1.6)
    axes[1].plot(xaxis*1e3, 10*np.log10(np.maximum(L_n,  1e-6)), 'r--', lw=1.6)
    axes[1].set(xlabel='x (mm)', ylabel='dB rel. peak',
                xlim=[-50, 50], ylim=[-30, 1],
                title=f'dB; loss/pI ratio = {L_fwhm/pI_fwhm:.3f}')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f'Hemi cubic shear-shock focal plane: intensity vs loss\n'
                 f'α₀={ALPHA0}, β₃={BETA3:g}, V₀={V0} m/s, F#=0.5', fontsize=11)
    fig.savefig(os.path.join(OUT, 'intensity_vs_loss.png'), dpi=180)
    plt.close(fig)

    summary = [
        '3D cubic-Burgers HEMISPHERICAL shear-shock test (lossy)',
        '========================================================',
        f'f₀ = {F0} Hz, c = {C0} m/s, ρ = {RHO0} kg/m³, β₃ = {BETA3}',
        f'Loss: α₀ = {ALPHA0} Np/m/Hz^{Y}, useAttenLoss = True',
        f'Geometry: HEMISPHERE (outer_r = ROC = focus = {ROC*100:.1f} cm '
        f'= {ROC/LAM:.1f} λ)',
        f'F# = {FOCUS/(2*OUTER_R):.2f}, curvature delay = {ROC/C0*1e3:.1f} ms',
        f'Grid: {nX}×{nX}×{nT}, DX={DX*1e3:.2f} mm, DT={DT*1e3:.2f} ms',
        f'',
        f'Result:',
        f'  z-steps: {len(zaxis)}, wall {wall:.0f}s',
        f'  Volume peak ppp/pnp: {ppp.max():.4f} / {pnp.min():.4f}',
        f'  z_focus_observed: {z_focus*100:.2f} cm '
        f'(geometric: {FOCUS*100:.0f} cm)',
        f'  Focal pI FWHM: {fwhm_cm:.2f} cm = {fwhm_cm/(LAM*100):.3f} λ',
        f'  Focal loss FWHM: {L_fwhm:.2f} mm = {L_fwhm/(LAM*1e3):.3f} λ',
        f'  Loss/pI ratio: {L_fwhm/pI_fwhm:.3f}',
        f'  Edge steepness: {edge_steepness:.2f}, odd-frac: {odd_frac:.3f}',
    ]
    text = '\n'.join(summary) + '\n'
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
