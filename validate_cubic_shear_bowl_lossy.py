"""
3D shear-shock smoke test with attenuation — focused bowl in brain
shear regime, α(f) = α₀·f (linear-in-f, the canonical brain shear law,
Catheline 2003 / Pinton 2010), useAttenLoss=True for true per-pixel
attenuation tracking.

Same physical setup as validate_cubic_shear_bowl.py but with shear
absorption turned on. Adds an intensity-vs-loss focal-plane panel
analogous to the quadratic phase7_intensity_vs_loss_native_a0.5
figure but in the cubic regime.

Outputs:
  validation_results/cubic_shear_bowl_lossy/
    initial.png
    on_axis.png
    focal_xy.png
    xz_slice.png
    intensity_vs_loss.png   — focal-plane pI vs pIloss/dZ (lin + dB)
    summary.txt
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
                   'validation_results', 'cubic_shear_bowl_lossy')
os.makedirs(OUT, exist_ok=True)

# ---- physical regime (brain shear, lossy) ----
F0      = 100.0          # Hz
C0      = 2.0            # m/s
RHO0    = 1000.0
BETA3   = 5.0e8
LAM     = C0 / F0
ROC     = 5 * LAM
OUTER_R = 2.5 * LAM
FOCUS   = ROC
V0      = 0.15
NCYCLES = 4
DUR     = 4

# ---- numerics ----
DX        = LAM / 15
DT        = 1.0 / (60.0 * F0)
DOMAIN    = 6 * LAM
PROP_DIST = 1.05 * FOCUS

# ---- SHEAR loss (Catheline / Pinton 2010 convention: linear-in-f) ----
ALPHA0    = 0.5          # Np/m / Hz  (brain shear scale)
Y         = 1.0


def main():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 2.5 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    print(f'Grid: nX={nX}, nT={nT}, DX={DX*1e3:.2f} mm, DT={DT*1e3:.2f} ms')
    print(f'Bowl: a={OUTER_R*100:.1f} cm, ROC=focus={FOCUS*100:.1f} cm')
    print(f'λ = {LAM*100:.1f} cm, propDist = {PROP_DIST*100:.1f} cm')
    print(f'Source v₀ = {V0} m/s (M = {V0/C0:.3f})')
    print(f'Loss: α₀ = {ALPHA0} Np/m/Hz^{Y}, useAttenLoss = True')

    init = make_bowl_source(
        xaxis, xaxis, taxis, F0, C0, V0,
        radius=OUTER_R, roc=ROC, focus=FOCUS,
        ncycles=NCYCLES, dur=DUR,
    )
    print(f'Init: shape={init.shape}, peak={np.max(np.abs(init)):.4f}')

    # initial-conditions plot (same as lossless)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cy = nX // 2
    ext = [taxis[0]*1e3, taxis[-1]*1e3, xaxis[0]*100, xaxis[-1]*100]
    axes[0].imshow(init[:, cy, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r', vmin=-V0, vmax=V0)
    axes[0].set(xlabel='t (ms)', ylabel='x (cm)', title='Initial X-T slice')
    axes[1].imshow(init[cy, :, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r', vmin=-V0, vmax=V0)
    axes[1].set(xlabel='t (ms)', ylabel='y (cm)', title='Initial Y-T slice')
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
    print(f'N₃ = {N3:.3e}, focus gain ≈ {focus_gain:.2f}')
    print(f'dZmin = {dZmin*1e3:.2f} mm')

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
        useSuperAbsorbing=False,
        boundaryFactor=0.15,
        dZmin=dZmin,
        useGPUReductions=False,
        useAttenLoss=True,
        diagnostic=True,
        diagnosticInterval=50,
        diagnosticDir=diag_dir,
        diagnosticSummary=True,
        diagnosticInitialConditions=True,
    )

    print('Running 3D cubic-Burgers (lossy) solve...')
    t0 = time.time()
    field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
        init, sp, verbose=True)
    wall = time.time() - t0
    print(f'\nDone in {wall:.0f}s ({len(zaxis)} z-steps)')
    print(f'Volume peak: ppp = {ppp.max():.4f}, pnp = {pnp.min():.4f}')

    on_axis_pI = pI[nX//2, cy, :]
    z_focus_idx = int(np.argmax(on_axis_pI))
    z_focus = zaxis[z_focus_idx]
    print(f'z_focus = {z_focus*100:.2f} cm (geometric: {FOCUS*100:.0f} cm)')

    # ---- on-axis trace + spectrum ----
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
    omega0 = 2 * np.pi * F0
    v_peak = max(focal_trace.max(), -focal_trace.min())
    edge_steepness = np.max(np.abs(dvdt)) / max(v_peak * omega0, 1e-30)

    print(f'Focal: ppp={focal_trace.max():.3f}, pnp={focal_trace.min():.3f}')
    print(f'  edge steepness = {edge_steepness:.2f}, odd-high = {odd_frac:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(taxis*1e3, focal_trace, 'b-', lw=1.2)
    axes[0].set(xlabel='t (ms)', ylabel='v (m/s)',
                title=f'On-axis trace at focus (z={z_focus*100:.2f} cm)\n'
                      f'ppp={focal_trace.max():.3f}, pnp={focal_trace.min():.3f}, '
                      f'edge/lin={edge_steepness:.1f}, odd-frac={odd_frac:.2f}')
    axes[0].axhline(0, color='k', lw=0.4); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(freqs, spec / spec.max(), 'k-', lw=0.9)
    for n in range(1, 11):
        color = 'r' if n % 2 == 1 else 'b'
        axes[1].axvline(n * F0, color=color, ls=':', lw=0.4, alpha=0.6)
    axes[1].set(xlabel='Frequency (Hz)', ylabel='|V| (norm)',
                title=f'Spectrum (odd-high frac={odd_frac:.2f})',
                xlim=[0, 1500], ylim=[1e-5, 2])
    axes[1].grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, 'on_axis.png'), dpi=160); plt.close(fig)

    # ---- focal_xy lateral ----
    pI_lat = pI[:, cy, z_focus_idx]
    ppp_lat = ppp[:, cy, z_focus_idx]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xaxis*100, pI_lat / pI_lat.max(), 'b-', lw=1.6, label='pI')
    ax.plot(xaxis*100, ppp_lat / ppp_lat.max(), 'r--', lw=1.4, label='ppp')
    half = pI_lat.max() / 2
    above = np.where(pI_lat >= half)[0]
    fwhm_cm = ((xaxis[above[-1]] - xaxis[above[0]]) * 100
               if len(above) >= 2 else float('nan'))
    ax.set_title(f'Focal-plane lateral (z={z_focus*100:.2f} cm)\n'
                 f'pI FWHM = {fwhm_cm:.2f} cm = {fwhm_cm/(LAM*100):.2f} λ')
    ax.set(xlabel='x (cm)', ylabel='Normalised', xlim=[-10, 10])
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, 'focal_xy.png'), dpi=160); plt.close(fig)

    # ---- xz slice ----
    pI_xz = pI[:, cy, :]
    pI_xz_db = 10 * np.log10(np.maximum(pI_xz, pI_xz.max()*1e-4) / pI_xz.max())
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ext_xz = [zaxis[0]*100, zaxis[-1]*100, xaxis[0]*100, xaxis[-1]*100]
    im = ax.imshow(pI_xz_db, extent=ext_xz, aspect='auto',
                   origin='lower', cmap='inferno', vmin=-30, vmax=0)
    ax.set(xlabel='z (cm)', ylabel='x (cm)',
           title='pI(x, y=0, z) [dB rel. max]', ylim=[-7, 7])
    ax.axvline(z_focus*100, color='cyan', ls=':', lw=1,
               label=f'focus={z_focus*100:.1f} cm')
    ax.axvline(FOCUS*100, color='white', ls='--', lw=0.8, alpha=0.6,
               label=f'geom={FOCUS*100:.0f} cm')
    plt.colorbar(im, ax=ax, label='dB')
    ax.legend()
    fig.savefig(os.path.join(OUT, 'xz_slice.png'), dpi=160); plt.close(fig)

    # ---- INTENSITY vs LOSS-INTENSITY at focal plane ----
    if z_focus_idx > 0:
        dZ_local = float(zaxis[z_focus_idx] - zaxis[z_focus_idx - 1])
    else:
        dZ_local = float(zaxis[1] - zaxis[0])
    loss_rate_lat = pIloss[:, cy, z_focus_idx] / max(dZ_local, 1e-30)

    pI_n = pI_lat / (pI_lat.max() + 1e-30)
    loss_n = loss_rate_lat / (loss_rate_lat.max() + 1e-30)

    def fwhm_of(profile, axis_mm):
        h = profile.max() / 2
        ab = np.where(profile >= h)[0]
        return float((axis_mm[ab[-1]] - axis_mm[ab[0]])) if len(ab) >= 2 else float('nan')

    pI_fwhm  = fwhm_of(pI_lat,         xaxis*1e3)
    Q_fwhm   = fwhm_of(loss_rate_lat,  xaxis*1e3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(xaxis*1e3, pI_lat / pI_lat.max(), 'b-',  lw=1.6,
                 label=f'Intensity (FWHM={pI_fwhm:.2f} mm)')
    axes[0].plot(xaxis*1e3, loss_rate_lat / loss_rate_lat.max(), 'r--', lw=1.6,
                 label=f'Loss rate (FWHM={Q_fwhm:.2f} mm)')
    axes[0].set(xlabel='x (mm)', ylabel='Normalised', xlim=[-50, 50],
                title='Linear scale')
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc='upper right')

    axes[1].plot(xaxis*1e3, 10*np.log10(np.maximum(pI_n, 1e-6)), 'b-', lw=1.6,
                 label=f'Intensity (FWHM={pI_fwhm:.2f} mm)')
    axes[1].plot(xaxis*1e3, 10*np.log10(np.maximum(loss_n, 1e-6)), 'r--', lw=1.6,
                 label=f'Loss rate (FWHM={Q_fwhm:.2f} mm)')
    axes[1].set(xlabel='x (mm)', ylabel='dB rel. peak',
                xlim=[-50, 50], ylim=[-30, 1],
                title=f'dB; loss-narrowing ratio Q/pI = {Q_fwhm/pI_fwhm:.3f}')
    axes[1].grid(True, alpha=0.3); axes[1].legend(loc='upper right')

    fig.suptitle(f'Cubic shear-shock focal-plane: intensity vs loss-intensity\n'
                 f'α₀={ALPHA0} (linear-in-f shear), β₃={BETA3:g}, '
                 f'V₀={V0} m/s, useAttenLoss=True', fontsize=11)
    fig.savefig(os.path.join(OUT, 'intensity_vs_loss.png'), dpi=180)
    plt.close(fig)
    print(f'Loss-narrowing ratio Q/pI = {Q_fwhm/pI_fwhm:.3f}')

    # ---- summary ----
    summary = [
        '3D cubic-Burgers shear-shock smoke test (lossy)',
        '================================================',
        f'f₀ = {F0} Hz, c = {C0} m/s, ρ = {RHO0} kg/m³, β₃ = {BETA3}',
        f'Loss: α₀ = {ALPHA0} Np/m/Hz^{Y}, useAttenLoss = True',
        f'λ = {LAM*100:.1f} cm, bowl outer_r = {OUTER_R*100:.1f} cm '
        f'({2*OUTER_R/LAM:.1f} λ diameter)',
        f'Source V₀ = {V0} m/s, F# = {FOCUS/(2*OUTER_R):.2f}',
        f'Grid: {nX}×{nX}×{nT}, DX = {DX*1e3:.2f} mm, DT = {DT*1e3:.2f} ms',
        f'',
        f'Result:',
        f'  z-steps: {len(zaxis)}, wall {wall:.0f} s',
        f'  Volume peak ppp / pnp: {ppp.max():.4f} / {pnp.min():.4f}',
        f'  z_focus_observed: {z_focus*100:.2f} cm '
        f'(geometric: {FOCUS*100:.0f} cm)',
        f'  Focal pI FWHM: {fwhm_cm:.2f} cm = {fwhm_cm/(LAM*100):.2f} λ',
        f'  Focal loss-rate FWHM: {Q_fwhm:.2f} mm '
        f'({Q_fwhm/(LAM*1e3):.3f} λ)',
        f'  Loss-narrowing ratio Q/pI = {Q_fwhm/pI_fwhm:.3f} '
        f'(<1 = hyperfocal effect)',
        f'  Edge steepness {edge_steepness:.2f}, '
        f'odd-harmonic frac {odd_frac:.3f}',
    ]
    text = '\n'.join(summary) + '\n'
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
