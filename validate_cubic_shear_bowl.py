"""
3D shear-shock smoke test — focused bowl in brain-tissue regime.

Brain shear parameters (Pinton 2010, Giammarinaro 2016):
  c_T = 2 m/s, ρ = 1000 kg/m³, β₃ ≈ 100
  f₀ = 100 Hz, λ_T = 2 cm
  Bowl outer radius = 2.5λ = 5 cm, focus = ROC = 5λ = 10 cm (F# ≈ 1)
  Source particle velocity v₀ = 0.05 m/s (M = 0.025 at source)

Expected: cumulative cubic cascade through the converging bowl
produces a sub-wavelength shock at the focus, with asymmetric
ppp/pnp ratio characteristic of cubic shocks.

Outputs:
  validation_results/cubic_shear_bowl/
    initial.png       — initial-condition lateral and temporal slices
    on_axis.png       — on-axis pressure trace at focus + spectrum
    focal_xy.png      — focal-plane lateral profile (intensity & shape)
    xz_slice.png      — XZ propagation slice in dB
    summary.txt       — peak amplitudes, FWHM, asymmetry
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
                   'validation_results', 'cubic_shear_bowl')
os.makedirs(OUT, exist_ok=True)

# ---- physical regime ----
F0      = 100.0          # Hz
C0      = 2.0            # m/s (brain shear)
RHO0    = 1000.0         # kg/m³
# β₃ in the solver convention N₃ = β₃/(3·c₀⁵·ρ₀²): the canonical shear
# β₃ ≈ 100 from Pinton 2010 / Catheline 2003 is given in nondimensional
# (rescaled-field) form, which leaves the c⁵·ρ² unit factor implicit.
# For the pressure/velocity-amplitude convention used by this solver,
# the equivalent β₃ to give brain-shear shock formation at z ≈ λ/2 with
# M ≈ 0.1 is ∼1e8. We keep the physical-unit interpretation explicit
# here.
BETA3   = 5.0e8
LAM     = C0 / F0        # 2 cm
ROC     = 5 * LAM        # 10 cm
OUTER_R = 2.5 * LAM      # 5 cm (= 5λ aperture diameter)
FOCUS   = ROC
V0      = 0.15           # source particle velocity (m/s, brain-trauma scale)
NCYCLES = 4
DUR     = 4

# ---- numerics ----
DX        = LAM / 15           # 1.33 mm
DT        = 1.0 / (60.0 * F0)  # 167 µs
DOMAIN    = 6 * LAM            # ±6λ lateral
PROP_DIST = 1.05 * FOCUS       # propagate slightly past focus
ALPHA0    = -1.0               # water-like default; shear absorption is f^y but
                               # this smoke test isolates the cubic mechanism;
                               # set α₀ > 0 once shock formation is confirmed
Y         = 2.0


def main():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 2.5 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    print(f'Grid: nX={nX}, nT={nT}, DX={DX*1e3:.2f} mm, DT={DT*1e3:.2f} ms')
    print(f'Bowl: a={OUTER_R*100:.1f} cm, ROC=focus={FOCUS*100:.1f} cm '
          f'(F# = {FOCUS/(2*OUTER_R):.2f})')
    print(f'λ = {LAM*100:.1f} cm, propDist = {PROP_DIST*100:.1f} cm '
          f'({PROP_DIST/LAM:.1f} λ)')
    print(f'Source v₀ = {V0} m/s (M = {V0/C0:.3f})')

    init = make_bowl_source(
        xaxis, xaxis, taxis, F0, C0, V0,
        radius=OUTER_R, roc=ROC, focus=FOCUS,
        ncycles=NCYCLES, dur=DUR,
    )
    print(f'Init field: shape={init.shape}, peak={np.max(np.abs(init)):.4f}')

    # --- initial conditions plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    cy = nX // 2
    ext = [taxis[0]*1e3, taxis[-1]*1e3, xaxis[0]*100, xaxis[-1]*100]
    axes[0].imshow(init[:, cy, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r',
                   vmin=-V0, vmax=V0)
    axes[0].set(xlabel='t (ms)', ylabel='x (cm)', title='Initial X-T slice')
    axes[1].imshow(init[cy, :, :], aspect='auto', extent=ext,
                   origin='lower', cmap='RdBu_r',
                   vmin=-V0, vmax=V0)
    axes[1].set(xlabel='t (ms)', ylabel='y (cm)', title='Initial Y-T slice')
    axes[2].plot(taxis*1e3, init[cy, cy, :])
    axes[2].set(xlabel='t (ms)', ylabel='v (m/s)',
                title='On-axis source trace')
    axes[2].grid(True, alpha=0.3)
    out = os.path.join(OUT, 'initial.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f'wrote {out}')

    # --- conservative dZmin for cubic CFL: λ²(u) = u² so margin = N₃·dZ·u²/dT
    # Estimate expected peak via paraxial focus gain; cubic cascade can lift
    # this further so we use 0.25× safety margin.
    N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * OUTER_R**2 / (2 * FOCUS)
    expected_peak = max(V0 * focus_gain, V0)
    safe_dZ = 0.15 * DT / max(expected_peak**2 * N3, 1e-30)
    # Cap dZmin at λ/10 so we always get adequate spatial sampling along
    # the propagation axis even when the cubic nonlinearity is weak.
    dZmin = max(0.25 * safe_dZ, DT * C0 / 4.0)
    dZmin = min(dZmin, LAM / 10.0)
    n_steps_est = int(np.ceil(PROP_DIST / dZmin))
    print(f'Cubic N₃ = {N3:.3e} (β₃/(3c⁵ρ²))')
    print(f'Paraxial focus gain ≈ {focus_gain:.2f} → expected peak '
          f'{expected_peak:.3f}')
    print(f'dZmin = {dZmin*1e3:.2f} mm, ~{n_steps_est} steps')

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
    )

    print('Running 3D cubic-Burgers solve...')
    t0 = time.time()
    field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
        init, sp, verbose=True)
    wall = time.time() - t0
    print(f'\nDone in {wall:.0f}s ({len(zaxis)} z-steps)')
    print(f'Volume peak: ppp = {ppp.max():.4f}, pnp = {pnp.min():.4f}')
    print(f'Asymmetry ratio = {ppp.max()/abs(pnp.min()):.2f}')

    # --- on-axis trace + spectrum at focus ---
    on_axis_pI = pI[nX//2, cy, :]
    z_focus_idx = int(np.argmax(on_axis_pI))
    z_focus = zaxis[z_focus_idx]
    print(f'Focal-plane index: {z_focus_idx}, z_focus = {z_focus*100:.2f} cm')

    focal_trace = field[nX//2, cy, :]

    # --- cubic-shock signatures ---
    # Cubic Burgers preserves u → -u symmetry, so asymmetry is ≈1 by
    # construction. Real cubic-shock diagnostics:
    #   1. Odd-harmonic energy fraction in the spectrum binned at n·f₀.
    #   2. Edge steepness (max |dv/dt|) normalised by the linear slope
    #      V_peak·ω₀ that an unshocked sinusoid of the same peak would give.
    spec = np.abs(np.fft.rfft(focal_trace))
    freqs = np.fft.rfftfreq(nT, DT)
    # Energy at each harmonic n=1..10 (sum over a ±0.4·f₀ bin).
    bin_w = 0.4 * F0
    e_n = np.zeros(11)
    for n in range(1, 11):
        m = (freqs >= n*F0 - bin_w) & (freqs <= n*F0 + bin_w)
        e_n[n] = np.sum(spec[m]**2)
    e_odd_high = e_n[3] + e_n[5] + e_n[7] + e_n[9]   # cubic-cascade signal
    e_total_h = e_n[1:11].sum()
    odd_frac = e_odd_high / max(e_total_h, 1e-30)

    dvdt = np.gradient(focal_trace, DT)
    omega0 = 2 * np.pi * F0
    v_peak = max(focal_trace.max(), -focal_trace.min())
    linear_slope = v_peak * omega0
    edge_steepness = np.max(np.abs(dvdt)) / max(linear_slope, 1e-30)

    cubic_sig = {
        'ppp': float(focal_trace.max()),
        'pnp': float(focal_trace.min()),
        'asym': float(focal_trace.max() / abs(focal_trace.min())),
        'odd_high_frac': float(odd_frac),
        'edge_steepness': float(edge_steepness),
    }
    print('Cubic-shock signatures at focus:')
    for k, v in cubic_sig.items():
        print(f'  {k:18s} = {v:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(taxis*1e3, focal_trace, 'b-', lw=1.2)
    axes[0].set(xlabel='t (ms)', ylabel='v (m/s)',
                title=f'On-axis trace at focus (z={z_focus*100:.2f} cm)\n'
                      f'ppp={cubic_sig["ppp"]:.3f}, pnp={cubic_sig["pnp"]:.3f}, '
                      f'odd-high frac={odd_frac:.2f}, '
                      f'edge/linear={edge_steepness:.1f}')
    axes[0].axhline(0, color='k', lw=0.4); axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(freqs, spec / spec.max(), 'k-', lw=0.9)
    for n in range(1, 11):
        color = 'r' if (n % 2 == 1) else 'b'
        axes[1].axvline(n * F0, color=color, ls=':', lw=0.4, alpha=0.6)
    axes[1].set(xlabel='Frequency (Hz)', ylabel='|V| (norm)',
                title=f'On-axis spectrum at focus '
                      f'(odd-harm fraction={odd_frac:.2f})',
                xlim=[0, 1500], ylim=[1e-5, 2])
    axes[1].grid(True, alpha=0.3)
    out = os.path.join(OUT, 'on_axis.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f'wrote {out}')

    # --- focal-plane lateral profile ---
    pI_lat = pI[:, cy, z_focus_idx]
    ppp_lat = ppp[:, cy, z_focus_idx]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(xaxis*100, pI_lat / pI_lat.max(), 'b-', lw=1.6,
            label='pI (intensity)')
    ax.plot(xaxis*100, ppp_lat / ppp_lat.max(), 'r--', lw=1.4,
            label='ppp (peak +v)')
    # FWHM
    half = pI_lat.max() / 2
    above = np.where(pI_lat >= half)[0]
    if len(above) >= 2:
        fwhm = (xaxis[above[-1]] - xaxis[above[0]]) * 100
        ax.set_title(f'Focal-plane lateral profile (z={z_focus*100:.2f} cm)\n'
                     f'pI FWHM = {fwhm:.2f} cm = {fwhm/(LAM*100):.2f} λ')
    ax.set(xlabel='x (cm)', ylabel='Normalised',
           xlim=[-10, 10])
    ax.legend(); ax.grid(True, alpha=0.3)
    out = os.path.join(OUT, 'focal_xy.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f'wrote {out}')

    # --- XZ propagation slice (intensity in dB) ---
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
               label=f'geometric focus={FOCUS*100:.0f} cm')
    plt.colorbar(im, ax=ax, label='dB')
    ax.legend()
    out = os.path.join(OUT, 'xz_slice.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f'wrote {out}')

    # --- summary text ---
    summary_lines = [
        '3D cubic-Burgers shear-shock smoke test',
        '========================================',
        f'f₀ = {F0} Hz, c = {C0} m/s, ρ = {RHO0} kg/m³, β₃ = {BETA3}',
        f'λ = {LAM*100:.1f} cm',
        f'Bowl: outer_r = {OUTER_R*100:.1f} cm = {OUTER_R/LAM:.1f} λ '
        f'(diameter = {2*OUTER_R/LAM:.1f} λ)',
        f'      focus = ROC = {FOCUS*100:.1f} cm = {FOCUS/LAM:.1f} λ '
        f'(F# = {FOCUS/(2*OUTER_R):.2f})',
        f'Source v₀ = {V0} m/s (M = {V0/C0:.3f})',
        f'Grid: {nX}×{nX}×{nT}, DX = {DX*1e3:.2f} mm, DT = {DT*1e3:.2f} ms',
        f'Solver: nonlinearityOrder = 3, fluxScheme = kt, '
        f'useAttenLoss off, α₀ = {ALPHA0} (lossless smoke test)',
        f'',
        f'Result:',
        f'  z-steps: {len(zaxis)}',
        f'  Wall time: {wall:.0f} s',
        f'  ppp peak (volume): {ppp.max():.4f} m/s',
        f'  pnp min (volume): {pnp.min():.4f} m/s',
        f'  z_focus_observed: {z_focus*100:.2f} cm '
        f'(geometric: {FOCUS*100:.0f} cm)',
        f'  Focal pI FWHM: {fwhm:.2f} cm = {fwhm/(LAM*100):.2f} λ',
        f'',
        f'Cubic-shock signatures at focus:',
        f'  Asymmetry ppp/|pnp| (cubic expects ≈1): '
        f'{cubic_sig["asym"]:.3f}',
        f'  Edge steepness (max|dv/dt|/V·ω₀, '
        f'sine=1, sawtooth≫1): {cubic_sig["edge_steepness"]:.2f}',
        f'  Odd-harmonic fraction (n=3,5,7,9 / total): '
        f'{cubic_sig["odd_high_frac"]:.3f}',
    ]
    text = '\n'.join(summary_lines) + '\n'
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
