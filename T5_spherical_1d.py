"""
T5 — 1-D spherical-coordinate cubic-Burgers solver for cascade
validation under focused convergence.

Solves the spherical-Burgers equation for a converging shear wave
propagating from R = R_bowl (source distance from focus) inward
toward R = R_min (diffraction-limited stopping point):

  ∂U/∂s + (N₃ / R²) · U² · ∂U/∂τ + α(f) · U  =  0,

  s    ≡ R_bowl − R   (propagation variable, increases inward)
  U    ≡ R · v        (scaled velocity; finite at R → 0)
  N₃   = β₃ / (3 c⁵ ρ²)
  α(f) = α₀ · f^y   (linear-in-f shear loss for our brain regime, y=1)

The 1/R² coefficient is the spherical-convergence enhancement of the
nonlinearity: as the wave converges, its local amplitude v = U/R grows
and the nonlinear term in v² grows even faster.  This is the
geometric-amplification physics that plane-wave 1-D (T4 / convergence_1d.py)
cannot capture.

Numerical scheme: split-step
  (1) KT cubic-Burgers RK2 step on U with coefficient N₃ · (R_mid)⁻² over Δs
  (2) frequency-domain attenuation step on U over Δs

After marching from R = R_bowl down to R = R_min, the focal-axis
velocity at the diffraction-limited plane is v_focal(τ) = U(R_min, τ)/R_min.
Extract the harmonic spectrum |V_n|² from v_focal and compare to:
  (a) the 3-D simulation's measured focal-axis spectrum from T1, and
  (b) the central-lobe-rule prediction of the ring radii.

Outputs:
  validation_results/theory_validation/T5_spherical_1d/
    spectra_comparison.txt
    spectra_comparison.png
    ring_prediction_table.txt
    sweep_summary.json
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'cm'})

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import _kt_flux_cubic
from theory_predictor import on_axis_harmonic_amplitudes, predict_ratios
from T1_validate_gaussian import (
    F0, C0, RHO0, BETA3, ALPHA0, Y_ABS,
    NCYCLES, LAM, WAIST_SRC, FOCUS, V0_LIST,
)

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T5_spherical_1d',
)
os.makedirs(OUT_ROOT, exist_ok=True)

T1_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T1_gaussian',
)


def linear_focal_waist():
    """Paraxial Gaussian focal waist and Rayleigh range."""
    w0 = LAM * FOCUS / (np.pi * WAIST_SRC)
    z_R = np.pi * w0 * w0 / LAM
    return w0, z_R


def build_pulse(taxis: np.ndarray, V0: float) -> np.ndarray:
    """Match the make_gaussian_source pulse envelope (super-Gaussian)."""
    omega0 = 2.0 * np.pi * F0
    pulse_dur = NCYCLES / F0
    # super-Gaussian envelope of order 2·dur=4 (dur=2 matches T1)
    envelope = np.exp(-(1.05 * taxis * omega0 / (NCYCLES * np.pi))**4)
    return (V0 * envelope * np.sin(omega0 * taxis)).astype(np.float32)


def march_spherical(V0: float,
                    samples_per_cycle: int = 120,
                    dR_factor: float = 0.05):
    """Propagate 1-D spherical Burgers from R=R_bowl down to R=z_R.

    Returns (v_focal_axis, taxis_s, R_history, R_min, diagnostics).
    """
    w0, z_R = linear_focal_waist()
    R_bowl = FOCUS                     # source distance from focal point
    R_min  = z_R                       # diffraction-limited stop
    s_max  = R_bowl - R_min

    DT = 1.0 / (samples_per_cycle * F0)
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT

    # Initial v at R = R_bowl
    v0_field = build_pulse(taxis, V0)
    U = R_bowl * v0_field                              # scaled variable
    U = jnp.asarray(U.reshape(1, 1, -1), dtype=jnp.float32)

    N_burgers = BETA3 / (3 * C0**5 * RHO0**2)
    # Choose dR: scale by R² so nonlinearity step is bounded
    # CFL-like estimate: dR · N₃ · v² < α · DT
    # At R near focus, v_focal ≈ V0 · R_bowl/R_min
    V_focal_est = max(V0 * R_bowl / R_min, V0)
    safe_dR_at_focus = 0.15 * DT / max(V_focal_est**2 * N_burgers, 1e-30)
    dR_base = max(dR_factor * safe_dR_at_focus, R_bowl * 1e-4)
    dR_base = min(dR_base, s_max * 0.001)  # at least 1000 steps

    # Attenuation operator (computed once).
    # MUST match the AS solver convention: input alpha0 is in
    # dB/MHz^pw/cm; convert to Np/m as
    #   conv = alpha0 / 1e6^pw * 1e2 / (20·log10(e))
    # Despite the cubic-shear lossy script's docstring claiming
    # "Np/m/Hz", the actual application is dB/MHz/cm — so the wave
    # sees ~5.76e-4 Np/m at F_0=100 Hz, essentially loss-free over F.
    freqs = np.fft.fftfreq(nT, DT)
    conv = ALPHA0 / (1e6 ** Y_ABS) * 1e2 / (20 * np.log10(np.e))
    alpha_f = conv * np.abs(freqs)**Y_ABS

    R = R_bowl
    R_hist = [R]
    v_axis_history = []     # for diagnostics
    n_steps = 0
    t0 = time.time()
    while R > R_min + 1e-12:
        dR = min(dR_base, R - R_min)
        R_mid = R - 0.5 * dR
        # (1) KT cubic Burgers step on U with coefficient N₃/R²_mid
        N_eff = N_burgers / (R_mid * R_mid)
        U = _kt_flux_cubic(U, N_eff, dR, DT)
        # (2) Frequency-domain attenuation
        U_np = np.asarray(U[0, 0, :])
        Uf = np.fft.fft(U_np)
        Uf *= np.exp(-alpha_f * dR)
        U_np = np.real(np.fft.ifft(Uf)).astype(np.float32)
        U = jnp.asarray(U_np.reshape(1, 1, -1))
        R -= dR
        n_steps += 1
        R_hist.append(R)
        if n_steps % 50 == 0:
            v_now = U_np / R
            v_axis_history.append((R, v_now.copy()))
    wall = time.time() - t0
    v_focal = np.asarray(U[0, 0, :]) / R_min

    return dict(
        v_focal=v_focal,
        taxis=taxis,
        R_hist=np.array(R_hist),
        R_bowl=R_bowl,
        R_min=R_min,
        w0=w0,
        z_R=z_R,
        DT=DT,
        n_steps=n_steps,
        wall_s=wall,
        V_focal_peak=float(np.max(np.abs(v_focal))),
    )


def main():
    print('T5 — 1D-spherical cubic-Burgers solver')
    w0, z_R = linear_focal_waist()
    print(f'Gaussian focal waist w_0 = {w0*1e3:.3f} mm')
    print(f'Rayleigh range z_R = {z_R*100:.3f} cm')
    print(f'R_bowl = F = {FOCUS*100:.1f} cm, R_min = z_R = '
          f'{z_R*100:.3f} cm (focal_gain = {FOCUS/z_R:.3f})')
    print()

    rows = []
    for v0 in V0_LIST:
        print(f'=== V₀ = {v0} m/s ===')
        out = march_spherical(v0)
        print(f'  n_steps={out["n_steps"]}, wall={out["wall_s"]:.2f}s')
        print(f'  V_focal peak (1D-spherical) = {out["V_focal_peak"]:.4e} m/s')

        # Extract focal-axis spectrum
        Vn2_1d = on_axis_harmonic_amplitudes(out['v_focal'], out['DT'], F0,
                                              n_max=15)
        Vn2_1d_n = Vn2_1d / max(Vn2_1d.sum(), 1e-30)
        # Theory predictions using 1D-derived spectrum
        n_idx = np.arange(1, 16, dtype=float)
        pred = predict_ratios(Vn2_1d, n_idx, w0=w0)
        print(f'  ⟨n⟩_v   (1D) = {pred["n_v"]:.4f}')
        print(f'  ⟨n⟩_a²  (1D) = {pred["n_a2"]:.4f}')
        print(f'  wa/wv pred (1D-based) = {pred["wa_over_wv"]:.4f}')
        print(f'  r_ring_ε̇ pred (1D-based) = '
              f'{pred["r_ring_eps"]*1e3:.3f} mm')

        # Compare to T1 3-D results
        t1m_path = os.path.join(T1_ROOT, f'v0_{v0:.3f}', 'metrics.json')
        if os.path.exists(t1m_path):
            with open(t1m_path) as f:
                t1m = json.load(f)
            v_focal_3d = t1m['nonlinear']['on_axis_v_peak']
            Vn2_3d = np.array(t1m['nonlinear']['Vn2'])
            Vn2_3d_n = Vn2_3d / max(Vn2_3d.sum(), 1e-30)
            n_v_3d  = t1m['predicted']['n_v']
            n_a2_3d = t1m['predicted']['n_a2']
            wa_3d_meas = t1m['measured']['wa_over_wv']
            print(f'  ---- 3D comparison ----')
            print(f'  V_focal peak (3D) = {v_focal_3d:.4e} m/s')
            print(f'  ⟨n⟩_v   (3D) = {n_v_3d:.4f}')
            print(f'  ⟨n⟩_a²  (3D) = {n_a2_3d:.4f}')
            print(f'  wa/wv meas (3D) = {wa_3d_meas:.4f}')
            row = dict(
                v0=v0,
                V_focal_1d=out['V_focal_peak'],
                V_focal_3d=v_focal_3d,
                V_focal_ratio=out['V_focal_peak']/max(v_focal_3d, 1e-30),
                n_v_1d=pred['n_v'],
                n_a2_1d=pred['n_a2'],
                n_v_3d=n_v_3d,
                n_a2_3d=n_a2_3d,
                wa_over_wv_pred_1d=pred['wa_over_wv'],
                wa_over_wv_meas_3d=wa_3d_meas,
                r_ring_eps_pred_1d_mm=pred['r_ring_eps']*1e3,
                r_ring_eps_meas_3d_mm=t1m['measured']['r_ring_eps_mm'],
                Vn2_1d=Vn2_1d_n.tolist(),
                Vn2_3d=Vn2_3d_n.tolist(),
            )
            rows.append(row)
        print()

    # Tabulate
    lines = [f'T5 — 1D-spherical cubic-Burgers cascade comparison',
             '=' * 100,
             f'Geometric: R_bowl = F = {FOCUS*100:.1f} cm, '
             f'R_min = z_R = {z_R*100:.2f} cm, '
             f'focal_gain (geometric) = {FOCUS/z_R:.3f}',
             '',
             f'{"V₀":>6} {"V_focal_1D":>12} {"V_focal_3D":>12} {"ratio":>8} '
             f'{"⟨n⟩_v 1D":>10} {"3D":>8} {"⟨n⟩_a² 1D":>11} {"3D":>8} '
             f'{"wa/wv pred":>11} {"meas":>8}',
             '-' * 100]
    for r in rows:
        lines.append(
            f'{r["v0"]:>6.3f} {r["V_focal_1d"]:>12.4e} '
            f'{r["V_focal_3d"]:>12.4e} {r["V_focal_ratio"]:>8.3f} '
            f'{r["n_v_1d"]:>10.4f} {r["n_v_3d"]:>8.4f} '
            f'{r["n_a2_1d"]:>11.4f} {r["n_a2_3d"]:>8.4f} '
            f'{r["wa_over_wv_pred_1d"]:>11.4f} '
            f'{r["wa_over_wv_meas_3d"]:>8.4f}'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'spectra_comparison.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    # Spectra-comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    n_ax = np.arange(1, 16)
    for ax, r in zip(axes.flat, rows):
        ax.semilogy(n_ax, np.array(r['Vn2_1d']), 'rs--', lw=1.4,
                    markersize=6, markerfacecolor='none',
                    label='1-D spherical')
        ax.semilogy(n_ax, np.array(r['Vn2_3d']), 'bo-', lw=1.4,
                    markersize=6, label='3-D simulation')
        ax.set(xlabel='Harmonic n',
               ylabel=r'$|V_n|^2$ (normalised)',
               title=f'V_0 = {r["v0"]} m/s',
               xlim=[0.5, 12], ylim=[1e-7, 2])
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    for ax in axes.flat[len(rows):]:
        ax.axis('off')
    fig.suptitle('T5 — on-axis focal harmonic spectrum:  1-D spherical '
                 'cubic-Burgers vs 3-D angular-spectrum',
                 fontsize=11)
    fig.savefig(os.path.join(OUT_ROOT, 'spectra_comparison.png'), dpi=170)
    plt.close(fig)

    # Predicted (from 1D) vs measured (in 3D) ring-radius table
    ring_lines = [f'T5 — central-lobe-rule prediction using 1D-spherical '
                  f'spectrum, vs 3D simulation ring radius',
                  '=' * 110,
                  f'{"V₀":>6} {"⟨n⟩_a² (1D)":>13} {"r_eps pred":>13} '
                  f'{"r_eps meas (3D)":>17} {"err":>9} '
                  f'{"wa/wv pred (1D)":>17} {"wa/wv meas (3D)":>17} '
                  f'{"err":>9}',
                  '-' * 110]
    for r in rows:
        e_ring = (r['r_ring_eps_meas_3d_mm'] - r['r_ring_eps_pred_1d_mm']) \
                 / max(r['r_ring_eps_pred_1d_mm'], 1e-30) * 100
        e_wa   = (r['wa_over_wv_meas_3d'] - r['wa_over_wv_pred_1d']) \
                 / max(r['wa_over_wv_pred_1d'], 1e-30) * 100
        ring_lines.append(
            f'{r["v0"]:>6.3f} {r["n_a2_1d"]:>13.4f} '
            f'{r["r_ring_eps_pred_1d_mm"]:>13.2f} '
            f'{r["r_ring_eps_meas_3d_mm"]:>17.2f} '
            f'{e_ring:>+8.1f}% '
            f'{r["wa_over_wv_pred_1d"]:>17.4f} '
            f'{r["wa_over_wv_meas_3d"]:>17.4f} '
            f'{e_wa:>+8.1f}%'
        )
    ring_text = '\n'.join(ring_lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'ring_prediction_table.txt'), 'w') as f:
        f.write(ring_text)
    print('\n' + ring_text)

    with open(os.path.join(OUT_ROOT, 'sweep_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)


if __name__ == '__main__':
    main()
