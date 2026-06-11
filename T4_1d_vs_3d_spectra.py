"""
T4 — 1D vs 3D on-axis harmonic-spectrum cross-validation.

Goal: confirm that the 1-D cubic-Burgers solver and the 3-D angular-
spectrum solver produce the same per-harmonic content at matched
operating conditions, isolating cascade physics from focal-pattern
physics.

Setup
-----
For each V₀ in V0_LIST (matching T1):
  3D run: take T1's nonlinear Gaussian focal-plane on-axis trace
          at z = F.  Extract on-axis |V_n|² for n=1..15.
  1D run: prescribe the SAME on-axis pulse shape and amplitude at
          z=0 (i.e. the source pulse profile that the 3-D wave saw on
          its way to focus), and propagate via 1-D KT cubic + linear-
          in-f loss out to an equivalent path length.

The "equivalent path length" choice is the focal length F (the
distance over which the 3-D wave accumulated nonlinearity from
near-source to focus).  In linear focused-Gaussian geometry the wave
spends most of its lifetime near focus, so a 1D match using just the
focal-region peak amplitude as the constant amplitude over a Rayleigh-
range path is a reasonable surrogate.

A simpler and more direct check (used here): match the 3D ON-AXIS
focal velocity peak as the 1-D source amplitude, propagate 1D over
half the Rayleigh range z_R = π·w_focal²/λ where w_focal is the linear
Gaussian focal waist of the 3-D source.  Compare the resulting 1-D
spectrum with the 3-D focal-plane spectrum.

Outputs: validation_results/theory_validation/T4_1d_3d/
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from convergence_1d import (
    march_1d as march_1d_SHK,           # 1-D solver (LIN/MID/SHK style)
    F0 as F0_1D, C0 as C0_1D,
    BETA3 as BETA3_1D, ALPHA0 as ALPHA0_1D,
    Y as Y_1D, NCYCLES as NCYCLES_1D,
)
from T1_validate_gaussian import (
    F0, C0, LAM, WAIST_SRC, FOCUS, V0_LIST,
)

T1_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T1_gaussian',
)
OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T4_1d_3d',
)
os.makedirs(OUT_ROOT, exist_ok=True)


def main():
    print('T4 — 1D vs 3D on-axis harmonic-spectrum cross-validation')
    # Linear Gaussian focal waist (analytic paraxial)
    w_focal = LAM * FOCUS / (np.pi * WAIST_SRC)
    z_R = np.pi * w_focal**2 / LAM
    print(f'  Linear focal Gaussian waist w_focal = {w_focal*1e3:.2f} mm')
    print(f'  Rayleigh range z_R = π w_focal²/λ = {z_R*100:.2f} cm')

    rows = []
    for v0 in V0_LIST:
        t1_metrics_path = os.path.join(T1_ROOT, f'v0_{v0:.3f}',
                                       'metrics.json')
        if not os.path.exists(t1_metrics_path):
            print(f'  [skip] V₀={v0}: T1 metrics not found')
            continue
        with open(t1_metrics_path) as f:
            t1m = json.load(f)
        # 3D focal on-axis peak velocity
        v_focal_3d = t1m['nonlinear']['on_axis_v_peak']
        Vn2_3d = np.array(t1m['nonlinear']['Vn2'])
        # Normalise spectrum
        Vn2_3d_norm = Vn2_3d / max(Vn2_3d.sum(), 1e-30)

        # 1-D match: drive at v_focal_3d, propagate over z_R/2 (median
        # nonlinearity length inside the focal region).
        target_z = z_R * 0.5
        print(f'\n=== V₀={v0:.3f} m/s | 3D focal v_peak = {v_focal_3d:.4e} ===')
        print(f'    1-D target: amp={v_focal_3d:.4e}, z={target_z*100:.3f} cm')
        # Use a fixed (production) grid for the 1-D solver
        m_1d, traces_1d = march_1d_SHK(samples_per_cycle=120, dZ_factor=0.125,
                                       target_z=float(target_z),
                                       v0=float(v_focal_3d))
        # Extract |V_n|² from 1-D trace
        v_1d = traces_1d['v']
        DT_1d = 1.0 / (120 * F0_1D)
        nT_1d = len(v_1d)
        spec_1d = np.abs(np.fft.rfft(v_1d))**2
        freqs_1d = np.fft.rfftfreq(nT_1d, DT_1d)
        bin_w = 0.4 * F0_1D
        Vn2_1d = np.zeros(15)
        for k, n in enumerate(range(1, 16)):
            m = (freqs_1d >= n*F0_1D - bin_w) & (freqs_1d <= n*F0_1D + bin_w)
            Vn2_1d[k] = float(np.sum(spec_1d[m]))
        Vn2_1d_norm = Vn2_1d / max(Vn2_1d.sum(), 1e-30)

        print(f'    1-D edge-steepness = '
              f'{m_1d["a_peak"]/(2*np.pi*F0_1D*m_1d["v_peak"]):.2f}')
        rows.append(dict(
            v0=v0,
            v_focal_3d=v_focal_3d,
            target_z_1d=float(target_z),
            Vn2_3d=Vn2_3d.tolist(),
            Vn2_1d=Vn2_1d.tolist(),
            Vn2_3d_norm=Vn2_3d_norm.tolist(),
            Vn2_1d_norm=Vn2_1d_norm.tolist(),
            edge_1d=float(m_1d['a_peak']/(2*np.pi*F0_1D*m_1d['v_peak'])),
        ))

    if not rows:
        print('No data to aggregate.')
        return

    # Comparison plot: per-harmonic normalised |V_n|², 1D vs 3D
    n_axis = np.arange(1, 16)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, r in zip(axes.flat, rows):
        ax.semilogy(n_axis, np.array(r['Vn2_3d_norm']), 'bo-',
                    label='3-D focal on-axis', markerfacecolor='b', lw=1.6)
        ax.semilogy(n_axis, np.array(r['Vn2_1d_norm']), 'rs--',
                    label='1-D matched', markerfacecolor='none', lw=1.4)
        ax.set(xlabel='Harmonic n', ylabel='|V_n|² (normalised)',
               title=f'V₀={r["v0"]:.3f} m/s  '
                     f'(3D v_focal={r["v_focal_3d"]:.3e}, '
                     f'1D edge={r["edge_1d"]:.2f})',
               xlim=[0.5, 11], ylim=[1e-6, 2])
        ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
    for ax in axes.flat[len(rows):]:
        ax.axis('off')
    fig.suptitle('T4 — 1D vs 3D on-axis harmonic spectra',
                 fontsize=12)
    fig.savefig(os.path.join(OUT_ROOT, 'spectra_comparison.png'), dpi=160)
    plt.close(fig)

    # Tabulate
    lines = [f'T4 — 1D vs 3D harmonic-spectrum comparison',
             '=' * 90]
    lines.append(f'1-D matched: amplitude = 3D focal v_peak, '
                 f'z_target = z_R/2 = {0.5*np.pi*(LAM*FOCUS/(np.pi*WAIST_SRC))**2/LAM*100:.3f} cm')
    lines.append('')
    lines.append(f'{"V₀":>6} {"v_focal":>10} {"|V_3/V_1|² 3D":>14} {"1D":>10} '
                 f'{"|V_5/V_1|² 3D":>14} {"1D":>10} {"|V_7/V_1|² 3D":>14} {"1D":>10}')
    for r in rows:
        V3 = r['Vn2_3d_norm']; V1d = r['Vn2_1d_norm']
        lines.append(
            f'{r["v0"]:>6.3f} {r["v_focal_3d"]:>10.3e} '
            f'{V3[2]/max(V3[0],1e-30):>14.3e} '
            f'{V1d[2]/max(V1d[0],1e-30):>10.3e} '
            f'{V3[4]/max(V3[0],1e-30):>14.3e} '
            f'{V1d[4]/max(V1d[0],1e-30):>10.3e} '
            f'{V3[6]/max(V3[0],1e-30):>14.3e} '
            f'{V1d[6]/max(V1d[0],1e-30):>10.3e}'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'comparison_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    with open(os.path.join(OUT_ROOT, 'comparison.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)


if __name__ == '__main__':
    main()
