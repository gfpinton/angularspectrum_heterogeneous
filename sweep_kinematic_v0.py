"""
V₀ sweep for kinematic-FWHM investigation (Step 2 of KINEMATIC_PIVOT.md,
moved up before Step 0 per user direction).

The 1× cached result showed a/pI ≈ 1.89 (wider than pI, opposite to the
pivot doc's hypothesis). Before committing to expensive refinement, we
sweep V₀ at the cheap 1× grid to see whether any operating point shows
a kinematic-narrowing ratio < 1 (i.e. whether the hypothesis is alive
anywhere in the σ_f-axis).

V₀ ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40} m/s → M = V₀/c ∈
       {0.025, 0.05, 0.075, 0.10, 0.15, 0.20}.

For each V₀:
  - Run lossy cubic bowl solve to PROP_DIST just past the expected
    z_focus.
  - Use per_step_callback to capture the FIELD at the z step that
    maximises on-axis ∫v²dt — i.e. the true z_focus.
  - Run kinematic_analyzer.summarize on the field-at-z_focus.
  - Save kinematic_metrics.json + headline figure to:
      validation_results/kinematic_sigma_sweep/v0_{V₀:.3f}/

After all V₀s finish, aggregate into:
  validation_results/kinematic_sigma_sweep/
    sweep_table.txt
    sweep_ratios.png   — area_ratio and FWHM_ratio vs M, per quantity
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

import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source,
)
from cubic_setup import (
    cubic_dzmin, cubic_kt_params, expected_focal_peak_bowl,
)
from kinematic_analyzer import summarize, area_above_threshold


# ---- physical regime (brain shear, lossy) — identical to
#      validate_cubic_shear_bowl_lossy.py except V₀ varies ----
F0      = 100.0
C0      = 2.0
RHO0    = 1000.0
BETA3   = 5.0e8
LAM     = C0 / F0          # 0.02 m
ROC     = 5 * LAM          # 0.10 m
OUTER_R = 2.5 * LAM        # 0.05 m
FOCUS   = ROC
NCYCLES = 4
DUR     = 4

# ---- 1× numerics (current baseline) ----
DX        = LAM / 15       # 1.333 mm
DT        = 1.0 / (60.0 * F0)
DOMAIN    = 6 * LAM
# In the lossy cubic shear regime, z_focus is observed near ~3.9 cm
# (well before geometric focus at 10 cm) due to heavy linear-in-f
# attenuation. Propagating to 5 cm covers it with headroom across the
# V₀ range; the callback captures field at the actual max-pI step.
PROP_DIST = 0.05

# ---- shear loss (Catheline / Pinton 2010 convention: linear-in-f) ----
ALPHA0    = 0.5
Y         = 1.0

# ---- sweep ----
V0_LIST   = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'kinematic_sigma_sweep',
)
os.makedirs(OUT_ROOT, exist_ok=True)


class FocalCapture:
    """per_step_callback that snapshots the field at max on-axis pI(z).

    Computes the running on-axis pI as a JAX scalar (cheap sync) and
    materialises the full field to host only when a new running-max is
    observed *within the focal region* [z_min, z_max]. The focal-region
    gate is needed because at high σ_f, the on-axis pI can be higher
    near the source than at the geometric focal region — that's
    acoustic saturation, not focusing, and would mislead the kinematic
    analysis.
    """
    def __init__(self, cx: int, cy: int,
                 z_min: float = 0.025, z_max: float = float('inf')):
        self.cx = cx; self.cy = cy
        self.z_min = z_min
        self.z_max = z_max
        self.pI_max = -1.0
        self.z_focus = None
        self.idx = None
        self.field_at_focus = None
        self.history = []     # (cc, z_cum, pI_axial)

    def __call__(self, cc, z_cum, field):
        on_axis = field[self.cx, self.cy, :]
        pI_axial = float(jnp.sum(on_axis * on_axis))
        self.history.append((int(cc), float(z_cum), pI_axial))
        if z_cum < self.z_min or z_cum > self.z_max:
            return
        if pI_axial > self.pI_max:
            self.pI_max = pI_axial
            self.z_focus = float(z_cum)
            self.idx = int(cc)
            self.field_at_focus = np.asarray(field, dtype=np.float32).copy()


def run_one(v0: float, out_dir: str, enable_diagnostic: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, 'fields_at_zfocus.npz')

    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 2.5 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT

    if os.path.exists(cache):
        d = np.load(cache)
        print(f'  V₀={v0}: using cached field-at-focus '
              f'(z_focus={float(d["z_focus"])*100:.2f} cm)')
        return (d['field_at_focus'], float(d['z_focus']),
                d['zaxis'], d['xaxis'])

    print(f'  V₀={v0}: starting solve to PROP_DIST = {PROP_DIST*100:.2f} cm '
          f'(grid {nX}×{nX}×{nT})')

    init = make_bowl_source(
        xaxis, xaxis, taxis, F0, C0, v0,
        radius=OUTER_R, roc=ROC, focus=FOCUS,
        ncycles=NCYCLES, dur=DUR,
    )
    expected_peak = expected_focal_peak_bowl(v0, F0, C0, OUTER_R, FOCUS)
    dZmin = cubic_dzmin(expected_peak, DT, C0, LAM, BETA3, RHO0)
    diag_kwargs = {}
    if enable_diagnostic:
        diag_dir = os.path.join(out_dir, 'as_diagnostic_frames')
        os.makedirs(diag_dir, exist_ok=True)
        diag_kwargs = dict(
            diagnostic=True,
            diagnosticInterval=50,
            diagnosticDir=diag_dir,
            diagnosticSummary=True,
            diagnosticInitialConditions=True,
        )

    sp = cubic_kt_params(
        dX=DX, dT=DT, c0=C0, rho0=RHO0,
        beta3=BETA3, alpha0=ALPHA0, attenPow=Y, f0=F0,
        propDist=PROP_DIST, dZmin=dZmin, useAttenLoss=True,
        **diag_kwargs,
    )

    cx = cy = nX // 2
    cap = FocalCapture(cx, cy)
    t0 = time.time()
    field_zend, _, _, pI, _, zaxis, _ = angular_spectrum_solve(
        init, sp, verbose=False, per_step_callback=cap)
    wall = time.time() - t0
    print(f'  V₀={v0}: solve done in {wall:.0f}s, '
          f'{len(zaxis)} z-steps, z_focus={cap.z_focus*100:.2f} cm')

    np.savez(cache,
             field_at_focus=cap.field_at_focus,
             z_focus=cap.z_focus,
             zaxis=zaxis.astype(np.float32),
             xaxis=xaxis.astype(np.float32),
             # for verification
             pI=pI.astype(np.float32),
             v0=v0, wall_s=wall)
    return cap.field_at_focus, cap.z_focus, zaxis, xaxis


def analyze_one(field, z_focus, xaxis, v0, out_dir):
    s = summarize(field, DT, DX, F0, xaxis)
    M = v0 / C0

    # ---- write JSON ----
    metrics = {
        'v0_m_per_s': v0,
        'M_v0_over_c': M,
        'z_focus_m': z_focus,
        'DX_m': DX, 'DT_s': DT, 'DX_over_lam': LAM/DX,
        'samples_per_cycle': 1.0/(DT*F0),
        'fwhms_mm': s['fwhms_mm'],
        'fwhm_ratios_to_pI': s['ratios'],
        'areas_mm2': s['areas_mm2'],
        'area_ratios_to_pI': s['area_ratios'],
        'on_axis': s['on_axis'],
        'diag': s['diag'],
    }
    with open(os.path.join(out_dir, 'kinematic_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ---- write text summary (all maps are dim-matched to pI: quadratic-in-v) ----
    lines = []
    lines.append(f'V₀ = {v0} m/s   (M = V₀/c = {M:.4f})')
    lines.append(f'z_focus = {z_focus*100:.2f} cm')
    lines.append(f'On-axis peaks: v={s["on_axis"]["v_peak"]:.4f} m/s, '
                 f'|a|_pk={s["on_axis"]["a_peak"]:.3e} m/s², '
                 f'|ε̇|_pk={s["on_axis"]["eps_peak"]:.3e} 1/s')
    lines.append(f'Edge-steepness={s["diag"]["edge_steepness"]:.2f}, '
                 f'odd-frac={s["diag"]["odd_frac"]:.3f}')
    lines.append('')
    lines.append('All maps below are dimensionally matched to pI '
                 '(quadratic in v).')
    lines.append(f'{"Quantity":<14} {"FWHM_mm":>10} '
                 f'{"FWHM/pI":>10} {"Area_mm²":>12} {"Area/pI":>10}')
    lines.append('-' * 60)
    for k in ['pI', 'a_meansq', 'a_peaksq',
              'eps_meansq', 'eps_peaksq',
              'gamma_vm_meansq', 'gamma_vm_peaksq']:
        w = s['fwhms_mm'][k]
        if k == 'pI':
            ar = s['areas_mm2']['pI']
            lines.append(f'{k:<14} {w:>10.3f} {"-":>10} '
                         f'{ar:>12.1f} {"-":>10}')
        else:
            w_r = w / s['fwhms_mm']['pI']
            ar = s['areas_mm2'].get(k, float('nan'))
            ar_r = s['area_ratios'].get(f'{k}/pI', float('nan'))
            lines.append(f'{k:<14} {w:>10.3f} {w_r:>10.3f} '
                         f'{ar:>12.1f} {ar_r:>10.3f}')
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(out_dir, 'kinematic_summary.txt'), 'w') as f:
        f.write(text)
    print(text)

    # ---- per-V₀ headline plot ----
    x_mm = s['x_mm']
    P = s['profiles']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    keys = ['pI', 'a_meansq', 'a_peaksq', 'eps_peaksq', 'gamma_vm_peaksq']
    colors = ['k', 'tab:blue', 'tab:cyan', 'tab:red', 'tab:purple']
    styles = ['-', '-', '--', '--', ':']
    for k, c, ls in zip(keys, colors, styles):
        p = np.asarray(P[k], dtype=np.float64)
        pn = p / (p.max() + 1e-30)
        w = s['fwhms_mm'][k]
        axes[0].plot(x_mm, pn, ls, color=c, lw=1.6,
                     label=f'{k} (FWHM={w:.2f} mm)')
        axes[1].plot(x_mm, 10*np.log10(np.maximum(pn, 1e-5)),
                     ls, color=c, lw=1.4, label=f'{k}')
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.set(xlabel='x (mm)', xlim=[-50, 50])
    axes[0].set(ylabel='Normalised', title='Lateral profiles (linear)')
    axes[1].set(ylabel='dB rel. peak', ylim=[-30, 1], title='Same, dB')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'V₀={v0} m/s, M={M:.4f}, z_focus={z_focus*100:.2f} cm  '
                 f'(1× grid)', fontsize=11)
    fig.savefig(os.path.join(out_dir, 'kinematic_focal_FWHM.png'), dpi=160)
    plt.close(fig)

    return metrics


def aggregate(all_metrics):
    """Build sweep_table.txt + sweep_ratios.png."""
    lines = []
    lines.append('V₀ sweep — kinematic-FWHM at 1× grid (cubic shear bowl)')
    lines.append('=' * 65)
    lines.append(f'F#=1, F0=100 Hz, α₀=0.5, y=1, β₃=5e8, '
                 f'c=2 m/s, ρ=1000 kg/m³')
    lines.append(f'Grid: DX=λ/15={DX*1e3:.2f} mm, DT=1/(60·F0)={DT*1e3:.3f} ms')
    lines.append('')
    lines.append('All ratios use dimensionally-matched maps '
                 '(quadratic-in-v) so the linear-Gaussian baseline is 1.0.')
    lines.append(f'{"V₀":>6} {"M":>6} {"z_f_cm":>8} '
                 f'{"v_pk":>8} {"a_pk":>10} {"eps_pk":>10} '
                 f'{"edge":>6} {"odd%":>6} '
                 f'{"a²_mn_w":>9} {"a²_pk_w":>9} '
                 f'{"a²_mn_A":>9} {"a²_pk_A":>9} {"ε̇²_pk_A":>9}')
    lines.append('-' * 130)
    for m in all_metrics:
        v0 = m['v0_m_per_s']
        lines.append(
            f'{v0:>6.3f} {m["M_v0_over_c"]:>6.4f} '
            f'{m["z_focus_m"]*100:>8.2f} '
            f'{m["on_axis"]["v_peak"]:>8.4f} '
            f'{m["on_axis"]["a_peak"]:>10.3e} '
            f'{m["on_axis"]["eps_peak"]:>10.3e} '
            f'{m["diag"]["edge_steepness"]:>6.2f} '
            f'{m["diag"]["odd_frac"]*100:>6.1f} '
            f'{m["fwhm_ratios_to_pI"]["a_meansq/pI"]:>9.3f} '
            f'{m["fwhm_ratios_to_pI"]["a_peaksq/pI"]:>9.3f} '
            f'{m["area_ratios_to_pI"]["a_meansq/pI"]:>9.3f} '
            f'{m["area_ratios_to_pI"]["a_peaksq/pI"]:>9.3f} '
            f'{m["area_ratios_to_pI"]["eps_peaksq/pI"]:>9.3f}'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'sweep_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    # ---- plot ratios vs M ----
    Ms = [m['M_v0_over_c'] for m in all_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    keys = ['a_meansq/pI', 'a_peaksq/pI',
            'eps_meansq/pI', 'eps_peaksq/pI',
            'gamma_vm_peaksq/pI']
    colors = ['tab:blue', 'tab:cyan', 'tab:red', 'tab:orange', 'tab:purple']

    # Panel 1: FWHM ratio (meaningful for a*, dodgy for eps*)
    for k, c in zip(keys, colors):
        ratios = [m['fwhm_ratios_to_pI'][k] for m in all_metrics]
        axes[0].plot(Ms, ratios, 'o-', color=c, lw=1.6, label=k)
    axes[0].axhline(1, color='k', ls=':', lw=0.8)
    axes[0].set(xlabel='M = V₀/c', ylabel='FWHM ratio to pI',
                title='Lateral FWHM(y=0) ratio vs M')
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)

    # Panel 2: AREA ratio (uniformly meaningful across all quantities)
    for k, c in zip(keys, colors):
        ratios = [m['area_ratios_to_pI'][k] for m in all_metrics]
        axes[1].plot(Ms, ratios, 'o-', color=c, lw=1.6, label=k)
    axes[1].axhline(1, color='k', ls=':', lw=0.8)
    axes[1].set(xlabel='M = V₀/c',
                ylabel='Area-above-half-peak ratio to pI',
                title='Focal-plane half-max area ratio vs M')
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    fig.suptitle('Cubic shear bowl, F#=1, lossy (α₀=0.5, y=1), 1× grid — '
                 'kinematic concentration vs source Mach', fontsize=11)
    fig.savefig(os.path.join(OUT_ROOT, 'sweep_ratios.png'), dpi=170)
    plt.close(fig)

    # JSON dump
    with open(os.path.join(OUT_ROOT, 'sweep_summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)


def main():
    print(f'V₀ sweep: {V0_LIST}')
    print(f'Output root: {OUT_ROOT}')
    all_metrics = []
    for v0 in V0_LIST:
        out_dir = os.path.join(OUT_ROOT, f'v0_{v0:.3f}')
        field, z_focus, zaxis, xaxis = run_one(v0, out_dir)
        m = analyze_one(field, z_focus, xaxis, v0, out_dir)
        all_metrics.append(m)
    aggregate(all_metrics)
    print(f'\nSweep done. See {OUT_ROOT}/sweep_table.txt and sweep_ratios.png')


if __name__ == '__main__':
    main()
