"""
T1 — Theoretical-prediction validation with a paraxial converging
Gaussian source.

Setup
-----
  Cubic shear shock, brain regime: F0=100 Hz, c=2 m/s, ρ=1000 kg/m³,
  β₃=5e8, α₀=0.5 Np/m·Hz, y=1.
  Gaussian source apodization: w_src = 3 cm.
  Geometric focus: F = 10 cm  →  paraxial focal waist
                                  w_focal = λF/(π w_src) ≈ 2.12 cm.
  Domain: DOMAIN_LAT = 6λ = 12 cm so 4σ of source apodization fits.
  PROP_DIST = F (capture exactly at geometric focal plane).

For each V₀ in V0_LIST:
  (1) Linear reference: same source but β₃=0, α₀=0 (linear, lossless).
      Returns the focal v² 2D map and its FWHM, defining a measured w_v_lin.
  (2) Nonlinear reference: full physics (β₃=5e8, α₀=0.5, y=1).
      Returns the focal v² map plus the on-axis trace at z=F.
  (3) Extract |V_n|² for n=1..15 from on-axis trace via FFT bin-sum.
  (4) Predict via theory_predictor.predict_ratios() using w0 = w_v_lin
      (a clean self-consistent baseline that's intrinsic to the simulation).
  (5) Measure simulated w_v, w_a, A_v², A_a², r_ring^(ε̇), r_ring^(γ)
      from the nonlinear field's 2D maps.
  (6) Tabulate predicted vs measured.

Outputs
-------
  validation_results/theory_validation/T1_gaussian/v0_<V>/
    linear_focal.npz, nonlinear_focal.npz
    metrics.json, comparison_table.txt
  validation_results/theory_validation/T1_gaussian/
    summary_table.txt, predicted_vs_measured.png, sweep_summary.json
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
from angular_spectrum_solver import SolverParams, angular_spectrum_solve
from cubic_setup import (
    cubic_dzmin, cubic_kt_params, expected_focal_peak_gaussian,
)
from make_gaussian_source import make_gaussian_source
from kinematic_analyzer import (
    kinematic_fields, lateral_2d_maps, area_above_threshold, fwhm,
)
from theory_predictor import (
    on_axis_harmonic_amplitudes, predict_ratios,
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
LAM       = C0 / F0
WAIST_SRC = 0.03            # 3 cm Gaussian source waist
FOCUS     = 0.10            # 10 cm geometric focus
W0_PRED   = LAM * FOCUS / (np.pi * WAIST_SRC)  # analytical Gaussian focal waist
PROP_DIST = FOCUS           # capture at z = F

# ---- 1× grid ----
DX        = LAM / 15
DT        = 1.0 / (60.0 * F0)
DOMAIN    = 6 * LAM

# ---- Sweep ----
V0_LIST = [0.01, 0.02, 0.03, 0.05, 0.10]

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T1_gaussian',
)
os.makedirs(OUT_ROOT, exist_ok=True)


def _grid_axes():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    return xaxis, taxis, nX, nT


def run_one(v0: float, linear: bool, cache_path: str):
    """One solve. linear=True → β₃=0, α₀=0."""
    if os.path.exists(cache_path):
        return np.load(cache_path)

    xaxis, taxis, nX, nT = _grid_axes()
    init = make_gaussian_source(
        xaxis, xaxis, taxis, F0, C0, v0,
        waist_source=WAIST_SRC, focus=FOCUS, ncycles=NCYCLES,
    )

    if linear:
        beta3_use = 0.0
        alpha0_use = 0.0
        atten_loss = False
    else:
        beta3_use = BETA3
        alpha0_use = ALPHA0
        atten_loss = True

    # dZmin (conservative; smaller for nonlinear)
    expected_peak = expected_focal_peak_gaussian(v0, F0, C0, WAIST_SRC, FOCUS)
    dZmin = cubic_dzmin(expected_peak, DT, C0, LAM, BETA3, RHO0, linear=linear)

    sp = cubic_kt_params(
        dX=DX, dT=DT, c0=C0, rho0=RHO0,
        beta3=beta3_use, alpha0=alpha0_use, attenPow=Y_ABS, f0=F0,
        propDist=PROP_DIST, dZmin=dZmin, useAttenLoss=atten_loss,
    )
    t0 = time.time()
    field, _, _, pI, _, zaxis, _ = angular_spectrum_solve(
        init, sp, verbose=False)
    wall = time.time() - t0
    np.savez(cache_path,
             field_focal=field.astype(np.float32),
             pI=pI.astype(np.float32),
             zaxis=zaxis.astype(np.float32),
             xaxis=xaxis.astype(np.float32),
             v0=v0, linear=linear, wall_s=wall)
    return np.load(cache_path)


def measure_focal_metrics(d_npz, label: str):
    """From an npz of a focal capture, return measured w_v, w_a, areas,
    ring radii, and on-axis harmonic spectrum."""
    field = d_npz['field_focal']
    xaxis = d_npz['xaxis']
    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    x_mm = xaxis * 1e3

    kf = kinematic_fields(field, DT, DX)
    maps = lateral_2d_maps(field, kf)

    # FWHMs from y=0 lateral profiles (centred maps)
    profile_v   = maps['pI'][:, cy]                  # ∫v² dt (centred)
    profile_a   = maps['a_meansq'][:, cy]            # ⟨a²⟩ (centred)
    fwhm_v = fwhm(profile_v, x_mm)                   # mm
    fwhm_a = fwhm(profile_a, x_mm)

    # Half-max areas from full 2-D maps
    area_v = area_above_threshold(maps['pI'],            DX, DX) * 1e6  # mm²
    area_a = area_above_threshold(maps['a_meansq'],      DX, DX) * 1e6

    # Ring radii from annular maps (max of lateral profile at y=0 over x>0)
    prof_eps = maps['eps_meansq'][:, cy]
    prof_gam = maps['gamma_vm_meansq'][:, cy]
    # Search x>0 half for ring peak
    i_pos = np.where(xaxis > 0)[0]
    r_eps_idx = i_pos[int(np.argmax(prof_eps[i_pos]))]
    r_gam_idx = i_pos[int(np.argmax(prof_gam[i_pos]))]
    r_ring_eps = float(xaxis[r_eps_idx])
    r_ring_gam = float(xaxis[r_gam_idx])

    # On-axis spectrum
    trace = field[cx, cy, :]
    Vn2 = on_axis_harmonic_amplitudes(trace, DT, F0, n_max=15)

    return dict(
        fwhm_v_mm=float(fwhm_v),
        fwhm_a_mm=float(fwhm_a),
        area_v_mm2=float(area_v),
        area_a_mm2=float(area_a),
        r_ring_eps_mm=float(r_ring_eps * 1e3),
        r_ring_gam_mm=float(r_ring_gam * 1e3),
        on_axis_v_peak=float(np.max(np.abs(trace))),
        on_axis_a_peak=float(np.max(np.abs(kf['a'][cx, cy, :]))),
        Vn2=Vn2.tolist(),
        label=label,
    )


def per_v0(v0: float):
    out_dir = os.path.join(OUT_ROOT, f'v0_{v0:.3f}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n=== V₀ = {v0} m/s ===')
    print(f'  Gaussian w_src = {WAIST_SRC*100:.1f} cm, F = {FOCUS*100:.0f} cm')
    print(f'  Analytical w_0 (paraxial Gaussian focus) = {W0_PRED*1e3:.2f} mm')

    # 1. Linear reference
    d_lin = run_one(v0, linear=True,
                    cache_path=os.path.join(out_dir, 'linear_focal.npz'))
    m_lin = measure_focal_metrics(d_lin, 'linear')
    print(f'  Linear:    v_peak={m_lin["on_axis_v_peak"]:.4e}, '
          f'pI FWHM={m_lin["fwhm_v_mm"]:.2f} mm')

    # 2. Nonlinear (cubic + lossy)
    d_nl = run_one(v0, linear=False,
                   cache_path=os.path.join(out_dir, 'nonlinear_focal.npz'))
    m_nl = measure_focal_metrics(d_nl, 'nonlinear')
    print(f'  Nonlinear: v_peak={m_nl["on_axis_v_peak"]:.4e}, '
          f'pI FWHM={m_nl["fwhm_v_mm"]:.2f} mm, '
          f'a_peak={m_nl["on_axis_a_peak"]:.3e}')

    # 3. Theory predictions from nonlinear on-axis spectrum
    Vn2 = np.array(m_nl['Vn2'])
    # We can also restrict to odd n (cubic shear cascade)
    n_indices = np.arange(1, len(Vn2) + 1, dtype=float)
    # Use the GAUSSIAN-FIT w_0 from the LINEAR reference, which is the
    # 'linear focal waist' of the simulation (self-consistent baseline):
    w0_lin_fit = m_lin['fwhm_v_mm'] / np.sqrt(2 * np.log(2)) * 1e-3   # m
    # Predicted ratios use the same w0 throughout
    pred = predict_ratios(Vn2, n_indices, w0=w0_lin_fit)
    print(f'  ⟨n⟩_v = {pred["n_v"]:.4f}, ⟨n⟩_{{a²}} = {pred["n_a2"]:.4f}')
    print(f'  Predicted:  w_a/w_v={pred["wa_over_wv"]:.4f}, '
          f'A_a²/A_v²={pred["Aa2_over_Av2"]:.4f}')
    print(f'              r_ring^(ε̇)={pred["r_ring_eps"]*1e3:.2f} mm, '
          f'r_ring^(γ)={pred["r_ring_gam"]*1e3:.2f} mm')

    # 4. Measured ratios from nonlinear simulation
    meas_wa_over_wv = (m_nl['fwhm_a_mm']
                       / m_nl['fwhm_v_mm']) if m_nl['fwhm_v_mm'] > 0 else float('nan')
    meas_Aa_over_Av = (m_nl['area_a_mm2']
                       / m_nl['area_v_mm2']) if m_nl['area_v_mm2'] > 0 else float('nan')
    print(f'  Measured:   w_a/w_v={meas_wa_over_wv:.4f}, '
          f'A_a²/A_v²={meas_Aa_over_Av:.4f}')
    print(f'              r_ring^(ε̇)={m_nl["r_ring_eps_mm"]:.2f} mm, '
          f'r_ring^(γ)={m_nl["r_ring_gam_mm"]:.2f} mm')

    # 5. Pack and save
    meas = dict(
        wa_over_wv=meas_wa_over_wv,
        Aa2_over_Av2=meas_Aa_over_Av,
        r_ring_eps_mm=m_nl['r_ring_eps_mm'],
        r_ring_gam_mm=m_nl['r_ring_gam_mm'],
        wv_fwhm_mm=m_nl['fwhm_v_mm'],
        wv_lin_fwhm_mm=m_lin['fwhm_v_mm'],
    )
    pred_named = dict(
        n_v=pred['n_v'], n_a2=pred['n_a2'],
        wa_over_wv=pred['wa_over_wv'],
        Aa2_over_Av2=pred['Aa2_over_Av2'],
        r_ring_eps_mm=pred['r_ring_eps'] * 1e3,
        r_ring_gam_mm=pred['r_ring_gam'] * 1e3,
        wv_over_w0=pred['wv_over_w0'],
        wa_over_w0=pred['wa_over_w0'],
        w0_lin_fit_mm=w0_lin_fit * 1e3,
    )
    pct = lambda p, m: (m - p) / max(abs(p), 1e-30) * 100 if (p == p and m == m) else float('nan')
    errors = dict(
        wa_over_wv_pct=pct(pred_named['wa_over_wv'], meas['wa_over_wv']),
        Aa2_over_Av2_pct=pct(pred_named['Aa2_over_Av2'], meas['Aa2_over_Av2']),
        r_ring_eps_pct=pct(pred_named['r_ring_eps_mm'], meas['r_ring_eps_mm']),
        r_ring_gam_pct=pct(pred_named['r_ring_gam_mm'], meas['r_ring_gam_mm']),
    )
    print(f'  Error:      w_a/w_v {errors["wa_over_wv_pct"]:+.1f}%, '
          f'A_a²/A_v² {errors["Aa2_over_Av2_pct"]:+.1f}%, '
          f'r_ring^(ε̇) {errors["r_ring_eps_pct"]:+.1f}%, '
          f'r_ring^(γ) {errors["r_ring_gam_pct"]:+.1f}%')

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(dict(v0=v0, w0_pred_analytic_mm=W0_PRED * 1e3,
                       linear=m_lin, nonlinear=m_nl,
                       predicted=pred_named, measured=meas,
                       errors=errors), f, indent=2)

    return dict(v0=v0, linear=m_lin, nonlinear=m_nl,
                predicted=pred_named, measured=meas, errors=errors)


def aggregate(all_results):
    # Text table
    lines = []
    lines.append('T1 — Gaussian-source theory validation')
    lines.append('=' * 96)
    lines.append(f'Gaussian source: w_src = {WAIST_SRC*100:.1f} cm, F = '
                 f'{FOCUS*100:.0f} cm')
    lines.append(f'Analytical paraxial focal waist '
                 f'w_0^{{ana}} = λF/(π w_src) = {W0_PRED*1e3:.2f} mm')
    lines.append(f'Brain shear regime: F0={F0} Hz, c={C0} m/s, β₃={BETA3:.1e},'
                 f' α₀={ALPHA0}, y={Y_ABS}')
    lines.append(f'1× grid: DX = λ/15 = {DX*1e3:.2f} mm, '
                 f'DT = 1/(60·F0) = {DT*1e3:.3f} ms')
    lines.append('')
    lines.append('Each row: predicted from on-axis harmonic spectrum at '
                 'z=F vs measured from 2D maps.')
    lines.append('')
    h = (f'{"V₀":>6} {"⟨n⟩_v":>7} {"⟨n⟩_a²":>7} '
         f'{"w_a/w_v pred":>13} {"meas":>7} {"%err":>7} '
         f'{"A_a²/A_v² pred":>15} {"meas":>7} {"%err":>7} '
         f'{"r_ε̇ pred":>10} {"meas":>7} {"%err":>7} '
         f'{"r_γ pred":>10} {"meas":>7} {"%err":>7}')
    lines.append(h)
    lines.append('-' * len(h))
    for r in all_results:
        p = r['predicted']; m = r['measured']; e = r['errors']
        lines.append(
            f'{r["v0"]:>6.3f} {p["n_v"]:>7.3f} {p["n_a2"]:>7.3f} '
            f'{p["wa_over_wv"]:>13.4f} {m["wa_over_wv"]:>7.4f} {e["wa_over_wv_pct"]:>+6.1f}% '
            f'{p["Aa2_over_Av2"]:>15.4f} {m["Aa2_over_Av2"]:>7.4f} {e["Aa2_over_Av2_pct"]:>+6.1f}% '
            f'{p["r_ring_eps_mm"]:>10.2f} {m["r_ring_eps_mm"]:>7.2f} {e["r_ring_eps_pct"]:>+6.1f}% '
            f'{p["r_ring_gam_mm"]:>10.2f} {m["r_ring_gam_mm"]:>7.2f} {e["r_ring_gam_pct"]:>+6.1f}%'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'summary_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    # Predicted-vs-measured scatter
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    quantities = [
        ('wa_over_wv',  'wa_over_wv',      'w_a / w_v (dimensionless)'),
        ('Aa2_over_Av2','Aa2_over_Av2',    'A_a² / A_v² (dimensionless)'),
        ('r_ring_eps_mm','r_ring_eps_mm',  'r_ring^(ε̇)  (mm)'),
        ('r_ring_gam_mm','r_ring_gam_mm',  'r_ring^(γ)   (mm)'),
    ]
    for ax, (pk, mk, lbl) in zip(axes.flat, quantities):
        p_vals = [r['predicted'][pk] for r in all_results]
        m_vals = [r['measured'][mk]  for r in all_results]
        v0s    = [r['v0']            for r in all_results]
        ax.plot(p_vals, m_vals, 'o', markersize=9)
        for px, mx, v0 in zip(p_vals, m_vals, v0s):
            ax.annotate(f'V₀={v0:.3f}', (px, mx),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=8)
        lo = min(min(p_vals), min(m_vals))
        hi = max(max(p_vals), max(m_vals))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x (perfect)')
        ax.set(xlabel=f'predicted {lbl}', ylabel=f'measured {lbl}',
               title=lbl)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('T1 Gaussian-source theory validation: predicted vs measured',
                 fontsize=12)
    fig.savefig(os.path.join(OUT_ROOT, 'predicted_vs_measured.png'), dpi=170)
    plt.close(fig)

    with open(os.path.join(OUT_ROOT, 'sweep_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)


def main():
    print(f'T1 — Gaussian-source theory validation')
    print(f'Source: w_src = {WAIST_SRC*100} cm, F = {FOCUS*100} cm')
    print(f'V₀ sweep: {V0_LIST}')
    print(f'Output: {OUT_ROOT}')
    all_results = []
    for v0 in V0_LIST:
        all_results.append(per_v0(v0))
    aggregate(all_results)


if __name__ == '__main__':
    main()
