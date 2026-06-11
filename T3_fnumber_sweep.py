"""
T3 — F# sweep validating Airy-correction direction.

Bowl source (Airy focal pattern, F#-dependent main-lobe FWHM), fixed V₀
near the kinematic optimum. Theory predicts that tighter F# (smaller
relative aperture) gives a sharper central lobe; for the central-lobe
rule predictions of hyperfocal_kinematic.tex, the IDENTICAL spectrum
should reproduce the SAME ⟨n⟩-based ratios up to Airy-sharpening
corrections that strengthen narrowing as F# decreases.

Setup: bowl source (make_bowl_source from angular_spectrum_solver),
fixed F=10 cm, sweep aperture radius a so F# = F/(2a) ∈ {1, 1.5, 2, 3}.
Capture at z = F (geometric focus).

For each F#:
  (1) Linear-reference solve.
  (2) Cubic+lossy solve.
  (3) Extract on-axis spectrum.
  (4) Predict central-lobe-rule ratios.
  (5) Measure 2D-map ratios.
  (6) Compute predicted-vs-measured percent error.

Outputs: validation_results/theory_validation/T3_fnumber/fn_<F#>/...
         validation_results/theory_validation/T3_fnumber/summary_table.txt
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
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve, make_bowl_source,
)
from kinematic_analyzer import (
    kinematic_fields, lateral_2d_maps, area_above_threshold, fwhm,
)
from theory_predictor import on_axis_harmonic_amplitudes, predict_ratios
from T1_validate_gaussian import (
    F0, C0, RHO0, BETA3, ALPHA0, Y_ABS, NCYCLES,
    LAM, FOCUS, DOMAIN, DX, DT,
)

V0_REF   = 0.025
FN_LIST  = [1.0, 1.5, 2.0, 3.0]

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T3_fnumber',
)
os.makedirs(OUT_ROOT, exist_ok=True)


def _grid():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    return xaxis, taxis, nX, nT


def run_one(fn, linear, cache):
    if os.path.exists(cache):
        return np.load(cache)
    xaxis, taxis, nX, nT = _grid()
    outer_r = FOCUS / (2.0 * fn)
    init = make_bowl_source(
        xaxis, xaxis, taxis, F0, C0, V0_REF,
        radius=outer_r, roc=FOCUS, focus=FOCUS,
        ncycles=NCYCLES, dur=4,
    )
    if linear:
        beta3_use, alpha0_use, atten_loss = 0.0, 0.0, False
        N3 = 0.0
    else:
        beta3_use, alpha0_use, atten_loss = BETA3, ALPHA0, True
        N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * outer_r**2 / (2 * FOCUS)
    expected_peak = max(V0_REF * focus_gain, V0_REF)
    safe_dZ = 0.15 * DT / max(expected_peak**2 * N3, 1e-30) if not linear \
              else LAM / 10.0
    dZmin = max(0.25 * safe_dZ, DT * C0 / 4.0)
    dZmin = min(dZmin, LAM / 10.0)
    sp = SolverParams(
        dX=DX, dY=DX, dT=DT, c0=C0, rho0=RHO0,
        beta=0.0, beta3=beta3_use, nonlinearityOrder=3,
        alpha0=alpha0_use, attenPow=Y_ABS, f0=F0,
        propDist=FOCUS,
        useSplitStep=True, fluxScheme='kt', useTVD=True,
        useAdaptiveFiltering=True,
        boundaryProfile='quadratic',
        useFreqWeightedBoundary=False,
        useSuperAbsorbing=False, boundaryFactor=0.15,
        dZmin=dZmin, useGPUReductions=False, useAttenLoss=atten_loss,
    )
    t0 = time.time()
    field, _, _, pI, _, zaxis, _ = angular_spectrum_solve(init, sp, verbose=False)
    wall = time.time() - t0
    np.savez(cache,
             field_focal=field.astype(np.float32),
             pI=pI.astype(np.float32),
             zaxis=zaxis.astype(np.float32),
             xaxis=xaxis.astype(np.float32),
             v0=V0_REF, linear=linear, wall_s=wall, fn=fn,
             outer_r=outer_r)
    return np.load(cache)


def measure(d):
    field = d['field_focal']
    xaxis = d['xaxis']
    nX, nY, nT = field.shape
    cx, cy = nX // 2, nY // 2
    x_mm = xaxis * 1e3
    kf = kinematic_fields(field, DT, DX)
    maps = lateral_2d_maps(field, kf)
    fwhm_v = fwhm(maps['pI'][:, cy], x_mm)
    fwhm_a = fwhm(maps['a_meansq'][:, cy], x_mm)
    area_v = area_above_threshold(maps['pI'], DX, DX) * 1e6
    area_a = area_above_threshold(maps['a_meansq'], DX, DX) * 1e6
    prof_eps = maps['eps_meansq'][:, cy]
    prof_gam = maps['gamma_vm_meansq'][:, cy]
    i_pos = np.where(xaxis > 0)[0]
    r_eps = float(xaxis[i_pos[int(np.argmax(prof_eps[i_pos]))]]) * 1e3
    r_gam = float(xaxis[i_pos[int(np.argmax(prof_gam[i_pos]))]]) * 1e3
    trace = field[cx, cy, :]
    Vn2 = on_axis_harmonic_amplitudes(trace, DT, F0, n_max=15)
    return dict(
        fwhm_v_mm=float(fwhm_v),
        fwhm_a_mm=float(fwhm_a),
        area_v_mm2=float(area_v),
        area_a_mm2=float(area_a),
        r_ring_eps_mm=r_eps,
        r_ring_gam_mm=r_gam,
        on_axis_v_peak=float(np.max(np.abs(trace))),
        on_axis_a_peak=float(np.max(np.abs(kf['a'][cx, cy, :]))),
        Vn2=Vn2.tolist(),
    )


def main():
    print(f'T3 — F# sweep at V₀ = {V0_REF} m/s, bowl source')
    rows = []
    for fn in FN_LIST:
        print(f'\n=== F# = {fn:.1f} (aperture radius = '
              f'{FOCUS/(2*fn)*100:.2f} cm) ===')
        out_dir = os.path.join(OUT_ROOT, f'fn_{fn:.1f}')
        os.makedirs(out_dir, exist_ok=True)
        d_lin = run_one(fn, linear=True,
                        cache=os.path.join(out_dir, 'linear.npz'))
        m_lin = measure(d_lin)
        d_nl = run_one(fn, linear=False,
                       cache=os.path.join(out_dir, 'nonlinear.npz'))
        m_nl = measure(d_nl)
        print(f'  Linear w_v FWHM (Airy main lobe) = {m_lin["fwhm_v_mm"]:.2f} mm')
        print(f'  Nonlinear w_v FWHM = {m_nl["fwhm_v_mm"]:.2f} mm, '
              f'a_peak = {m_nl["on_axis_a_peak"]:.3e}')

        Vn2 = np.array(m_nl['Vn2'])
        w0_lin = m_lin['fwhm_v_mm'] / np.sqrt(2*np.log(2)) * 1e-3
        pred = predict_ratios(Vn2, np.arange(1, len(Vn2)+1, dtype=float),
                              w0=w0_lin)
        meas_wa_wv = m_nl['fwhm_a_mm'] / max(m_nl['fwhm_v_mm'], 1e-30)
        meas_Aa_Av = m_nl['area_a_mm2'] / max(m_nl['area_v_mm2'], 1e-30)
        print(f'  ⟨n⟩_v={pred["n_v"]:.4f}, ⟨n⟩_a²={pred["n_a2"]:.4f}')
        print(f'  pred wa/wv={pred["wa_over_wv"]:.4f}, meas={meas_wa_wv:.4f}')
        print(f'  pred Aa/Av={pred["Aa2_over_Av2"]:.4f}, meas={meas_Aa_Av:.4f}')
        rows.append(dict(
            fn=fn, outer_r_cm=FOCUS/(2*fn)*100,
            n_v=pred['n_v'], n_a2=pred['n_a2'],
            wv_lin_fwhm_mm=m_lin['fwhm_v_mm'],
            wv_nl_fwhm_mm=m_nl['fwhm_v_mm'],
            wa_nl_fwhm_mm=m_nl['fwhm_a_mm'],
            pred_wa_over_wv=pred['wa_over_wv'],
            meas_wa_over_wv=meas_wa_wv,
            pred_Aa_over_Av=pred['Aa2_over_Av2'],
            meas_Aa_over_Av=meas_Aa_Av,
            pred_r_eps_mm=pred['r_ring_eps']*1e3,
            meas_r_eps_mm=m_nl['r_ring_eps_mm'],
            pred_r_gam_mm=pred['r_ring_gam']*1e3,
            meas_r_gam_mm=m_nl['r_ring_gam_mm'],
            edge_steep=m_nl['on_axis_a_peak']/(2*np.pi*F0*m_nl['on_axis_v_peak']),
        ))

    # Aggregate
    lines = [f'T3 — F# sweep, bowl source, V₀ = {V0_REF}',
             '=' * 110]
    lines.append(f'{"F#":>5} {"a(cm)":>7} {"wv_lin":>7} {"wv_nl":>7} '
                 f'{"⟨n⟩_v":>7} {"⟨n⟩_a²":>7} '
                 f'{"wa/wv pred":>11} {"meas":>7} {"%err":>7} '
                 f'{"Aa/Av pred":>11} {"meas":>7} {"%err":>7} '
                 f'{"edge":>6}')
    lines.append('-' * 115)
    def pct(p, m): return (m-p)/max(abs(p),1e-30)*100
    for r in rows:
        lines.append(
            f'{r["fn"]:>5.1f} {r["outer_r_cm"]:>7.2f} '
            f'{r["wv_lin_fwhm_mm"]:>7.2f} {r["wv_nl_fwhm_mm"]:>7.2f} '
            f'{r["n_v"]:>7.4f} {r["n_a2"]:>7.4f} '
            f'{r["pred_wa_over_wv"]:>11.4f} {r["meas_wa_over_wv"]:>7.4f} '
            f'{pct(r["pred_wa_over_wv"], r["meas_wa_over_wv"]):>+6.1f}% '
            f'{r["pred_Aa_over_Av"]:>11.4f} {r["meas_Aa_over_Av"]:>7.4f} '
            f'{pct(r["pred_Aa_over_Av"], r["meas_Aa_over_Av"]):>+6.1f}% '
            f'{r["edge_steep"]:>6.2f}'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'summary_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    with open(os.path.join(OUT_ROOT, 'sweep_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fns = [r['fn'] for r in rows]
    for ax, (pk, mk, lbl) in zip(axes, [
            ('pred_wa_over_wv', 'meas_wa_over_wv', 'w_a/w_v'),
            ('pred_Aa_over_Av', 'meas_Aa_over_Av', 'A_a²/A_v²'),
        ]):
        ax.plot(fns, [r[pk] for r in rows], 'b-o', label='predicted')
        ax.plot(fns, [r[mk] for r in rows], 'r-s', label='measured (Airy)')
        ax.set(xlabel='F#', ylabel=lbl,
               title=f'{lbl} vs F# (V₀={V0_REF})')
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle('T3 F# sweep (bowl Airy source vs Gaussian theory)',
                 fontsize=12)
    fig.savefig(os.path.join(OUT_ROOT, 'fnumber_plot.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
