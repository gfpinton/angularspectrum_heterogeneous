"""
T2 — Grid convergence on the Gaussian-source validation.

For one V₀ at the kinematic sweet spot, repeat the T1 test at three grid
refinements: 1×, 1.5×, 2×.  Verifies that the predicted-vs-measured
agreement at the 1× grid is the converged answer, not a grid artefact.

Refinement factor R rescales:
  DX → DX / R   (lateral grid spacing)
  DT → DT / R   (time-step)
  dZ_floor → dZ_floor / R   (axial step floor)

At R=1: nX≈91, nT≈601, ~5 min wall.
At R=1.5: nX≈137, nT≈900, ~13 min.
At R=2:   nX≈181, nT≈1201, ~50 min.

Outputs: validation_results/theory_validation/T2_grid_gaussian/
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
from make_gaussian_source import make_gaussian_source
from kinematic_analyzer import (
    kinematic_fields, lateral_2d_maps, area_above_threshold, fwhm,
)
from theory_predictor import on_axis_harmonic_amplitudes, predict_ratios
from T1_validate_gaussian import (
    F0, C0, RHO0, BETA3, ALPHA0, Y_ABS, NCYCLES,
    LAM, WAIST_SRC, FOCUS,
    DOMAIN,
)

V0_REF = 0.025
R_LIST = [1.0, 1.5, 2.0]

OUT_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T2_grid_gaussian',
)
os.makedirs(OUT_ROOT, exist_ok=True)


def run_at_R(v0, R, linear, cache):
    if os.path.exists(cache):
        return np.load(cache)
    DX = LAM / (15.0 * R)
    DT = 1.0 / (60.0 * F0 * R)
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    init = make_gaussian_source(
        xaxis, xaxis, taxis, F0, C0, v0,
        waist_source=WAIST_SRC, focus=FOCUS, ncycles=NCYCLES,
    )
    if linear:
        beta3_use, alpha0_use, atten_loss = 0.0, 0.0, False
        N3 = 0.0
    else:
        beta3_use, alpha0_use, atten_loss = BETA3, ALPHA0, True
        N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * WAIST_SRC**2 / FOCUS
    expected_peak = max(v0 * focus_gain, v0)
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
             v0=v0, linear=linear, wall_s=wall, R=R,
             DX=DX, DT=DT)
    return np.load(cache)


def measure(d_npz):
    field = d_npz['field_focal']
    xaxis = d_npz['xaxis']
    DX = float(d_npz['DX']); DT = float(d_npz['DT'])
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
        DX=DX, DT=DT,
    )


def main():
    print(f'T2 — Grid convergence at V₀ = {V0_REF} m/s, R ∈ {R_LIST}')
    rows = []
    for R in R_LIST:
        print(f'\n=== R = {R} ===')
        out_dir = os.path.join(OUT_ROOT, f'R{R:.2f}')
        os.makedirs(out_dir, exist_ok=True)
        d_lin = run_at_R(V0_REF, R, linear=True,
                         cache=os.path.join(out_dir, 'linear.npz'))
        m_lin = measure(d_lin)
        d_nl = run_at_R(V0_REF, R, linear=False,
                        cache=os.path.join(out_dir, 'nonlinear.npz'))
        m_nl = measure(d_nl)

        Vn2 = np.array(m_nl['Vn2'])
        w0_lin = m_lin['fwhm_v_mm'] / np.sqrt(2*np.log(2)) * 1e-3
        pred = predict_ratios(Vn2, np.arange(1, len(Vn2)+1, dtype=float),
                              w0=w0_lin)
        meas_wa_wv = m_nl['fwhm_a_mm'] / max(m_nl['fwhm_v_mm'], 1e-30)
        meas_Aa_Av = m_nl['area_a_mm2'] / max(m_nl['area_v_mm2'], 1e-30)
        print(f'  pI FWHM lin = {m_lin["fwhm_v_mm"]:.3f} mm '
              f'(w_0 = {w0_lin*1e3:.3f} mm)')
        print(f'  ⟨n⟩_v = {pred["n_v"]:.4f}, ⟨n⟩_a² = {pred["n_a2"]:.4f}')
        print(f'  Predicted: w_a/w_v={pred["wa_over_wv"]:.4f}, '
              f'A_a²/A_v²={pred["Aa2_over_Av2"]:.4f}, '
              f'r_ε̇={pred["r_ring_eps"]*1e3:.3f}mm, '
              f'r_γ={pred["r_ring_gam"]*1e3:.3f}mm')
        print(f'  Measured:  w_a/w_v={meas_wa_wv:.4f}, '
              f'A_a²/A_v²={meas_Aa_Av:.4f}, '
              f'r_ε̇={m_nl["r_ring_eps_mm"]:.3f}mm, '
              f'r_γ={m_nl["r_ring_gam_mm"]:.3f}mm')
        rows.append(dict(
            R=R, DX=m_nl['DX'], DT=m_nl['DT'],
            n_v=pred['n_v'], n_a2=pred['n_a2'],
            wv_lin_fwhm_mm=m_lin['fwhm_v_mm'],
            pred_wa_over_wv=pred['wa_over_wv'],
            meas_wa_over_wv=meas_wa_wv,
            pred_Aa_over_Av=pred['Aa2_over_Av2'],
            meas_Aa_over_Av=meas_Aa_Av,
            pred_r_eps_mm=pred['r_ring_eps']*1e3,
            meas_r_eps_mm=m_nl['r_ring_eps_mm'],
            pred_r_gam_mm=pred['r_ring_gam']*1e3,
            meas_r_gam_mm=m_nl['r_ring_gam_mm'],
        ))

    # Aggregate
    lines = [f'T2 — Grid convergence on Gaussian source, V₀ = {V0_REF} m/s',
             '=' * 100]
    lines.append(f'{"R":>5} {"DX(mm)":>8} {"DT(μs)":>8} {"w_0(mm)":>8} '
                 f'{"⟨n⟩_v":>7} {"⟨n⟩_a²":>7} '
                 f'{"wa/wv pred":>11} {"meas":>7} {"%err":>7} '
                 f'{"Aa/Av pred":>11} {"meas":>7} {"%err":>7}')
    lines.append('-' * 105)
    def pct(p, m): return (m-p)/max(abs(p),1e-30)*100
    for r in rows:
        lines.append(
            f'{r["R"]:>5.2f} {r["DX"]*1e3:>8.3f} {r["DT"]*1e6:>8.2f} '
            f'{r["wv_lin_fwhm_mm"]/np.sqrt(2*np.log(2)):>8.3f} '
            f'{r["n_v"]:>7.4f} {r["n_a2"]:>7.4f} '
            f'{r["pred_wa_over_wv"]:>11.4f} {r["meas_wa_over_wv"]:>7.4f} '
            f'{pct(r["pred_wa_over_wv"], r["meas_wa_over_wv"]):>+6.1f}% '
            f'{r["pred_Aa_over_Av"]:>11.4f} {r["meas_Aa_over_Av"]:>7.4f} '
            f'{pct(r["pred_Aa_over_Av"], r["meas_Aa_over_Av"]):>+6.1f}%'
        )
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_ROOT, 'convergence_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    with open(os.path.join(OUT_ROOT, 'convergence_runs.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=float)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    Rs = [r['R'] for r in rows]
    for ax, (pk, mk, lbl) in zip(axes, [
            ('pred_wa_over_wv', 'meas_wa_over_wv', 'w_a/w_v'),
            ('pred_Aa_over_Av', 'meas_Aa_over_Av', 'A_a²/A_v²'),
        ]):
        p = [r[pk] for r in rows]; m = [r[mk] for r in rows]
        ax.plot(Rs, p, 'b-o', label='predicted')
        ax.plot(Rs, m, 'r-s', label='measured')
        ax.set(xlabel='Refinement R', ylabel=lbl,
               title=f'{lbl} vs grid refinement')
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f'T2 grid convergence at V₀ = {V0_REF}', fontsize=12)
    fig.savefig(os.path.join(OUT_ROOT, 'convergence_plot.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
