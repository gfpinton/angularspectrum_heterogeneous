"""
T3 F#=1 with the immersed (plane-by-plane) bowl source.

Replaces the flat-projection source (`make_bowl_source`) with the
true curved-bowl source (`make_bowl_source_planes`).  This injects
the bowl's curved surface as a stack of z-slices rather than
collapsing it onto z=0.

For F#=1 with our F=10 cm and aperture a=5 cm, the bowl cap depth is
   ROC − √(ROC² − a²) = 0.10 − √(0.0075) = 13.4 mm
which is 13% of the focusing distance — large enough that the
flat-projection should differ measurably from the true immersed bowl.

Output: validation_results/theory_validation/T3_fnumber/fn_1.0_immersed/
"""
import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'cm'})

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    SolverParams, angular_spectrum_solve,
    make_bowl_source, make_bowl_source_planes,
)
from kinematic_analyzer import (
    kinematic_fields, lateral_2d_maps, area_above_threshold, fwhm,
)
from theory_predictor import on_axis_harmonic_amplitudes, predict_ratios
from T1_validate_gaussian import (
    F0, C0, RHO0, BETA3, ALPHA0, Y_ABS, NCYCLES,
    LAM, FOCUS, DOMAIN, DX, DT,
)

V0_REF = 0.025
FN     = 1.0
OUTER_R = FOCUS / (2.0 * FN)        # 5 cm

OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'theory_validation', 'T3_fnumber',
    'fn_1.0_immersed',
)
os.makedirs(OUT_DIR, exist_ok=True)


def grid():
    nX = int(np.ceil(DOMAIN / DX));  nX += (nX % 2 == 0)
    xaxis = (np.arange(nX) - nX // 2) * DX
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT
    return xaxis, taxis, nX, nT


def run_immersed(v0, linear=False, cache=None):
    if cache is not None and os.path.exists(cache):
        return np.load(cache)
    xaxis, taxis, nX, nT = grid()

    # Use the same dZmin as the AS solver internal will start with —
    # the slice thickness for the bowl-source-planes must be coordinated
    # with the solver's z-step.  Use 1/10 of LAM as a safe slice size
    # (bowl depth ≈ 13 mm = LAM/1.5, so we want sub-LAM slicing).
    dZ_slice = LAM / 10.0    # 2 mm slice thickness → ~7 slices over the 13.4 mm bowl cap

    print(f'  Building immersed bowl (V₀={v0}, F#={FN}, '
          f'a={OUTER_R*100:.1f} cm, dZ_slice={dZ_slice*1e3:.2f} mm)...')
    source_planes, bowl_depth = make_bowl_source_planes(
        xaxis, xaxis, taxis, F0, C0, v0,
        radius=OUTER_R, roc=FOCUS, dZ=dZ_slice,
        focus=FOCUS, ncycles=NCYCLES, dur=4,
    )
    print(f'  Bowl depth: {bowl_depth*1e3:.2f} mm, '
          f'{len(source_planes)} z-slices')

    # First plane = initial field; remaining = sourcePlanes for injection
    initial_field = source_planes[0][1]
    inject_planes = source_planes[1:]

    if linear:
        beta3_use, alpha0_use, atten_loss = 0.0, 0.0, False
        N3 = 0.0
    else:
        beta3_use, alpha0_use, atten_loss = BETA3, ALPHA0, True
        N3 = BETA3 / (3 * C0**5 * RHO0**2)
    k0 = 2 * np.pi * F0 / C0
    focus_gain = k0 * OUTER_R**2 / (2 * FOCUS)
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
        sourcePlanes=inject_planes,
    )
    t0 = time.time()
    field, _, _, pI, _, zaxis, _ = angular_spectrum_solve(
        initial_field, sp, verbose=False)
    wall = time.time() - t0
    print(f'  Immersed-bowl solve: {wall:.0f}s, {len(zaxis)} z-steps')

    if cache is not None:
        np.savez(cache,
                 field_focal=field.astype(np.float32),
                 pI=pI.astype(np.float32),
                 zaxis=zaxis.astype(np.float32),
                 xaxis=xaxis.astype(np.float32),
                 v0=v0, linear=linear, wall_s=wall,
                 bowl_depth=bowl_depth,
                 n_slices=len(source_planes))
        return np.load(cache)
    return None


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
    print(f'T3 F#={FN} with IMMERSED bowl source (V₀={V0_REF})')
    print(f'Domain {DOMAIN*100:.0f}×{DOMAIN*100:.0f} cm, '
          f'F={FOCUS*100:.0f} cm, a={OUTER_R*100:.1f} cm')

    # ---- Linear reference (immersed) ----
    print('\nLinear reference (immersed bowl):')
    d_lin = run_immersed(V0_REF, linear=True,
                         cache=os.path.join(OUT_DIR, 'linear.npz'))
    m_lin = measure(d_lin)
    print(f'  Linear FWHM = {m_lin["fwhm_v_mm"]:.3f} mm '
          f'(v_pk={m_lin["on_axis_v_peak"]:.3e})')

    # ---- Nonlinear cubic + lossy (immersed) ----
    print('\nNonlinear cubic+lossy (immersed bowl):')
    d_nl = run_immersed(V0_REF, linear=False,
                        cache=os.path.join(OUT_DIR, 'nonlinear.npz'))
    m_nl = measure(d_nl)
    print(f'  Nonlinear FWHM = {m_nl["fwhm_v_mm"]:.3f} mm, '
          f'v_pk={m_nl["on_axis_v_peak"]:.3e}, '
          f'a_pk={m_nl["on_axis_a_peak"]:.3e}')

    # Predicted vs measured kinematic ratios
    Vn2 = np.array(m_nl['Vn2'])
    w0 = m_lin['fwhm_v_mm'] / np.sqrt(2 * np.log(2)) * 1e-3
    pred = predict_ratios(Vn2, np.arange(1, len(Vn2)+1, dtype=float), w0=w0)
    meas_wa_wv = m_nl['fwhm_a_mm'] / max(m_nl['fwhm_v_mm'], 1e-30)
    meas_Aa_Av = m_nl['area_a_mm2'] / max(m_nl['area_v_mm2'], 1e-30)
    print(f'\n  ⟨n⟩_v       = {pred["n_v"]:.4f}')
    print(f'  ⟨n⟩_a²      = {pred["n_a2"]:.4f}')
    print(f'  pred wa/wv  = {pred["wa_over_wv"]:.4f}')
    print(f'  meas wa/wv  = {meas_wa_wv:.4f}')
    print(f'  err: {(meas_wa_wv-pred["wa_over_wv"])/pred["wa_over_wv"]*100:+.2f}%')
    print(f'  pred Aa/Av  = {pred["Aa2_over_Av2"]:.4f}')
    print(f'  meas Aa/Av  = {meas_Aa_Av:.4f}')
    print(f'  err: {(meas_Aa_Av-pred["Aa2_over_Av2"])/pred["Aa2_over_Av2"]*100:+.2f}%')

    # ---- Comparison with flat-source T3 F#=1 ----
    flat_path = os.path.join(
        os.path.dirname(__file__),
        'validation_results', 'theory_validation',
        'T3_fnumber', 'fn_1.0', 'nonlinear.npz')
    d_flat = np.load(flat_path)
    m_flat = measure(d_flat)
    d_flat_lin = np.load(os.path.join(
        os.path.dirname(__file__),
        'validation_results', 'theory_validation',
        'T3_fnumber', 'fn_1.0', 'linear.npz'))
    m_flat_lin = measure(d_flat_lin)
    Vn2_flat = np.array(m_flat['Vn2'])
    w0_flat = m_flat_lin['fwhm_v_mm'] / np.sqrt(2 * np.log(2)) * 1e-3
    pred_flat = predict_ratios(Vn2_flat,
                               np.arange(1, len(Vn2_flat)+1, dtype=float),
                               w0=w0_flat)
    meas_wa_wv_flat = m_flat['fwhm_a_mm'] / max(m_flat['fwhm_v_mm'], 1e-30)
    meas_Aa_Av_flat = m_flat['area_a_mm2'] / max(m_flat['area_v_mm2'], 1e-30)

    # ---- Side-by-side comparison ----
    lines = []
    lines.append('T3 F#=1.0 — flat-projection vs immersed-bowl source')
    lines.append('=' * 80)
    lines.append(f'V₀ = {V0_REF} m/s,  F = {FOCUS*100:.0f} cm,  '
                 f'aperture a = {OUTER_R*100:.1f} cm,  '
                 f'bowl-cap depth = {float(d_nl["bowl_depth"])*1e3:.2f} mm')
    lines.append(f'Bowl cap depth / focal length = '
                 f'{float(d_nl["bowl_depth"])/FOCUS:.3f}')
    lines.append('')
    lines.append(f'{"Quantity":<28} {"FLAT":>13} {"IMMERSED":>13} '
                 f'{"flat→imm Δ":>12}')
    lines.append('-' * 80)
    def fmt(a, b, fmtstr='8.4f'):
        d = (b - a) / max(abs(a), 1e-30) * 100
        return (f'{a:{fmtstr}}'.rjust(13)
                + f'{b:{fmtstr}}'.rjust(13)
                + f'{d:+.1f}%'.rjust(12))
    lines.append(f'{"Linear FWHM (mm)":<28}' + fmt(
        m_flat_lin['fwhm_v_mm'], m_lin['fwhm_v_mm']))
    lines.append(f'{"Linear w_0 (mm)":<28}' + fmt(
        w0_flat*1e3, w0*1e3))
    lines.append(f'{"Nonlinear FWHM_v (mm)":<28}' + fmt(
        m_flat['fwhm_v_mm'], m_nl['fwhm_v_mm']))
    lines.append(f'{"Nonlinear FWHM_a (mm)":<28}' + fmt(
        m_flat['fwhm_a_mm'], m_nl['fwhm_a_mm']))
    lines.append(f'{"on-axis v_peak (m/s)":<28}' + fmt(
        m_flat['on_axis_v_peak'], m_nl['on_axis_v_peak']))
    lines.append(f'{"on-axis a_peak (m/s²)":<28}' + fmt(
        m_flat['on_axis_a_peak'], m_nl['on_axis_a_peak'], '8.3e'))
    lines.append(f'{"⟨n⟩_v":<28}' + fmt(
        pred_flat['n_v'], pred['n_v']))
    lines.append(f'{"⟨n⟩_a²":<28}' + fmt(
        pred_flat['n_a2'], pred['n_a2']))
    lines.append(f'{"pred wa/wv":<28}' + fmt(
        pred_flat['wa_over_wv'], pred['wa_over_wv']))
    lines.append(f'{"meas wa/wv":<28}' + fmt(
        meas_wa_wv_flat, meas_wa_wv))
    lines.append(f'{"pred Aa/Av":<28}' + fmt(
        pred_flat['Aa2_over_Av2'], pred['Aa2_over_Av2']))
    lines.append(f'{"meas Aa/Av":<28}' + fmt(
        meas_Aa_Av_flat, meas_Aa_Av))
    lines.append(f'{"r_ring^(ε̇) meas (mm)":<28}' + fmt(
        m_flat['r_ring_eps_mm'], m_nl['r_ring_eps_mm']))
    lines.append(f'{"r_ring^(γ) meas (mm)":<28}' + fmt(
        m_flat['r_ring_gam_mm'], m_nl['r_ring_gam_mm']))
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_DIR, 'comparison_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    # ---- Spectra side-by-side ----
    Vn2_flat = np.array(m_flat['Vn2']) / max(np.array(m_flat['Vn2']).sum(), 1e-30)
    Vn2_imm  = np.array(m_nl['Vn2'])  / max(np.array(m_nl['Vn2']).sum(),  1e-30)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    n_axis = np.arange(1, len(Vn2_flat)+1)
    ax.semilogy(n_axis, Vn2_flat, 'b-o', lw=1.4, markersize=6,
                label='flat-projection bowl')
    ax.semilogy(n_axis, Vn2_imm,  'r-s', lw=1.4, markersize=6,
                markerfacecolor='none', label='immersed (plane-by-plane) bowl')
    ax.set(xlabel='Harmonic n', ylabel='$|V_n|^2$ (normalised)',
           title=f'T3 F#=1 on-axis focal-plane spectrum: '
                 f'flat vs immersed bowl ($V_0={V0_REF}$)',
           xlim=[0.5, 12], ylim=[1e-6, 2])
    ax.grid(alpha=0.3, which='both'); ax.legend()
    fig.savefig(os.path.join(OUT_DIR, 'spectra_comparison.png'), dpi=170)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(dict(
            immersed=dict(linear=m_lin, nonlinear=m_nl, predicted=pred,
                          meas_wa_over_wv=meas_wa_wv,
                          meas_Aa_over_Av=meas_Aa_Av),
            flat=dict(linear=m_flat_lin, nonlinear=m_flat,
                      predicted=pred_flat,
                      meas_wa_over_wv=meas_wa_wv_flat,
                      meas_Aa_over_Av=meas_Aa_Av_flat),
            bowl_depth_mm=float(d_nl['bowl_depth'])*1e3,
            n_slices=int(d_nl['n_slices']),
        ), f, indent=2)


if __name__ == '__main__':
    main()
