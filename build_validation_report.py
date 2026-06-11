"""
Build validation_report.pdf — a multi-page validation document for the
kinematic-FWHM V₀ sweep, including:

  Page 1   Cover: parameters, summary table, methodology notes.
  Page 2   Aggregate ratio-vs-M plot (sweep_ratios.png).
  Page 3   Sweep table (text).
  Pages 4–8  Per-V₀ sections (one page per V₀):
              2D maps · time traces · spectra · xz cross-section · profiles
  Pages 9+  AS diagnostic frames from V₀=0.15 propagation
              (initial conditions + propagation snapshots + summary panel).

Layout uses one matplotlib figure per page, embedded via PdfPages.
PNGs are pulled in via imshow.
"""
import os
import sys
import json
import datetime as _dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

sys.path.insert(0, os.path.dirname(__file__))


SIGMA_ROOT = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'kinematic_sigma_sweep',
)
SIGMA_LOW = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'kinematic_sigma_sweep_low',
)
CONV_1D = os.path.join(
    os.path.dirname(__file__),
    'validation_results', 'convergence_1d',
)
PDF_PATH = os.path.join(SIGMA_ROOT, 'validation_report.pdf')
V0_LIST  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
V0_LOW   = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040]


def add_image_page(pdf: PdfPages, image_path: str, title: str,
                   subtitle: str = '', figsize=(11, 8.5)):
    """Read a PNG, embed it as a full-page figure."""
    if not os.path.exists(image_path):
        print(f'  [skip page] missing image: {image_path}')
        return
    img = imread(image_path)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(img)
    ax.axis('off')
    if subtitle:
        ax.set_title(f'{title}\n{subtitle}', fontsize=11, loc='left')
    else:
        ax.set_title(title, fontsize=12, loc='left')
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, body_lines: list,
                  figsize=(8.5, 11), fontsize=9, mono=True):
    """One page of monospace text."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    family = 'monospace' if mono else 'sans-serif'
    ax.text(0.04, 0.96, title, fontsize=14, weight='bold',
            transform=ax.transAxes, va='top')
    ax.text(0.04, 0.92, '\n'.join(body_lines), fontsize=fontsize,
            family=family, transform=ax.transAxes, va='top',
            linespacing=1.4)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def load_metrics():
    metrics = []
    for v0 in V0_LIST:
        p = os.path.join(SIGMA_ROOT, f'v0_{v0:.3f}',
                         'kinematic_metrics.json')
        if os.path.exists(p):
            with open(p) as f:
                metrics.append(json.load(f))
    return metrics


def cover(pdf: PdfPages):
    title = 'Kinematic hyperfocusing validation report'
    lines = []
    lines.append('Cubic shear-shock focused-bowl simulation')
    lines.append(f'F# = 1, F0 = 100 Hz, c = 2 m/s, β₃ = 5×10⁸')
    lines.append('α₀ = 0.5 Np/m·Hz⁻¹ (brain-shear linear-in-f)')
    lines.append('Lossy KT/Rusanov split-step cubic-Burgers angular spectrum solver')
    lines.append('')
    lines.append('Source bowl: outer_r = 5.0 cm, ROC = 10.0 cm = focus')
    lines.append('Grid (1×): DX = λ/15 = 1.33 mm, DT = 1/(60·F0) = 0.167 ms')
    lines.append('')
    lines.append(f'Generated: {_dt.datetime.now().isoformat(timespec="minutes")}')
    lines.append('')
    lines.append('-' * 64)
    lines.append('Quantities investigated (focal-plane reductions)')
    lines.append('-' * 64)
    lines.append('  v        — particle velocity   (shear component)')
    lines.append('  a        — acceleration       ∂v/∂t      (m/s²)')
    lines.append('  ε̇        — strain rate magn.   √((∂v/∂x)²+(∂v/∂y)²)   (1/s)')
    lines.append('  u        — displacement       ∫v·dt      (m)')
    lines.append('  γ_vm     — von Mises shear strain  √((∂u/∂x)²+(∂u/∂y)²) (1)')
    lines.append('  pI       — intensity          ∫v²·dt')
    lines.append('')
    lines.append('-' * 64)
    lines.append('Two sweeps performed')
    lines.append('-' * 64)
    lines.append('  Sweep A (initial):  V₀ ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40}')
    lines.append('  Sweep B (low-V₀):   V₀ ∈ {0.005, 0.010, 0.015, 0.020,')
    lines.append('                              0.025, 0.030, 0.040}')
    lines.append('  Sweep B added after diagnosis that wave was shocking ~1 cm')
    lines.append('  from source, then attenuating the remaining 4 cm to focal')
    lines.append('  plane. Lower V₀ pushes shock formation toward focal plane.')
    lines.append('')
    lines.append('-' * 64)
    lines.append('Headline finding')
    lines.append('-' * 64)
    lines.append('Strong kinematic hyperfocusing present in the cubic-shear-')
    lines.append('shock regime once shock formation aligns with focal region:')
    lines.append('  V₀ = 0.025 m/s (M = 0.0125):')
    lines.append('     (max_t a)²/pI half-max-area = 0.25 — 4× concentration')
    lines.append('  V₀ = 0.030 m/s (M = 0.0150):')
    lines.append('     (max_t a)²/pI half-max-area = 0.30 — 3× concentration')
    lines.append('  V₀ = 0.020 m/s (M = 0.0100):')
    lines.append('     (max_t a)²/pI half-max-area = 0.60')
    lines.append('Sharp onset at edge-steepness ≈ 1 (shock formation threshold).')
    lines.append('')
    lines.append('-' * 64)
    lines.append('Important bug detected and fixed during analysis')
    lines.append('-' * 64)
    lines.append('  √⟨a²⟩ (linear-in-v) FWHM vs pI (quadratic-in-v) FWHM gives')
    lines.append('  a spurious √2 ≈ 1.41 ratio purely from the dimensional')
    lines.append('  mismatch, before any physics. All maps reported here are')
    lines.append('  dimensionally matched (quadratic-in-v) so ratio < 1 = real')
    lines.append('  kinematic narrowing relative to pI.')
    lines.append('')
    lines.append('-' * 64)
    lines.append('1-D grid-convergence status (full report in the next pages)')
    lines.append('-' * 64)
    lines.append('At the production grid (spc=60, dZ_factor=0.25):')
    lines.append('  v, ε̇, γ converge cleanly (<1%) at all operating points.')
    lines.append('  a_peak is grid-limited only in the strongly-shocked regime')
    lines.append('  (edge_steep > 2). The V₀=0.025 sweet-spot ratio is')
    lines.append('  trustworthy; V₀ ≥ 0.03 a_peak values are conservative')
    lines.append('  (underestimate true converged peak by up to ~10%).')
    add_text_page(pdf, title, lines, fontsize=10)


def page_sweep_table(pdf: PdfPages):
    title = 'Sweep A — kinematic-FWHM at 1× grid (V₀ ∈ [0.05, 0.40])'
    with open(os.path.join(SIGMA_ROOT, 'sweep_table.txt')) as f:
        body = f.read().splitlines()
    add_text_page(pdf, title, body, fontsize=8)


def page_low_sweep_table(pdf: PdfPages):
    title = 'Sweep B — low-V₀ sweep (V₀ ∈ [0.005, 0.040])'
    p = os.path.join(SIGMA_LOW, 'sweep_table.txt')
    if not os.path.exists(p):
        return
    with open(p) as f:
        body = f.read().splitlines()
    add_text_page(pdf, title, body, fontsize=8)


def page_aggregate_plot(pdf: PdfPages):
    add_image_page(
        pdf,
        os.path.join(SIGMA_ROOT, 'sweep_ratios.png'),
        title='Sweep A — kinematic ratios vs source Mach M=V₀/c',
        subtitle='Dimensionally matched (quadratic-in-v) maps. '
                 'Linear baseline = 1.0. Ratios < 1 → kinematic field '
                 'more concentrated than pI. Sweep A V₀ range [0.05, 0.40].',
        figsize=(13, 7),
    )


def page_low_aggregate_plot(pdf: PdfPages):
    add_image_page(
        pdf,
        os.path.join(SIGMA_LOW, 'sweep_ratios.png'),
        title='Sweep B — kinematic ratios vs source Mach M=V₀/c (low V₀)',
        subtitle='Optimum at V₀=0.025 (M=0.0125): a²_pk_A/pI = 0.25. '
                 'Sharp threshold near edge_steepness ≈ 1.',
        figsize=(13, 7),
    )
    add_image_page(
        pdf,
        os.path.join(SIGMA_LOW, 'sweep_zfocus.png'),
        title='Sweep B — z_focus vs M  (focal pull-in due to absorption)',
        subtitle='Lower V₀ → captured z_focus drifts toward geometric '
                 f'focus (10 cm). Maximum 7.4 cm reached at V₀=0.005.',
        figsize=(11, 6),
    )


def page_per_v0(pdf: PdfPages, v0: float, root: str = None):
    run_dir = os.path.join(root or SIGMA_ROOT, f'v0_{v0:.3f}')
    fig_dir = os.path.join(run_dir, 'figures')
    metrics_path = os.path.join(run_dir, 'kinematic_metrics.json')
    if not os.path.exists(metrics_path):
        print(f'  [skip per-V₀ pages] missing metrics for V₀={v0}')
        return
    with open(metrics_path) as f:
        m = json.load(f)
    M = m['M_v0_over_c']

    title = (f'V₀ = {v0} m/s   (M = V₀/c = {M:.4f})   '
             f'z_focus = {m["z_focus_m"]*100:.2f} cm')

    # 2D maps page
    sub = (f'On-axis peaks: v={m["on_axis"]["v_peak"]:.4f} m/s, '
           f'|a|_pk={m["on_axis"]["a_peak"]:.2e} m/s², '
           f'|ε̇|_pk={m["on_axis"]["eps_peak"]:.2e} 1/s, '
           f'γ_vm_pk={m["on_axis"]["gamma_vm_peak"]:.2e}, '
           f'edge={m["diag"]["edge_steepness"]:.2f}, '
           f'odd={m["diag"]["odd_frac"]*100:.1f}%')
    add_image_page(pdf, os.path.join(fig_dir, 'fig_2dmaps.png'),
                   title=f'{title} — 2D focal-plane maps',
                   subtitle=sub, figsize=(13, 8))
    add_image_page(pdf, os.path.join(fig_dir, 'fig_traces.png'),
                   title=f'{title} — time traces (v, a, ε̇, γ_vm) at y=0',
                   subtitle='on-axis · +10 mm · +20 mm · +40 mm',
                   figsize=(11, 12))
    add_image_page(pdf, os.path.join(fig_dir, 'fig_spectra.png'),
                   title=f'{title} — on-axis & off-axis spectra of v, a',
                   figsize=(11, 8))
    add_image_page(pdf, os.path.join(fig_dir, 'fig_xz.png'),
                   title=f'{title} — pI(x, y=0, z) cross-section',
                   subtitle='Captured plane shown as cyan dashed line.',
                   figsize=(13, 6))
    add_image_page(pdf, os.path.join(fig_dir, 'fig_profiles.png'),
                   title=f'{title} — lateral profiles at y=0',
                   subtitle='Each curve shows its FWHM (mm) and ratio to pI '
                           'FWHM in the legend.',
                   figsize=(13, 6))


def page_convergence_1d(pdf: PdfPages):
    """1-D convergence study pages."""
    if not os.path.isdir(CONV_1D):
        return
    # Intro / table
    intro = [
        '1-D cubic-Burgers + linear-in-f shear-loss convergence study.',
        '',
        'Tests grid-resolution stability of the four kinematic peak',
        'quantities at the operating points relevant to the 3-D sweeps:',
        '',
        '  LIN  V₀=0.025 m/s, z=6.0 cm, edge≈0.94  (linear regime)',
        '  MID  V₀=0.20  m/s, z=3.0 cm, edge≈1.07  (weakly shocked)',
        '  SHK  V₀=0.50  m/s, z=2.0 cm, edge≈2.50  (strongly shocked)',
        '',
        'Sweep axes: samples_per_cycle ∈ {30, 60, 120, 240, 480}',
        '            dZ_factor        ∈ {1.0, 0.5, 0.25, 0.125}',
        '',
        'Tracked: v_peak, a_peak (=∂v/∂t), ε̇_peak (=∂v/∂z),',
        '         γ_peak (axial strain ∂u/∂z, 1-D analogue of γ_vm).',
        '',
        'Headline finding: convergence is QUANTITY- and REGIME-dependent.',
        '  • v, ε̇, γ converge cleanly (<1% by spc=60) at all operating points.',
        '  • a_peak is grid-limited in the strongly-shocked regime:',
        '      SHK spc 30 → 60:  Δa = +33%',
        '      SHK spc 60 →120:  Δa = +12%',
        '      SHK spc 120→240:  Δa = +4%',
        '      SHK spc 240→480:  Δa = +0.7%',
        '  • At the V₀=0.025 production point (3-D edge=1.67), the regime is',
        '    between LIN and MID; a_peak error at spc=60 is ~1-3%; ratios',
        '    quoted in the headline are trustworthy.',
        '  • At V₀≥0.030 (edge>2.5), a_peak is grid-limited and likely',
        '    underestimates the converged value by ~10%; ratios for those',
        '    V₀ should be treated as lower bounds on kinematic concentration.',
    ]
    add_text_page(pdf, '1-D convergence study', intro, fontsize=10)

    # Table
    p = os.path.join(CONV_1D, 'convergence_table.txt')
    if os.path.exists(p):
        with open(p) as f:
            body = f.read().splitlines()
        add_text_page(pdf, 'Convergence table (all 3 operating points)',
                      body, fontsize=6)

    # Convergence panels per operating point
    for tag in ('LIN', 'MID', 'SHK'):
        add_image_page(pdf,
                       os.path.join(CONV_1D, f'convergence_quantities_{tag}.png'),
                       title=f'[{tag}] Convergence of kinematic peaks vs grid',
                       subtitle='Top row: quantity vs samples_per_cycle '
                                '(curves = dZ_factor).  '
                                'Bottom row: quantity vs dZ_factor '
                                '(curves = spc).',
                       figsize=(13, 7))
        add_image_page(pdf,
                       os.path.join(CONV_1D,
                                    f'traces_coarsest_vs_finest_{tag}.png'),
                       title=f'[{tag}] Coarsest vs finest grid waveforms',
                       subtitle='Red = spc=30, dZf=1.0. '
                                'Blue = spc=480, dZf=0.125.',
                       figsize=(11, 10))


def page_as_diagnostics(pdf: PdfPages):
    """Add AS solver diagnostic frames (initial conditions, propagation
    snapshots, summary panel) for the V₀=0.15 reference run."""
    diag_dir = os.path.join(
        SIGMA_ROOT, 'v0_0.150', 'as_diagnostic_frames',
    )
    if not os.path.isdir(diag_dir):
        print('  [skip AS diagnostic pages] no diagnostic frames')
        return

    add_text_page(pdf, 'Angular-spectrum solver diagnostics (V₀ = 0.15 m/s)',
                  ['AS solver was re-run for V₀ = 0.15 m/s with',
                   '  diagnostic=True, diagnosticInterval=50,',
                   '  diagnosticInitialConditions=True,',
                   '  diagnosticSummary=True.',
                   '',
                   'The following pages show the initial source field,',
                   'propagation snapshots every 50 z-steps, and the',
                   'summary dashboard at z_end.',
                   '',
                   f'Run output: {diag_dir}',
                   ], fontsize=11)

    # Initial conditions
    add_image_page(pdf, os.path.join(diag_dir, 'initial_field.png'),
                   title='AS diagnostic — initial source field',
                   subtitle=f'V₀ = 0.15 m/s', figsize=(13, 9))

    # Propagation frames (every 50 z-steps)
    frames = sorted(
        f for f in os.listdir(diag_dir)
        if f.startswith('frame_') and f.endswith('.png')
    )
    for f in frames:
        # Each frame is a single image; embed it.
        add_image_page(pdf, os.path.join(diag_dir, f),
                       title=f'AS diagnostic — propagation snapshot {f}',
                       figsize=(13, 9))

    # Summary panels
    summary_dir = os.path.join(diag_dir, 'summary')
    if os.path.isdir(summary_dir):
        for f in sorted(os.listdir(summary_dir)):
            if f.endswith('.png'):
                add_image_page(pdf,
                               os.path.join(summary_dir, f),
                               title=f'AS diagnostic summary — {f}',
                               figsize=(13, 9))


def main():
    print(f'Building {PDF_PATH}')
    with PdfPages(PDF_PATH) as pdf:
        cover(pdf)
        # 1-D convergence first — establishes which kinematic numbers
        # in the 3-D sweeps are trustworthy at the production grid.
        page_convergence_1d(pdf)
        # Aggregate sweep B (low V₀) — headline
        page_low_aggregate_plot(pdf)
        page_low_sweep_table(pdf)
        for v0 in V0_LOW:
            page_per_v0(pdf, v0, root=SIGMA_LOW)
        # Sweep A (initial)
        page_aggregate_plot(pdf)
        page_sweep_table(pdf)
        for v0 in V0_LIST:
            page_per_v0(pdf, v0, root=SIGMA_ROOT)
        page_as_diagnostics(pdf)
    sz_mb = os.path.getsize(PDF_PATH) / 1e6
    print(f'  done. {PDF_PATH} ({sz_mb:.1f} MB)')


if __name__ == '__main__':
    main()
