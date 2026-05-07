# Angular Spectrum Method for Nonlinear Ultrasound Propagation

A three-dimensional nonlinear acoustic propagation solver based on the modified angular spectrum method (ASM), with extensions for heterogeneous tissue and immersed curved-source transducers.

## Features

- **Split-step propagation**: Strang splitting of diffraction (FFT-based angular spectrum), nonlinear distortion (Burgers equation), and frequency-dependent attenuation with Kramers-Kronig dispersion
- **Nonlinear flux schemes**: First-order Rusanov and second-order Kurganov-Tadmor (KT) with MUSCL reconstruction, SSP-RK2, and adaptive CFL sub-cycling
- **Heterogeneous propagation**: Phase-and-amplitude screens derived from CT data for transcranial ultrasound through skull bone
- **Immersed bowl sources**: Plane-by-plane source injection for spherical-bowl transducers (e.g., TIPS annular phased array)
- **Enhanced absorbing boundaries**: Wendland C2 taper, frequency-weighted damping, and Engquist-Majda super-absorbing correction (2.4x reflection reduction)
- **Intensity-loss tracking**: Spatially resolved absorbed-energy maps for radiation-force and thermal dose calculations
- **GPU acceleration**: JAX-based implementation with JIT compilation; opt-in `useGPUReductions` keeps per-step reductions on the device, and `JAX_ENABLE_X64=0` switches to FP32 for parameter sweeps on workstation GPUs
- **Diagnostic imagery**: Runtime 3×3 dashboards every N march steps and an end-of-run summary report (Isppa / MI / PNP / PPP cross-sections, harmonics-vs-z, boundary-leakage and stability checks, `metrics.json`)
- **Pre-flight sanity report**: Single-page LaTeX/PDF report with grid / pulse / material / numerics tables and an IC panels figure, plus per-screen phase + amplitude maps when phase screens are configured
- **Sub-sample TOF extraction**: Optional matched-filter-parabolic or envelope-based time-of-flight tracking per (x, y) at every z step

## Requirements

- Python 3.10+
- JAX (with GPU support recommended)
- NumPy
- SciPy (for `tof_extraction`)
- Matplotlib (for diagnostics + pre-flight)
- h5py (for transcranial validation)
- pdflatex (optional; pre-flight report falls back to `.tex` only when missing)

## Quick Start

```python
from angular_spectrum_solver import SolverParams, angular_spectrum_solve

# Define parameters
params = SolverParams(
    dX=3.08e-4, dY=3.08e-4, dT=4e-8,
    c0=1540.0, rho0=1000.0,
    beta=3.5, alpha0=0.5, f0=1e6,
    propDist=0.065,
    useSplitStep=True,
    fluxScheme='kt',
    boundaryProfile='wendland',
    useFreqWeightedBoundary=True,
    useSuperAbsorbing=True,
)

# initial_field: ndarray of shape (nX, nY, nT)
field, pnp, ppp, pI, pIloss, zaxis, pax = angular_spectrum_solve(
    initial_field, params
)
```

### Bowl Transducer Source

```python
from angular_spectrum_solver import make_bowl_source_planes

source_planes, bowl_depth = make_bowl_source_planes(
    xaxis, yaxis, taxis,
    f0=1e6, c0=1540.0, p0=400e3,
    radius=46e-3, roc=80e-3, dZ=dZ,
    focus=50e-3, inner_radius=20.5e-3,
    n_elements=8,
)

# First slice is the initial field; remaining are injected during propagation
initial_field = source_planes[0][1]
params.sourcePlanes = source_planes[1:]
```

### Pre-flight sanity report

A single-page LaTeX/PDF summary of the upcoming run (grid resolution, pulse
spectrum, material constants, derived quantities like Goldberg number and
shock-formation length, plus automatic warnings for under-resolved grids
or pulses touching the time-window edge):

```python
params = SolverParams(..., preflight=True,
                      preflightDir='./preflight',
                      preflightScenario='Bowl R=46 mm, ROC=80 mm')
angular_spectrum_solve(field, params)
# → ./preflight/preflight.pdf  (+ preflight_panels.png, preflight.tex)
```

If `phaseScreens` is set, a second page with per-screen phase maps,
histograms, and amplitude transmission maps is appended automatically.
Set `preflightDir = diagnosticDir` to co-locate the report with runtime
diagnostic frames.

You can also call it directly without launching a solve:

```python
from preflight import preflight_report
preflight_report(initial_field, params, output_dir='./preflight',
                 scenario='Quick check')
```

### Diagnostic imagery

Runtime 3×3 PNG dashboards every N march steps plus an end-of-run summary
report under `diagnosticDir`:

```python
params = SolverParams(..., diagnostic=True,
                      diagnosticInterval=5,
                      diagnosticDir='./run42_frames')
angular_spectrum_solve(field, params)
# → ./run42_frames/frame_NNNN.png    (per-step dashboards)
# → ./run42_frames/summary/          (Isppa / MI / PNP / PPP cross-sections,
#                                     boundaries.png, stability.png,
#                                     harmonics.png, metrics.json)
```

`metrics.json` consolidates max Isppa, MI, peak focal location, boundary
leakage fractions, stability margin, and runtime — useful as a single
grep target across parameter sweeps.

### Time-of-flight extraction

Per-(x, y) TOF tracked at every z step, useful for skull-induced delay
analysis. Sub-sample-accurate matched-filter-parabolic mode requires a
reference pulse and its envelope-peak time:

```python
out = angular_spectrum_solve(field, params,
                             tof_ref_trace=ref_pulse,
                             tof_t_ref_peak_s=t_peak_ref,
                             taxis=taxis)
*_, pax, tof = out  # 8-element return when TOF is enabled
```

Or pass `tof_env_ratio=1.0` for an envelope-based fallback (no reference
needed).

## Validation

The solver is validated through:

1. **Diffraction**: Baffled-piston and focused-piston benchmarks against Rayleigh-Fresnel and Airy analytical solutions
2. **Attenuation**: Power-law attenuation and Kramers-Kronig dispersion across 0.5-10 MHz
3. **Nonlinearity**: Riemann problem and shock-forming Burgers benchmarks
4. **Boundary treatments**: Cumulative 2.4x reflection reduction
5. **Convergence**: Strang+KT achieves second-order convergence rate
6. **Transcranial**: 1.1% RMS focal-plane agreement with FDTD through *ex vivo* human skull
7. **Bowl transducer**: 2.3% focal-depth agreement with FDTD for TIPS annular array

Run the validation scripts:

```bash
python validate_analytical.py    # Diffraction and nonlinear benchmarks
python validate_solver.py        # Convergence and flux comparison
python validate_boundaries.py   # Boundary treatment tests
python validate_transcranial.py  # Transcranial FDTD comparison
python validate_bowl.py          # Bowl transducer FDTD comparison
```

## Reference

G.F. Pinton, "An Angular Spectrum Method for Nonlinear Propagation in Heterogeneous Tissue with Immersed Sources for Transcranial and Therapeutic Ultrasound," 2026.

## License

This project is licensed under the GNU Lesser General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
