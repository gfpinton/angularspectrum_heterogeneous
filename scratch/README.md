# scratch/

One-off debug, tuning, and exploratory analysis scripts that are not
part of the publication pipeline. Retained for provenance but not
actively maintained, and not tracked in git.

These were written against specific cached runs under
`validation_results/` and assume those outputs exist. Each script adds
the repo root to `sys.path`, so they can be run from here directly,
e.g. `python scratch/debug_area_ratio.py`.

| Script | Purpose |
|---|---|
| `reaggregate_sweep.py` | Re-runs aggregation on cached kinematic sweep data |
| `refine_v0_endpoints.py` | One-off re-run of two errant V0 endpoints |
| `rerun_shocked_with_diagnostics.py` | Re-ran one shocked config with diagnostics on |
| `figs_per_v0.py` | Per-V0 figures, superseded by `build_publication_figures.py` |
| `analyze_cubic_lossy_beamplots.py` | Exploratory beamplot generation |
| `analyze_cubic_lossy_focal_Q.py` | Exploratory focal-Q reconstruction |
| `analyze_cubic_lossy_harmonic_Q.py` | Exploratory harmonic-Q reconstruction |
| `analyze_kinematic_1x.py` | One-off kinematic dashboard for sweep validation |
| `T1_central_lobe_diagnostic.py` | Per-harmonic central-lobe diagnostic for T1 |
| `sweep_kinematic_v0_low.py` | Low-amplitude variant of the kinematic sweep |
| `debug_area_ratio.py` | Area-ratio debugging against cached T1 runs |
