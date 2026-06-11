"""
1-D cubic-Burgers + linear-in-f shear-loss convergence study.

Mirrors the 3-D regime parameters (β₃, α₀, F0, c, ρ, V₀ at the kinematic
sweet spot V₀=0.025 m/s) on a pure 1-D plane-wave problem to verify
grid convergence of the kinematic peak quantities:

  v(t)       — particle velocity at z_target
  a(t)       — acceleration ∂v/∂t at z_target
  ε̇(t)       — axial strain rate ∂v/∂z at z_target (finite-difference
                 between adjacent z planes saved during the march)
  γ_axial(t) — axial strain ∂u/∂z at z_target where u = ∫v·dt

(In 1-D there is no lateral structure, so von Mises shear strain reduces
to |∂u/∂z|. The 3-D γ_vm analogy is the lateral-gradient magnitude;
the 1-D analogue is the axial gradient.)

Sweep axes:
  samples_per_cycle ∈ {30, 60, 120, 240, 480}  → DT = 1/(spc · F0)
  dZ_factor        ∈ {1.0, 0.5, 0.25, 0.125}  → dZ = dZ_factor · dZ_safe

Uses the same _kt_flux_cubic kernel from the 3-D solver, called with
a degenerate (1,1,nT) array so only the cubic-Burgers time advection
operates. Attenuation applied as a separate split-step in frequency
domain after each z-step.

Output: validation_results/convergence_1d/{convergence_table.txt,
        convergence_DT.png, convergence_DZ.png, traces_grid.png}
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
import jax
import jax.numpy as jnp
from angular_spectrum_solver import _kt_flux_cubic

# ---- physical regime (matches 3-D cubic shear lossy bowl) ----
F0     = 100.0
C0     = 2.0
RHO0   = 1000.0
BETA3  = 5.0e8
ALPHA0 = 0.5       # Np/m / Hz (linear-in-f shear loss, brain regime)
Y      = 1.0
NCYCLES = 4

# At the optimum identified by the low-V₀ sweep
V0      = 0.025
LAM     = C0 / F0
TARGET_Z = 0.06    # 6 cm — matches V₀=0.025 z_focus from the 3D sweep

OUT = os.path.join(os.path.dirname(__file__),
                   'validation_results', 'convergence_1d')
os.makedirs(OUT, exist_ok=True)


def build_pulse(taxis: np.ndarray, v0: float = V0) -> np.ndarray:
    """Single Gaussian-windowed sinusoidal pulse."""
    env = np.exp(-((taxis * F0 / (NCYCLES * 0.6))**2) * 4)
    return (v0 * np.sin(2 * np.pi * F0 * taxis) * env).astype(np.float32)


def attenuation_step(field_np: np.ndarray, dz: float, dt: float):
    """Apply linear-in-f shear loss in the frequency domain for one
    split-step of length dz. α(f) = ALPHA0 · f."""
    nT = field_np.shape[-1]
    fs = np.fft.fftfreq(nT, dt)
    alpha = ALPHA0 * np.abs(fs) ** Y    # Np/m
    decay = np.exp(-alpha * dz).astype(np.float32)
    spec = np.fft.fft(field_np, axis=-1)
    return np.real(np.fft.ifft(spec * decay, axis=-1)).astype(np.float32)


def march_1d(samples_per_cycle: int, dZ_factor: float,
             target_z: float = TARGET_Z, v0: float = V0):
    """One 1-D propagation run. Returns dict of fields at z_target and
    z_target - 2*dZ_save (for spatial finite-difference) plus diagnostics."""
    DT = 1.0 / (samples_per_cycle * F0)
    pulse_dur = 4.0 * NCYCLES / F0
    nT = int(np.ceil(pulse_dur / DT));  nT += (nT % 2 == 0)
    taxis = (np.arange(nT) - nT // 2) * DT

    N3 = BETA3 / (3 * C0**5 * RHO0**2)
    # CFL: KT cubic needs dZ · N3 · v² < ~0.15 · dT
    safe_dZ = 0.15 * DT / max(v0**2 * N3, 1e-30)
    dZ_target = max(0.125 * safe_dZ, DT * C0 / 4.0)
    dZ_target = min(dZ_target, LAM / 10.0)
    dZ = dZ_factor * dZ_target

    init = build_pulse(taxis, v0=v0)
    field = init.reshape(1, 1, nT)
    field_jax = jnp.asarray(field, dtype=jnp.float32)

    # March, capture at three z planes: z_target - dz_capture,
    # z_target, z_target + dz_capture. (dz_capture = dZ chosen per run.)
    captured = {}
    z = 0.0
    # planes to capture (cm); use absolute positions for repeatability
    z_pre  = target_z - dZ
    z_post = target_z + dZ
    z_planes = [(z_pre, 'pre'), (target_z, 'at'), (z_post, 'post')]
    next_plane_idx = 0

    n_steps = 0
    while z < target_z + dZ + 1e-12 and next_plane_idx < len(z_planes):
        dz_step = min(dZ, z_planes[next_plane_idx][0] - z)
        if dz_step < 1e-12:
            # already at next plane
            captured[z_planes[next_plane_idx][1]] = np.asarray(field_jax)
            next_plane_idx += 1
            continue
        # nonlinear KT step
        field_jax = _kt_flux_cubic(field_jax, N3, dz_step, DT)
        # attenuation step (np)
        field_np = np.asarray(field_jax)
        field_np = attenuation_step(field_np, dz_step, DT)
        field_jax = jnp.asarray(field_np)
        z += dz_step
        n_steps += 1
        if abs(z - z_planes[next_plane_idx][0]) < 1e-9:
            captured[z_planes[next_plane_idx][1]] = field_np.copy()
            next_plane_idx += 1

    # Quantities at z_target (the "at" plane)
    v_at   = captured['at'  ][0, 0, :]
    v_pre  = captured['pre' ][0, 0, :]
    v_post = captured['post'][0, 0, :]

    # Time derivatives at z_target
    a_at = np.gradient(v_at, DT)

    # Spatial derivative via central difference between pre/post planes
    eps_at = (v_post - v_pre) / (2.0 * dZ)        # ∂v/∂z

    # Displacement and axial strain
    u_at  = np.cumsum(v_at,  axis=-1) * DT
    u_pre = np.cumsum(v_pre, axis=-1) * DT
    u_post= np.cumsum(v_post, axis=-1) * DT
    gamma_axial_at = (u_post - u_pre) / (2.0 * dZ)   # ∂u/∂z

    metrics = {
        'samples_per_cycle': int(samples_per_cycle),
        'dZ_factor':         float(dZ_factor),
        'DT_s':              float(DT),
        'dZ_m':              float(dZ),
        'n_steps':           int(n_steps),
        'n_T':               int(nT),
        'v_peak':       float(np.max(np.abs(v_at))),
        'a_peak':       float(np.max(np.abs(a_at))),
        'eps_peak':     float(np.max(np.abs(eps_at))),
        'gamma_peak':   float(np.max(np.abs(gamma_axial_at))),
        'a_rms':        float(np.sqrt(np.mean(a_at**2))),
        'eps_rms':      float(np.sqrt(np.mean(eps_at**2))),
        'gamma_rms':    float(np.sqrt(np.mean(gamma_axial_at**2))),
    }
    traces = {
        't_ms':         (taxis * 1e3).astype(np.float32),
        'v':            v_at.astype(np.float32),
        'a':            a_at.astype(np.float32),
        'eps':          eps_at.astype(np.float32),
        'gamma':        gamma_axial_at.astype(np.float32),
    }
    return metrics, traces


def run_operating_point(v0: float, target_z: float,
                        SPC_LIST, DZ_LIST, tag: str):
    """One operating point — sweep spc × dZ_factor."""
    runs = []
    saved_traces = {}
    print(f'\n=== Operating point [{tag}]: V₀={v0} m/s, '
          f'z_target={target_z*100:.1f} cm ===')
    print(f'{"spc":>4} {"dZf":>6} {"DT (μs)":>9} {"dZ (μm)":>9} '
          f'{"n_st":>5} {"v_pk":>9} {"a_pk":>10} {"ε̇_pk":>10} {"γ_pk":>10} '
          f'{"edge":>6}')
    print('-' * 96)
    for spc in SPC_LIST:
        for dZf in DZ_LIST:
            t0 = time.time()
            m, traces = march_1d(spc, dZf, target_z=target_z, v0=v0)
            wall = time.time() - t0
            m['wall_s'] = wall
            m['tag'] = tag
            m['v0_input'] = v0
            m['target_z'] = target_z
            # Edge-steepness diagnostic
            omega0 = 2*np.pi*F0
            m['edge_steep'] = m['a_peak'] / max(omega0 * m['v_peak'], 1e-30)
            runs.append(m)
            saved_traces[f'spc{spc}_dz{dZf}'] = traces
            print(f'{spc:>4} {dZf:>6.3f} {m["DT_s"]*1e6:>9.2f} '
                  f'{m["dZ_m"]*1e6:>9.2f} {m["n_steps"]:>5} '
                  f'{m["v_peak"]:>9.4f} {m["a_peak"]:>10.3e} '
                  f'{m["eps_peak"]:>10.3e} {m["gamma_peak"]:>10.3e} '
                  f'{m["edge_steep"]:>6.2f}')
    return runs, saved_traces


def main():
    SPC_LIST = [30, 60, 120, 240, 480]
    DZ_LIST = [1.0, 0.5, 0.25, 0.125]

    # Three operating points covering the relevant kinematic regimes:
    #   LIN  — linear-ish, low V₀ (post-attenuation envelope at z_target)
    #   MID  — partial shock formation (edge_steep ~ 1.5)
    #   SHK  — strong shock (edge_steep > 3) over a shorter distance
    operating_points = [
        ('LIN',  0.025,  0.06),
        ('MID',  0.20,   0.03),
        ('SHK',  0.50,   0.02),
    ]

    runs = []
    saved_traces = {}
    for tag, v0, target_z in operating_points:
        rs, ts = run_operating_point(v0, target_z, SPC_LIST, DZ_LIST, tag)
        runs.extend(rs)
        saved_traces[tag] = ts

    # Save JSON metrics
    with open(os.path.join(OUT, 'convergence_runs.json'), 'w') as f:
        json.dump(runs, f, indent=2)
    # Save traces (per operating point)
    save_kwargs = {}
    for tag, runs_traces in saved_traces.items():
        for key, tr in runs_traces.items():
            for q in ['t_ms', 'v', 'a', 'eps', 'gamma']:
                save_kwargs[f'{tag}_{key}_{q}'] = tr[q]
    np.savez(os.path.join(OUT, 'convergence_traces.npz'), **save_kwargs)

    # ---- convergence table (per operating point) ----
    lines = []
    lines.append('1-D cubic-Burgers + shear-loss convergence study')
    lines.append('=' * 96)
    lines.append('Operating points:')
    for tag, v0, tz in operating_points:
        lines.append(f'  {tag}: V₀={v0} m/s, z_target={tz*100:.1f} cm')
    lines.append('')
    lines.append('Quantities at z_target (peak in time of |·| at on-axis trace):')
    lines.append('  v       — particle velocity   (m/s)')
    lines.append('  a       — acceleration       (m/s²)')
    lines.append('  ε̇       — axial strain rate ∂v/∂z (1/s)')
    lines.append('  γ_a     — axial strain ∂u/∂z (1)')
    lines.append('  edge    — a_peak / (ω₀ · v_peak)  — 1=linear, >1=shocked')
    lines.append('')
    for tag, v0, tz in operating_points:
        lines.append(f'[{tag}] V₀={v0} m/s, z_target={tz*100:.1f} cm')
        lines.append('-' * 96)
        lines.append(f'{"spc":>4} {"dZf":>6} {"DT(μs)":>8} {"dZ(μm)":>8} '
                     f'{"n_st":>5} {"v_pk":>10} {"a_pk":>11} {"ε̇_pk":>11} '
                     f'{"γ_pk":>11} {"edge":>6} {"wall(s)":>8}')
        for m in [r for r in runs if r['tag'] == tag]:
            lines.append(
                f'{m["samples_per_cycle"]:>4} {m["dZ_factor"]:>6.3f} '
                f'{m["DT_s"]*1e6:>8.2f} {m["dZ_m"]*1e6:>8.2f} '
                f'{m["n_steps"]:>5} '
                f'{m["v_peak"]:>10.5f} {m["a_peak"]:>11.4e} '
                f'{m["eps_peak"]:>11.4e} {m["gamma_peak"]:>11.4e} '
                f'{m["edge_steep"]:>6.2f} '
                f'{m["wall_s"]:>8.2f}'
            )
        lines.append('')

    lines.append('=' * 96)
    lines.append('Convergence rate vs samples_per_cycle (at dZ_factor=0.125)')
    lines.append('=' * 96)
    for tag, v0, tz in operating_points:
        lines.append(f'[{tag}] V₀={v0} m/s')
        refs = [r for r in runs if r['tag'] == tag and r['dZ_factor'] == 0.125]
        refs.sort(key=lambda x: x['samples_per_cycle'])
        prev = None
        for r in refs:
            if prev is None:
                lines.append(f'  spc {r["samples_per_cycle"]:>3}: '
                             f'v={r["v_peak"]:.5f}, a={r["a_peak"]:.4e}, '
                             f'ε̇={r["eps_peak"]:.4e}, γ={r["gamma_peak"]:.4e} '
                             f'(edge={r["edge_steep"]:.2f})')
            else:
                dv = (r['v_peak']     - prev['v_peak'])     / max(abs(prev['v_peak']),1e-30) * 100
                da = (r['a_peak']     - prev['a_peak'])     / max(abs(prev['a_peak']),1e-30) * 100
                de = (r['eps_peak']   - prev['eps_peak'])   / max(abs(prev['eps_peak']),1e-30) * 100
                dg = (r['gamma_peak'] - prev['gamma_peak']) / max(abs(prev['gamma_peak']),1e-30) * 100
                lines.append(f'  spc {prev["samples_per_cycle"]:>3} → '
                             f'{r["samples_per_cycle"]:>3}: '
                             f'Δv={dv:+6.2f}%, Δa={da:+6.2f}%, '
                             f'Δε̇={de:+6.2f}%, Δγ={dg:+6.2f}%')
            prev = r
        lines.append('')
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT, 'convergence_table.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)

    # ---- convergence plots ----
    # Panel 1: vs samples_per_cycle (one curve per dZ_factor)
    # Panel 2: vs dZ_factor (one curve per spc)
    quantities = [('v_peak',     'v_peak (m/s)'),
                  ('a_peak',     '|a|_peak (m/s²)'),
                  ('eps_peak',   '|ε̇|_peak (1/s)'),
                  ('gamma_peak', '|γ_a|_peak')]

    # Build one set of convergence plots per operating point.
    for tag, v0, tz in operating_points:
        ops_runs = [r for r in runs if r['tag'] == tag]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8),
                                 constrained_layout=True)
        for j, (k, ylab) in enumerate(quantities):
            ax = axes[0, j]
            for dZf in DZ_LIST:
                xs = [r['samples_per_cycle'] for r in ops_runs
                      if r['dZ_factor'] == dZf]
                ys = [r[k] for r in ops_runs if r['dZ_factor'] == dZf]
                ax.plot(xs, ys, 'o-', label=f'dZf={dZf:.3f}', lw=1.4)
            ax.set(xlabel='samples_per_cycle', ylabel=ylab,
                   xscale='log', title=f'{k} vs DT refinement')
            ax.grid(alpha=0.3, which='both')
            if j == 0:
                ax.legend(fontsize=8)
            ax = axes[1, j]
            for spc in SPC_LIST:
                xs = [r['dZ_factor'] for r in ops_runs
                      if r['samples_per_cycle'] == spc]
                ys = [r[k] for r in ops_runs
                      if r['samples_per_cycle'] == spc]
                order = np.argsort(xs)
                xs = np.array(xs)[order]; ys = np.array(ys)[order]
                ax.plot(xs, ys, 'o-', label=f'spc={spc}', lw=1.4)
            ax.set(xlabel='dZ_factor', ylabel=ylab, xscale='log',
                   title=f'{k} vs dZ refinement')
            ax.grid(alpha=0.3, which='both')
            if j == 0:
                ax.legend(fontsize=8)
        edge_typical = ops_runs[-1]['edge_steep']
        fig.suptitle(f'1-D convergence [{tag}] — V₀={v0} m/s, '
                     f'z_target={tz*100:.1f} cm, edge_steep≈{edge_typical:.2f}',
                     fontsize=12)
        fig.savefig(os.path.join(OUT,
                                 f'convergence_quantities_{tag}.png'), dpi=160)
        plt.close(fig)

        # Coarsest vs finest trace overlay for this operating point
        ts = saved_traces[tag]
        coarsest_key = 'spc30_dz1.0'
        finest_key   = 'spc480_dz0.125'
        if coarsest_key in ts and finest_key in ts:
            fig, axes = plt.subplots(4, 1, figsize=(13, 11),
                                     constrained_layout=True, sharex=True)
            quantities2 = [('v', 'v (m/s)'),
                           ('a', 'a (m/s²)'),
                           ('eps', '∂v/∂z (1/s)'),
                           ('gamma', '∂u/∂z')]
            for ax, (k, ylab) in zip(axes, quantities2):
                ax.plot(ts[coarsest_key]['t_ms'], ts[coarsest_key][k],
                        'r-', lw=0.8, alpha=0.8,
                        label='spc=30, dZf=1.0 (coarsest)')
                ax.plot(ts[finest_key]['t_ms'], ts[finest_key][k],
                        'b-', lw=0.9,
                        label='spc=480, dZf=0.125 (finest)')
                ax.set(ylabel=ylab); ax.grid(alpha=0.3)
                ax.axhline(0, color='k', lw=0.3)
                ax.legend(fontsize=8, loc='upper right')
            axes[-1].set_xlabel('t (ms)')
            fig.suptitle(f'[{tag}] coarsest vs finest grid waveforms at '
                         f'z = {tz*100:.1f} cm (V₀={v0} m/s)', fontsize=12)
            fig.savefig(os.path.join(OUT,
                                     f'traces_coarsest_vs_finest_{tag}.png'),
                        dpi=160)
            plt.close(fig)

    print(f'\nOutputs in {OUT}')


if __name__ == '__main__':
    main()
