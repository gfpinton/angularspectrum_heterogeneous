"""
Cubic-Burgers Riemann benchmark.

Tests the cubic flux kernels (`_rusanov_flux_standard_cubic`,
`_rusanov_flux_tvd_cubic`, `_kt_flux_cubic`) against analytical entropy
solutions of the 1D cubic Burgers equation

    ∂u/∂z + ∂(-u³/3)/∂τ = 0

with characteristic speed λ(u) = -u² (always non-positive, sonic at
u=0). f(u) = -u³/3 is non-convex (f'' = -2u changes sign at zero) so
admits compound waves.

Three Riemann problems are tested:

  Test A — compressive shock with same-sign states:
      u_L = 0,  u_R = +1
    Lax-admissible shock travels at s = -(u_L²+u_L·u_R+u_R²)/3 = -1/3.

  Test B — rarefaction (same-sign, opposite ordering):
      u_L = +1, u_R = 0
    Self-similar fan with u(τ,z) = √(-τ/z) for −z ≤ τ ≤ 0.

  Test C — compound wave (cross-zero states):
      u_L = +1, u_R = -1
    Shock from u_L=1 to u*=-1/2 at speed s_shock = -1/4 (touching the
    flux at the tangent point u*); rarefaction from u*=-1/2 to u_R=-1
    fanned over z·[-1, -1/4]/z.

The validation drives the same numerical kernels used by the 3-D
solver but in a degenerate 1×1 spatial grid, so only the cubic flux
operates — no FFT diffraction, no attenuation. The schemes' errors
are compared against the exact entropy solutions in L^2 norm.

Outputs: validation_results/cubic_riemann/{testA,testB,testC}.png
         validation_results/cubic_riemann/summary.txt
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from angular_spectrum_solver import (
    _rusanov_flux_standard_cubic,
    _rusanov_flux_tvd_cubic,
    _kt_flux_cubic,
)
import jax.numpy as jnp

OUT = os.path.join(os.path.dirname(__file__),
                   'validation_results', 'cubic_riemann')
os.makedirs(OUT, exist_ok=True)


# ---------- analytical solutions ----------
def exact_test_A(tau, z):
    """u_L=0, u_R=+1: Lax-admissible compressive shock at τ = -z/3."""
    s = -1.0 / 3.0
    return np.where(tau < s * z, 0.0, 1.0)


def exact_test_B(tau, z):
    """u_L=+1, u_R=0: rarefaction, u = √(-τ/z) on [-z, 0]."""
    out = np.empty_like(tau)
    out[tau < -z] = 1.0
    out[tau > 0] = 0.0
    mid = (tau >= -z) & (tau <= 0)
    out[mid] = np.sqrt(-tau[mid] / z)
    return out


def exact_test_C(tau, z):
    """u_L=+1, u_R=-1: rarefaction-then-shock compound wave.

    Upper concave envelope of f(u)=-u³/3 on [-1, 1] consists of the
    concave segment of f for u∈[1/2, 1] plus a tangent chord from
    (-1, 1/3) to (1/2, -1/24) (slope -1/4, the tangent point).
    Resulting entropy solution at depth z:
        τ < -z:           u = u_L = +1   (state to the left of the fan)
        -z ≤ τ ≤ -z/4:    u = +√(-τ/z)   (rarefaction)
        τ > -z/4:         u = u_R = -1   (state right of the sonic shock
                                            from u*=1/2 to u_R=-1)
    """
    out = np.empty_like(tau)
    out[tau < -z] = 1.0
    fan = (tau >= -z) & (tau <= -z / 4.0)
    out[fan] = np.sqrt(-tau[fan] / z)
    out[tau > -z / 4.0] = -1.0
    return out


# ---------- driver ----------
def make_initial(tau, u_L, u_R):
    """Step initial condition at τ=0, u_L for τ<0, u_R for τ>0."""
    field = np.where(tau < 0.0, u_L, u_R).astype(np.float32)
    return field[np.newaxis, np.newaxis, :]   # (1, 1, nT)


def march(scheme, field, dZ, dT, n_steps, beta_tvd=2.0):
    """Run cubic flux n_steps times. Returns (1,1,nT).

    The cubic flux schemes follow distinct conventions inherited from
    the existing quadratic implementation:
      - Rusanov (and TVD-Rusanov): F_code_smooth = -(u_L³+u_R³)/3
        ≈ -2u³/3 (the symmetric flux is doubled). Calibration to the
        canonical PDE ∂u/∂z = -∂(-u³/3)/∂τ requires N₃ = 0.5.
      - KT (Kurganov-Tadmor): the central-upwind formula reduces to
        f_plus = -u³/3 in the smooth limit (single form). Calibration
        to the same PDE requires N₃ = 1.0.
    Both schemes match the analytical Rankine-Hugoniot shock speed
    s = -(u_L²+u_L·u_R+u_R²)/3 with these scheme-specific coefficients.
    """
    # Standard Rusanov uses doubled symmetric flux (F_code = -2u³/3).
    # Both KT and TVD-Rusanov reduce to single form (F_code = -u³/3) in
    # the smooth limit, so they need 2× the N₃ to match the same PDE.
    if scheme == 'rusanov':
        N3 = 0.5
    else:                                    # 'rusanov_tvd' or 'kt'
        N3 = 1.0
    f = jnp.array(field)
    for _ in range(n_steps):
        if scheme == 'rusanov':
            f = _rusanov_flux_standard_cubic(f, N3, dZ, dT)
        elif scheme == 'rusanov_tvd':
            f = _rusanov_flux_tvd_cubic(f, N3, dZ, dT, beta_tvd)
        elif scheme == 'kt':
            f = _kt_flux_cubic(f, N3, dZ, dT)
        else:
            raise ValueError(f'unknown scheme {scheme}')
    return np.asarray(f)


def run_test(tag, u_L, u_R, exact_fn, z_target=1.0):
    """Run one Riemann problem and produce a comparison figure + L^2 errors."""
    nT = 4001
    tau = np.linspace(-2.0, 2.0, nT)
    dT = tau[1] - tau[0]
    dZ = 0.4 * dT      # CFL-safe for max|u|≤1: λ_max=1, N₃·dZ·u²/dT ≤ 0.4
    n_steps = int(round(z_target / dZ))
    z_actual = n_steps * dZ
    print(f'\n--- Test {tag} ({u_L}→{u_R}) ---')
    print(f'  nT={nT}, dT={dT:.4f}, dZ={dZ:.4f}, z={z_actual:.4f} '
          f'({n_steps} steps)')

    field0 = make_initial(tau, u_L, u_R)
    exact = exact_fn(tau, z_actual)

    schemes = ['rusanov', 'rusanov_tvd', 'kt']
    out = {}
    for s in schemes:
        u = march(s, field0, dZ, dT, n_steps)[0, 0, :]
        # exclude boundary cells (first/last 50) from error norm
        valid = slice(50, -50)
        err = np.sqrt(np.mean((u[valid] - exact[valid]) ** 2))
        out[s] = (u, err)
        print(f'  {s:14s}: L² error = {err:.5f}')

    # figure
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(tau, exact, 'k-', lw=1.5, label='exact (entropy soln)')
    colors = {'rusanov': 'C3', 'rusanov_tvd': 'C1', 'kt': 'C0'}
    markers = {'rusanov': '+', 'rusanov_tvd': 'x', 'kt': '.'}
    for s in schemes:
        u, err = out[s]
        sl = slice(None, None, 25)
        ax.plot(tau[sl], u[sl], markers[s], color=colors[s], ms=4,
                label=f'{s} (L²={err:.4f})')
    ax.set(xlabel='τ', ylabel='u(τ, z={:.2f})'.format(z_actual),
           title=f'Cubic Riemann Test {tag}: u_L={u_L}, u_R={u_R}',
           xlim=[-1.5, 1.5])
    ax.grid(True, alpha=0.3); ax.legend()
    out_path = os.path.join(OUT, f'test{tag}.png')
    fig.savefig(out_path, dpi=160); plt.close(fig)
    print(f'  wrote {out_path}')
    return out


# ---------- main ----------
def main():
    summary = []
    summary.append('Cubic-Burgers Riemann validation\n'
                   '================================\n')
    for tag, u_L, u_R, exact_fn in [
        ('A', 0.0, 1.0, exact_test_A),
        ('B', 1.0, 0.0, exact_test_B),
        ('C', 1.0, -1.0, exact_test_C),
    ]:
        summary.append(f'\nTest {tag}: u_L={u_L}, u_R={u_R}')
        out = run_test(tag, u_L, u_R, exact_fn)
        for s, (_, err) in out.items():
            summary.append(f'  {s:14s}: L² = {err:.5f}')

    text = '\n'.join(summary) + '\n'
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(text)
    print('\n' + text)


if __name__ == '__main__':
    main()
