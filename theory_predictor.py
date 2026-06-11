"""
Closed-form predictions of hyperfocal_kinematic.tex, given the on-axis
focal-plane harmonic spectrum.

For a paraxial Gaussian focused source with linear focal waist w_0, the
$n$-th harmonic obeys the central-lobe rule
    |v_n(r)|^2 = |V_n|^2 · exp(-2 n r^2 / w_0^2),
and the time-averaged kinematic maps are
    <v^2>(r)        = (1/2) Σ |V_n|² · exp(-2 n r²/w_0²)
    <a^2>(r)        = (1/2) ω_0² Σ n² |V_n|² · exp(-2 n r²/w_0²)
    <|ε̇|²>(r)      = (2 r²/w_0⁴) Σ n²|V_n|² · exp(-2 n r²/w_0²)
    <γ_vm²>(r)     = (2 r²/(w_0⁴ ω_0²)) Σ |V_n|² · exp(-2 n r²/w_0²).

Predicted lateral scales:
  centred maps (curvature waist):
      w_v / w_0 = ⟨n⟩_v^(-1/2),
      w_a / w_0 = ⟨n⟩_{a²}^(-1/2),
      ⟨n⟩_v   = Σ n|V_n|²  / Σ |V_n|²,
      ⟨n⟩_{a²} = Σ n³|V_n|² / Σ n²|V_n|².

  annular maps (ring radius of <|·|²>):
      r_ring^(ε̇) / w_0 = 1 / √(2 ⟨n⟩_{a²}),
      r_ring^(γ)  / w_0 = 1 / √(2 ⟨n⟩_v).

  half-max-area ratio (centred):
      A_{a²} / A_{v²} = (w_a/w_v)² = ⟨n⟩_v/⟨n⟩_{a²}.
"""
from __future__ import annotations
import numpy as np


def harmonic_moments(Vn2: np.ndarray, n_indices: np.ndarray | None = None):
    """Compute ⟨n⟩_v, ⟨n⟩_{a²} given an array of |V_n|² values.

    Parameters
    ----------
    Vn2 : array (K,)
        |V_n|² values for K harmonics (need not be all integers; could be
        odd-only).
    n_indices : array (K,) or None
        Harmonic indices corresponding to each Vn2 entry. If None, assumes
        n = 1, 2, ..., K.

    Returns
    -------
    dict with keys 'n_v' (⟨n⟩_v) and 'n_a2' (⟨n⟩_{a²}).
    """
    Vn2 = np.asarray(Vn2, dtype=np.float64)
    if n_indices is None:
        n_indices = np.arange(1, len(Vn2) + 1, dtype=np.float64)
    else:
        n_indices = np.asarray(n_indices, dtype=np.float64)
    eps = 1e-30
    n_v   = float((n_indices * Vn2).sum() / max(Vn2.sum(), eps))
    n_a2  = float((n_indices**3 * Vn2).sum()
                  / max((n_indices**2 * Vn2).sum(), eps))
    return dict(n_v=n_v, n_a2=n_a2)


def predict_ratios(Vn2: np.ndarray, n_indices: np.ndarray | None = None,
                   w0: float = 1.0):
    """Theory predictions for all four kinematic concentration metrics.

    Returns a dict with predicted ratios (dimensionless) and absolute
    lateral scales (in same units as w0). Ratios that are scale-free
    are the most direct comparison; absolute scales depend on w0.
    """
    m = harmonic_moments(Vn2, n_indices)
    n_v = m['n_v']
    n_a2 = m['n_a2']
    return dict(
        n_v=n_v,
        n_a2=n_a2,
        wv_over_w0=1.0 / np.sqrt(n_v),
        wa_over_w0=1.0 / np.sqrt(n_a2),
        wa_over_wv=np.sqrt(n_v / n_a2),
        Aa2_over_Av2=n_v / n_a2,
        r_ring_eps_over_w0=1.0 / np.sqrt(2.0 * n_a2),
        r_ring_gam_over_w0=1.0 / np.sqrt(2.0 * n_v),
        # Absolute scales
        wv=w0 / np.sqrt(n_v),
        wa=w0 / np.sqrt(n_a2),
        r_ring_eps=w0 / np.sqrt(2.0 * n_a2),
        r_ring_gam=w0 / np.sqrt(2.0 * n_v),
    )


def on_axis_harmonic_amplitudes(trace: np.ndarray, dt: float, F0: float,
                                n_max: int = 15,
                                bin_width_fraction: float = 0.4):
    """Extract |V_n|² for n = 1..n_max from a 1-D time trace.

    Sums the power-spectrum bins within a width 0.4·F0 of each n·F0.
    Returns an array (n_max,) of squared moduli.
    """
    nT = len(trace)
    spec = np.abs(np.fft.rfft(trace))**2
    freqs = np.fft.rfftfreq(nT, dt)
    bin_w = bin_width_fraction * F0
    Vn2 = np.zeros(n_max, dtype=np.float64)
    for k, n in enumerate(range(1, n_max + 1)):
        m = (freqs >= n * F0 - bin_w) & (freqs <= n * F0 + bin_w)
        Vn2[k] = float(np.sum(spec[m]))
    return Vn2
