"""
Paraxial converging Gaussian source for theory-validation simulations.

Generates an initial field at z=0 that focuses to a Gaussian beam at
z=focus.  The implementation mirrors `make_bowl_source` exactly except
that the hard top-hat aperture is replaced by a soft Gaussian
amplitude apodization, suppressing Airy sidelobes:

  v(x, y, t) = V_0 · apod(r) · envelope(t - d(r)) · sin(ω_0 (t - d(r))),
  apod(r)   = exp(-r²/waist_source²),
  d(r)      = (max_r √(r²+F²) - √(r²+F²)) / c_0          (centre fires latest)

The exact spherical path-to-focus distance √(r²+F²) is used (rather than
the paraxial r²/(2F) approximation), which matters once r/F ≳ 0.3 and is
the convention used by `make_bowl_source`.  Inverting the delays so the
outermost pixel fires first produces the converging-wave sign convention
expected by the angular-spectrum solver.

Pulse envelope uses the same super-Gaussian shape as `make_bowl_source`:
  envelope(t') = exp(-(1.05·t'·ω_0/(N_cycles·π))^(2·dur))
with `dur=2` giving a 4-th-order super-Gaussian, sharp-edged tone burst.
"""
from __future__ import annotations
import numpy as np


def make_gaussian_source(xaxis: np.ndarray,
                         yaxis: np.ndarray,
                         taxis: np.ndarray,
                         F0: float,
                         C0: float,
                         V0: float,
                         waist_source: float,
                         focus: float,
                         ncycles: float = 4.0,
                         dur: int = 2) -> np.ndarray:
    """Build initial-field array for converging Gaussian source.

    Parameters
    ----------
    xaxis, yaxis : 1-D arrays
        Spatial coordinates (m).
    taxis : 1-D array
        Time samples (s), centred on t=0.
    F0 : float
        Carrier frequency (Hz).
    C0 : float
        Wave speed (m/s).
    V0 : float
        Source-plane peak particle velocity (m/s).
    waist_source : float
        Source-plane Gaussian amplitude waist (m). Field amplitude
        falls as exp(-r²/waist_source²).
    focus : float
        Geometric focal distance from z=0 (m).
    ncycles : float
        Tone-burst length in carrier cycles.
    dur : int
        Super-Gaussian envelope order parameter (matches make_bowl_source).

    Returns
    -------
    init : ndarray (nX, nY, nT), float32
    """
    X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    r2 = X**2 + Y**2
    apod = np.exp(-r2 / waist_source**2)

    # Exact path-to-focus distance
    dist_flat = np.sqrt(r2 + focus**2)
    delays = dist_flat / C0       # absolute travel time to focus

    # Invert: outermost (longer path) fires earliest, centre fires latest
    delays = delays.max() - delays

    omega0 = 2.0 * np.pi * F0
    t_grid = taxis[np.newaxis, np.newaxis, :]
    d_grid = delays[:, :, np.newaxis]
    envelope = np.exp(-(1.05 * (t_grid - d_grid) * omega0
                        / (ncycles * np.pi))**(2 * dur))
    field = (apod[:, :, np.newaxis] * envelope
             * np.sin(omega0 * (t_grid - d_grid)) * V0).astype(np.float32)
    return field
