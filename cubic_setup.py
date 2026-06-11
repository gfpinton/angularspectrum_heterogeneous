"""Shared solver setup for the cubic shear-shock harnesses.

The T1-T6 validation harnesses, the kinematic sweeps, and the
shock-at-focus runs all march the same configuration: split-step KT
flux with TVD limiting, cubic Burgers nonlinearity, quadratic boundary
layer, and a CFL-safe dZ floor derived from the expected focal
velocity. That setup used to be copy-pasted into each script; it lives
here now so a change to the marching configuration or the dZ heuristic
propagates everywhere at once.
"""
from __future__ import annotations

import numpy as np

from angular_spectrum_solver import SolverParams


def expected_focal_peak_gaussian(v0, f0, c0, waist_src, focus):
    """Expected focal velocity for a focused Gaussian source.

    Paraxial Fraunhofer focal gain k0·w_src²/F, floored at the source
    amplitude so the estimate never drops below v0 for weak focusing.
    """
    k0 = 2 * np.pi * f0 / c0
    return max(v0 * k0 * waist_src ** 2 / focus, v0)


def expected_focal_peak_bowl(v0, f0, c0, outer_r, focus):
    """Expected focal velocity for a bowl source: gain k0·r²/(2F)."""
    k0 = 2 * np.pi * f0 / c0
    return max(v0 * k0 * outer_r ** 2 / (2 * focus), v0)


def cubic_dzmin(expected_peak, dt, c0, lam, beta3, rho0, linear=False):
    """CFL-safe axial step floor for the cubic-Burgers split-KT march.

    Cubic Burgers has characteristic speed λ(u) = u², so the CFL number
    is N₃·dZ·u²/dT; 0.15 is the safety target at the expected focal
    peak, with a further 4x margin (0.25·safe_dZ). The floor never goes
    below a quarter time-sample of travel (dT·c0/4) and never above
    λ/10 (diffraction accuracy). Linear runs only need the λ/10 cap.
    """
    n3 = 0.0 if linear else beta3 / (3 * c0 ** 5 * rho0 ** 2)
    safe_dz = (lam / 10.0 if linear
               else 0.15 * dt / max(expected_peak ** 2 * n3, 1e-30))
    dzmin = max(0.25 * safe_dz, dt * c0 / 4.0)
    return min(dzmin, lam / 10.0)


def cubic_kt_params(*, dX, dT, c0, rho0, f0, beta3, alpha0, attenPow,
                    propDist, dZmin, useAttenLoss, **overrides):
    """The standard cubic shear-shock SolverParams used by the harnesses.

    Square transverse grid (dY = dX), quadratic nonlinearity off,
    split-step KT + TVD, adaptive k-space filtering, quadratic boundary
    profile with boundaryFactor 0.15, host-side reductions. Extra
    keyword arguments override or extend the base configuration (e.g.
    sourcePlanes=..., diagnostic=True).
    """
    base = dict(
        dX=dX, dY=dX, dT=dT, c0=c0, rho0=rho0,
        beta=0.0, beta3=beta3, nonlinearityOrder=3,
        alpha0=alpha0, attenPow=attenPow, f0=f0,
        propDist=propDist,
        useSplitStep=True, fluxScheme='kt', useTVD=True,
        useAdaptiveFiltering=True,
        boundaryProfile='quadratic',
        useFreqWeightedBoundary=False,
        useSuperAbsorbing=False, boundaryFactor=0.15,
        dZmin=dZmin, useGPUReductions=False, useAttenLoss=useAttenLoss,
    )
    base.update(overrides)
    return SolverParams(**base)
