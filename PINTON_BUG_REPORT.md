# Solver bug report — DC-bin handling in `precalculate_mas`

**Repo**: `angularspectrum_heterogeneous`
**File**: `angular_spectrum_solver.py`, function `precalculate_mas`
**Found**: 2026-04 (during xAM port at f₀ = 7.8125 MHz)
**Status**: Fix applied in-place at lines 445–453

---

## Summary

The analytic-signal construction in `precalculate_mas` had an off-by-one in the
zeroing of negative frequencies, which left the **DC bin un-zeroed** for
even-length time transforms, and then **doubled** it along with the rest of
the spectrum. Any DC residue in the source field then grew by ≈2× per
propagation step, leading to exponential blow-up at high carrier frequencies
where many propagation steps are required.

The fix is one character: change the slice end from `(nT + 1) // 2` to
`nT // 2 + 1` so that for even `nT` the index `nT/2` (the DC of the
two-sided spectrum after the rfft layout used here) is included in the zeroed
half.

---

## Original (buggy) code

```python
# inside precalculate_mas, after building HH = fft(...) over the t-axis
HH[:, :, :(nT + 1) // 2] = 0    # zero "negative" half
HH *= 2                          # analytic signal: double positive half
```

For `nT = 60` (even):
* `(nT + 1) // 2 == 30`
* The slice `[:30]` zeros indices 0..29
* Index `30` (the Nyquist) is left as-is
* `HH *= 2` then doubles it

For `nT = 61` (odd):
* `(nT + 1) // 2 == 31`
* The slice `[:31]` zeros indices 0..30
* Correct.

So the bug appears **only for even `nT`** — which is exactly what most setups
use (xAM here uses `nT = 240`).

In our convention (verified with `print(HH[..., 0])` before the multiply, where
HH was a stack of forward-time FFTs), the index treated as "DC" ends up at
`nT/2`. Because that bin was not zeroed and was then doubled, any DC component
of the source pulse — even a tiny rounding-level residue from an asymmetric
Gaussian envelope — was amplified geometrically as the field is repeatedly
multiplied by the propagator.

## Fixed code

```python
# Zero negative frequencies AND DC bin, then double positive frequencies.
HH[:, :, :nT // 2 + 1] = 0       # +1 so the DC bin (index nT/2 for even nT)
                                  #  is included in the zeroed half
HH *= 2
if split_step:
    HH_half[:, :, :nT // 2 + 1] = 0
    HH_half *= 2
```

---

## Symptom

At `f₀ = 7.8125 MHz` with the standard xAM pulse (1-cycle Gaussian, p₀ ≈ 150
kPa per element, 25 mm propagation in water), the un-fixed solver produced a
peak field that grew exponentially with `z`:

| z [mm] | peak \|p\| [Pa] (buggy) | peak \|p\| [Pa] (fixed) |
|---|---|---|
|  1 |    8 × 10³ |    8 × 10³ |
|  5 |    3 × 10⁵ |    1.5 × 10⁵ |
| 10 |    1 × 10⁸ |    2.5 × 10⁵ |
| 15 |    3 × 10¹⁰ |    3.7 × 10⁵ |
| 20 |    1 × 10¹³ |    3.0 × 10⁵ |
| 25 |    NaN     |    1.0 × 10⁵ |

(Order-of-magnitude figures from a single-angle 6° xAM source.)

The growth factor per step empirically matched 2× — consistent with a DC
component being doubled by `HH *= 2` once per propagation step.

## Root cause walk-through

1. The source field `apa(x, y, t)` from a sine pulse with a finite envelope
   has a small DC component when `nT` is even, because the discrete waveform
   isn't perfectly antisymmetric.
2. The propagator `HH(kx, ky, ω)` has DC bin nonzero in the buggy version
   (because that bin is left as-is and then doubled).
3. Each propagation step does `field_fft *= HH`, so the DC component of the
   field is multiplied by `HH(kx, ky, ω=0)` — which has magnitude > 1 from
   the doubling — at every step.
4. Over N steps this grows as roughly `2^N`, swamping the actual signal.

The bug was masked at lower carrier frequencies (≤ 1 MHz) because the smaller
total number of propagation steps kept the cumulative gain bounded. At
f₀ = 7.81 MHz with `dz_mult = 1` we run ~500 steps over 25 mm, where 2^500 is
unrecoverable.

## Verification

After the fix:
* Stable at `f₀ = 7.81 MHz` with isotropic 50 µm grid (`dz_mult = 1`),
  500 propagation steps, 25 mm domain.
* Pre-fix: required `dz_mult = 8` (dZ = 8·dX = 400 µm) to stay numerically
  bounded, and even then exhibited ~11 % bias on through-skull peaks vs the
  finer grid (which was unusable).
* Post-fix: `dz_mult = 1` and `dz_mult = 2` differ by < 0.1 % on through-skull
  peaks; converged.

DC bin of the source field, measured directly:
```
np.abs(np.fft.fft(apa, axis=2)[..., 0]).max() = 6.4e-6 (≈ noise floor)
```
After 500 steps of the buggy propagator: this same DC component grows to
≈ 10¹³ — clearly the runaway mode.

## Recommendation for upstream

If `precalculate_mas` is shared with the upstream Pinton `angularspectrum`
package, the same fix should be applied there. The character change is:

```diff
-    HH[:, :, :(nT + 1) // 2] = 0
+    HH[:, :, :nT // 2 + 1] = 0
```

(and the analogous line for `HH_half` if `useSplitStep=True`).

Suggested test: drive a uniform plane wave at any frequency where `nT` is
even, propagate a fixed distance with several values of `nT` (e.g.
`nT ∈ {120, 240, 480}`), and check that the peak field magnitude is
independent of `nT`. Without the fix, the peak diverges as `nT` grows
because more even-`nT` configurations are exposed to the bug.

---

## Other things noticed during the port (FYI, not bugs)

* **Stability** at high `f₀` is now governed only by the usual CFL constraint
  (`dT ≤ dX/(4·c₀)` in our config) and the `dZmin` setting in `SolverParams`.
  The bug was masquerading as a stability problem before.
* **`useSplitStep=True`** path also had the analogous bug on `HH_half` —
  fixed at the same time.
* `useAdaptiveFiltering=False, useTVD=False, fluxScheme='rusanov'`,
  `useFreqWeightedBoundary=False` are the settings that work cleanly for our
  xAM port; we didn't probe other combinations after the DC fix.
