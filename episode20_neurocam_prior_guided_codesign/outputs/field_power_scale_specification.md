# Prospective development field-power scale estimator

Prepared October 7, 2026. The [array helper](tools/field_power_scales.py) and
[eight focused tests](tools/test_field_power_scales.py) implement the existing
EP20 definitions of `lambda = 1/(2*k90)` and `tau = 1/(2*f90)` using the explicit
development choices below. **The tests are unrun.** This is component preparation,
not a simulated candidate run, calibration, complete field-world configuration,
accepted reference model, independent-final implementation or hardware result.

The named estimator is `ep20_rectangular_dft_power90_v1`. Each return value records
its name, conventions, units, represented temporal support, grid size and finite
aperture. Its input role is supplied simulated truth or a separately authorized
known-input pilot; a candidate reconstruction cannot supply these scales.

## Grid, aperture and voltage reference

Inputs are a real finite array in `[time, y, x]` order, positive `dx_micrometres`,
`dy_micrometres`, `dt_seconds`, and an explicit Boolean common spatial mask. The
spacings declare a regular rectangular grid and uniformly sampled time. No
coordinates, missing samples, units or resampling operation are inferred.

The initial implementation requires the **entire supplied rectangular mask**.
Irregular masks and partial rectangular masks are refused. It never fills masked
sites with zeros or silently crops them. A caller-supplied smaller full rectangle
defines a different declared aperture; selecting such an aperture is outside this
helper. No complete development-world grid, duration or aperture has been selected
by adding this component. General irregular-support estimation remains unresolved.

All spatial samples have equal weight. The spatial common mode is retained, with
no spatial demeaning or new voltage-reference operation. The temporal mean is
removed separately at every spatial sample before the transform. This removes
temporal DC and leaves positive-frequency primary-band coefficients unchanged in
exact arithmetic. Spatial and temporal windows are rectangular, without taper or
padding; the DFT uses periodic extension of the supplied finite aperture. This is
a finite-record spectral statistic, not a correction for boundaries or an intrinsic
autocorrelation scale.

The return metadata distinguishes DFT periods `Nx*dx`, `Ny*dy`, `Nt*dt` from sample
centre spans `(Nx-1)*dx`, `(Ny-1)*dy`, `(Nt-1)*dt`. Finite-aperture leakage and changes
to a record's frequency grid can change its reported scales and are retained.

## Power and cutoffs

Transforms use orthonormal DFT normalization and float64 arithmetic. Power is
squared coefficient modulus, summed over all common samples. Frequencies are
cycles per micrometre and cycles per second, not angular frequency. Real-input
temporal FFT positive-frequency energies are doubled, except the even-length
Nyquist bin, which is counted once. Temporal DC is excluded.

For **lambda**, select temporal DFT bins whose frequencies satisfy `1 <= f <= 100`
Hz. Spatial transforms of these planes provide radial spatial power, including
`kx = ky = 0`. Equal computed squared spatial radii are grouped exactly, without
approximate radial bins. Negative temporal frequencies have reflected conjugate
spatial spectra and equal radial-shell energy; the real-signal doubling therefore
preserves their full radial energy. `k90` is the smallest represented radial
frequency whose cumulative complete-shell energy is at least 90% of the total.
There is no frequency interpolation or wavelength clipping.

For **tau**, use all positive temporal DFT bins of the full supplied time-demeaned
truth, summing power across every common spatial sample. This support extends
through Nyquist when represented and **is not restricted to 1–100 Hz**. `f90` is
the smallest represented temporal frequency reaching 90% of that total power.

The helper reports the actual largest temporal bin and sampling Nyquist. It cannot
recover frequencies absent from the supplied sampling rate or duration. A record
with no represented bins in the inclusive primary band has an explicit undefined
lambda status; lack of band bins is distinguished from zero power in existing bins.

## Undefined and infinite values

Zero eligible power yields an undefined scale (`None` and an explicit status).
No epsilon power floor turns numerical absence into a defined field scale. A
defined zero spatial cutoff yields `lambda = infinity`; it is not missing data.
Thus a temporally varying spatial common mode can have infinite lambda and finite
tau, whereas a zero or time-constant field has undefined positive-frequency scales.

Lambda is reported in micrometres. Pixels are reported only when the explicitly
supplied `dx` and `dy` are exactly equal; an anisotropic grid reports no single
pixel scale and retains both axis spacings. Tau is in seconds. Cutoff frequencies,
total spectral power in input-amplitude units squared, and cumulative fractions
immediately below/through the cutoff accompany the scales.

## Validation scope and execution limits

The eight prepared tests address analytic on-bin waves, anisotropic units, retained
spatial common mode versus zero power, out-of-band temporal support, real-FFT
Nyquist weighting, inclusive band edges, invalid/masked grids, exact radial ties,
and finite-aperture changes. They use small analytic fixtures only; they are not
evidence that a simulated device, field generator or reconstruction model is valid.

The array helper opens no files, performs no calibration or candidate selection,
and grants no execution authority. It stores a full temporal spectrum and streams
spatial transforms by temporal frequency plane; actual development-world memory
and runtime need sizing once that configuration exists. Its optional caller budget
callback is checked before transforms and between frequency planes. Root will run
the focused tests inside a separately selected substantive Slurm allocation.
