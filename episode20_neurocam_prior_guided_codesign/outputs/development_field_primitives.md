# Prospective scope of direct-surface development primitives

Prepared 2026-10-08 00:29:51 UTC, before running these primitives or inspecting
any resulting field or candidate outcome. This is a bounded implementation of
selected equations, not completion of the cortical-field model specification,
reference qualification, an executed condition panel or a candidate result.

The existing [component construction](component_details.md#concrete-implementations-of-the-existing-field-families)
permits direct surface fields in development. The
[selected catalogue](parameter_candidate_catalogue.md#2-selected-development-controls)
defines Gaussian widths, anisotropy, finite event support, wave conventions
and component-power fractions. [development_fields.py](tools/development_fields.py)
implements those reusable primitives with explicit caller arrays. It selects
no numerical world parameters and generates no random worlds.

## Implemented mathematical conventions

- Positions use micrometres, time seconds, voltage microvolts, angles radians.
  A covariance call produces only the explicit left-by-right point block.
  Its minor-axis SD is `ell`; its major-axis SD is `axis_ratio * ell`, at the
  supplied angle. A Gaussian spatial envelope converts its minor-axis FWHM
  to SD by `FWHM / (2 sqrt(2 log(2)))` and has peak one. Neither is calibrated
  voltage or an assertion that a supplied mode basis realizes that covariance.
- Caller-supplied spatial modes and temporal drivers combine by linear
  multiplication. No draw law, covariance factorization, periodic boundary,
  epsilon regularization, nonlinear collision or redraw is introduced.
- A focal event's `onset_s` is its finite-support **start**, with Gaussian centre
  `onset + 3 sigma` and end `onset + 6 sigma`. The caller supplies a sampled
  biphasic driver and a sampled outer taper: one through 2.5 SD, zero at/beyond
  3 SD and in `[0,1]` between. Sampling must retain both lobes. The caller must
  specify and justify the smooth continuous taper; sample checks cannot prove
  smoothness. A returned raw event is exactly zero before its support. Linear
  sums retain overlap. Envelope FWHM is not measured event waveform width.
- Planar phase is `2 pi (distance / wavelength - f (t - t0)) + phase0`, using
  cyclic spatial frequency. Expanding phase replaces distance with radial
  distance minus the supplied radius at `t0`. Positive frequency gives outward
  speed `f * wavelength`. These phase primitives do not choose wave amplitude,
  envelope, onset or behaviour at the centre/boundaries.
- A supplied fixed **linear** 1–100 Hz transform and exact common space-time
  mask determine normalization. There is no default filter, boundary rule,
  reference or spatial demeaning. Both raw and transformed fields are scaled;
  transform linearity is the caller's contract. Raw finite support and possible
  transformed pre-onset ringing remain separate. No scheduler accesses that
  ringing through these helpers. RMS is the square root of the mean square over
  selected samples.
  The optional `peak_to_peak` operation is explicitly the global maximum minus
  minimum over selected space-time samples; adopting that statistic for focal
  conditions remains a prospective estimand choice below.
- Mixtures set `P_local / (P_broad + P_local)` before cross terms. The supplied
  `component_rms_uv**2` sets that component-power sum. Reports retain both
  centred covariance and uncentred cross moment, with
  `P_sum = P_broad + P_local + 2 E[broad * local]`. Cancellation is kept; the sum
  is not silently renormalized. A selected active component with zero eligible
  power is undefined and raises an error; inactive endpoint components may be
  zero. No epsilon or replacement draw is used.
- Explicit additive nuisance enters observations. Explicit missingness marks
  observations NaN, including dead rows; neither changes truth nor intersects
  the caller's evaluation mask. Contact effects, reference transformations,
  drift and saturation remain in their separately specified forward stages.

Every function requires `max_elements`. Each existing NumPy input and any
expanded output must fit that caller-selected limit. Covariance is a block API,
not an allocation of the full cortical grid covariance. Temporary memory is a
constant multiple of that limit, so the caller must allow room for intermediate
arrays and separately bound transform workspace. Event contributions can be
accumulated one at a time. No runtime or storage budget is enlarged here.

## Choices still required before outcome use

1. Explicit spatial grid, time samples, duration, boundary convention, common
   mask and treatment of nonuniform sample weights. Current power arithmetic
   gives each selected sample equal weight; it does not claim area/time
   quadrature for irregular sampling.
2. Broad-field temporal spectrum and stochastic draw law, finite-grid covariance
   realization, seeds and common world pairing. Supplying a covariance block
   alone is not a selected random-field generator.
3. Exact biphasic waveform, continuous outer taper, event onset/location draws,
   edge-region definition and independent onset offsets. The selected catalogue
   does not freeze these. Wave amplitude/envelope and boundary choices also
   require explicit declaration.
4. The primary-band operator, padding/window/reference conventions, focal
   peak-to-peak estimand and achieved-waveform-width measurement. The generic
   global range helper does not silently settle per-location versus global
   event amplitude. Explicit masks do not make those definitions interchangeable.
5. The separately prepared [field-power scale estimator specification](field_power_scale_specification.md)
   declares a full rectangular-grid DFT estimator with spatial 1–100 Hz power
   for `lambda90` and full demeaned temporal power for `tau90`. It does not select
   a development-world grid, aperture or waveform/filter configuration. Those
   choices and any irregular/common-mask estimation remain unresolved; this
   primitive module does not select or execute the estimator.
6. Source-to-surface models, contact/electrical parameters, DAQ anchor context,
   authentic Rat5 geometry/acquisition mapping and all existing independent
   final-evaluation requirements. These primitives do not resolve or bypass them.

The focused [test module](tools/test_development_fields.py) contains ten small
analytic fixtures for axes/FWHM, finite support, wave propagation, filtering,
component interference and observation masks. Its values and periodic FFT
operator are test fixtures, not selected scientific conditions. Tests must run
in a coordinated Slurm allocation; this note records no test pass or run result.
