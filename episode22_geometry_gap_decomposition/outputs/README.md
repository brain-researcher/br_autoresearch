# EP22 outputs

**2026-10-12 22:05 UTC — D0–D3 complete; OOD still sealed.**

- D0 and D1–D3 have run on development data.
- Compute so far: 23.76 allocated CPU-hours of the 400 budgeted, and no GPU
  use.
- No OOD response has been opened, and no lock exists.

The full record is the
[D1–D3 log entry](experiment_log.md#2026-10-12-2205-utc--d1d2d3--completed-array-47548992-collector-47549215).

| Analysis | Result (exploratory development evidence) | Artifacts |
| --- | --- | --- |
| D0 outcome-blind spectra | Corrected OOD embeddings bound: exactly 116 OOD rows changed. VLM spectra are flatter than DINOv2's everywhere, more so on LAION than on THINGS/THINGSplus, and most on OOD images. | [results](runs/d0_outcome_blind_spectra/20261012-attempt01/results/), [figure](figures/d0_outcome_blind_spectra/20261012-attempt01/d0_outcome_blind_spectra.png) |
| D1 eigen factorial | The native-RSA gaps are spectral: E_val ≈ −0.04 for every VLM. CLIP and SigLIP2 eigenvectors are as good as DINOv2's, and PEcore's are better. | [summary](runs/d123_development/20261012-attempt02/summary/d123_summary.json), [figure](figures/d123_development/20261012-attempt02/d123_results.png) |
| D2 flexibility ladder | Reweighting each model's own axes favours DINOv2, widening the gap. Full linear readout closes it, with VLMs slightly ahead. All models prefer α ≈ 2. | same |
| D3 selection intervention | ED-maximizing selection in DINOv2's own space lowers its RSA (O = −0.039, 5/5). CLIP and SigLIP2 show no own-selection effect. | same |

**2026-10-12 23:01 UTC.** The scientist adopted the restated C1, the
extension of C2 to CLIP, the unchanged C3, and the rule restricting
confirmatory categories to those with at least 20 images. The outcome-blind
[OOD exposure review](runs/ood_exposure_review/20261012-review01/exposure_review.json)
passed, finding no evidence of OOD outcome exposure. See the
[log entry](experiment_log.md#2026-10-12-2301-utc--ood_exposure_review--passed-outcome-blind).

**2026-10-13 00:00 UTC — write-once lock written; the OOD opening awaits the
scientist's approval.**

- [ood_confirmation_lock.json](ood_confirmation_lock.json)
  (sha256 `efb07381…`, read-only) binds the evaluator, all settings and the
  pre-lock evidence.
- The pre-lock evidence comprises:
  - the exposure review;
  - coverage;
  - synthetic tests;
  - a bit-for-bit reader check;
  - the dev-test.
- C2 was rewritten post hoc into C2a and C2b. C2b is descriptive, because
  development did not support it.
- See the [lock entry](experiment_log.md#2026-10-13-0000-utc--configuration_lock--revalidation-passed-write-once-lock-written).

Next action: a single procedurally sealed OOD opening
(`ood_confirmation.py --mode confirm`), only after the scientist's explicit
approval is logged.
