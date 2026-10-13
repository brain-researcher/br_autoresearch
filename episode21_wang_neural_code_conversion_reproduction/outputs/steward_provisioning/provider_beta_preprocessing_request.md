# Provider beta preprocessing question — draft only

Our fixed adapted neural-code-conversion analysis uses regular LAION-fMRI
images and must exclude every OOD contribution from response-derived
normalization/selection. Training transforms also must exclude shared tau
variant 0 test responses. We have prepared independent-localizer native masks.

We authenticated the frozen sub-01/ses-01 GLMsingle processing metadata. It
lists12 tedana BOLD runs,1,044 trial outputs, GLMsingle1.2 TYPED with HRF
library/GLMdenoise/fractional ridge enabled, and a mean BOLD volume across runs.
The matching trial table includes36 OOD rows. This documents a mixed input
context but does not identify any role exclusions applied before normalization
or fitted preprocessing.

Could you supply the actual processing-wrapper revision/options and input
selection rules establishing the timepoints used for percent-signal-change
scaling, the session/trial reference used by internal normalization, and the
pools used for HRF/noise-PC/ridge/autoscaling? Please identify whether OOD or
shared tau-test responses contributed to any of these stages, and whether an
eligible regular-only, training-reference derivative is available. We need
specifications/provenance or an eligible-release path, not BOLD values, model
arrays, R²/quality metrics or OOD derivatives.

No contact has been made on the scientist's behalf. This request remains an
unsent draft; standard GLMsingle behavior is not substituted for actual-run
provenance.
