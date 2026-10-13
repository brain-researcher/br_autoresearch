# Independent component checker interface

Author the reference implementation without reading `actr_port.py`, its
tests, its qualification outputs or expected values. Use the source equations
and this input/output interface only. Select independent synthetic cases.
The reference must not import the primary implementation. Comparing outputs
after independent authorship is allowed and should be disclosed.

Permissible local MIT oracle files, all at
`outputs/code/author_oracle/` and commit
`3627746957975632cddb05bf8f6b26a4b6f901b6`:

- `dlRNN_engine.m`: recurrent and eligibility equations.
- `dlRNN_Pcheck_transfer.m`: state_simple/pass-thru behavioral plant.
- `dlRNN_train_learnDA.m`: calibrated stimulation and learning updates;
  use the section from eligibility selection through running baseline update.
- `LICENSE`: source reuse permission.

Gaussian mathematical oracle only, no copied TONIC source:
https://github.com/DudLab/TONIC/blob/f78fdbf4bca0e5c16859f885550b4e274fc3459d/CreateFunctions/TNC_CreateGaussian.m

Expose `evaluate_case(case: dict) -> dict` in
`outputs/code/independent_actr_checker.py`. Inputs and outputs must be ordinary
JSON-compatible lists, scalars and mappings. No file/data access is necessary.

| `case.kind` | Other input fields | Return fields |
| --- | --- | --- |
| `gaussian` | `mu`, `sigma`, `duration`, `dt` | `values`: sample array |
| `recurrent` | `network`, `stimulus` `[channels,T]`, `tau_ms`, `perturbations` null or `[N,T]` | `output` `[T]`, `hidden` `[N,T]`, `eligibility` `[T,N,N]` |
| `plant` | `activity` `[T]`, `reward` `[T]` or null for source default, `scale`, `transition_uniforms` `[T]`, `bout_offsets` integer list sufficient for qualifying bouts | `licks`: one-based source tick indices; `state` `[T]` |
| `stimulation` | `variant`, `arm`, `base_beta`, `error_r`, `pre_reward_lick`, `session_trials`, `prior_stimulations`, `uniform` | `beta`, `triggered` |
| `learning` | `network`, `eligibility` `[N,N]`, `error_r`, `baseline`, `beta`, `eta_j`, `tau_trans`, `saturation` | updated `recurrent`, updated `inputs`, updated `baseline` |

`network` contains `recurrent` `[N,N]`, `inputs` `[N,channels]`,
`feedback` `[N]`, `original_support` `[N,N]` booleans,
`bias_indices` zero-based list and `bias_value`. There is one output unit,
index0. Returned `hidden` means the pre-tanh activation `hidden_x`, not rates;
`output` is that output unit's tanh activation. For the explicit partial
qualification, activation is tanh and the
eligibility callback is cubic; authenticating callbacks against the unavailable
initialization workspace remains outside this component check. Source
activation averaging uses0.33 and timestep1 ms. An explicit zero perturbation
tape still selects the learning branch; null selects the unperturbed branch.

The two `variant` strings are `pinned_code_30ms_set4` and
`article_25ms_times2`. `arm` is `control`, `stimLick_minus` or `stimLick_plus`.
The pinned variant uses the exact calibrated source switch. The article
variant multiplies base beta by2 on no-lick trials for minus and lick trials
for plus; the plus variant permits only the first `floor(session_trials/2)`
qualifying trials. That cap ordering is an explicit synthetic scenario, not
verified online-task semantics. Neither variant or scenario is selected as
the production model by this check.

No expected fixture outputs, empirical values, proposed margin or desired
scientific direction is supplied. Report equation agreement at numerical
precision and discrepancies. This is not full800-trial parity, model recovery,
calibration, endpoint verification, or permission to open outcomes.

## Source-domain clarification after the first independent replay

The source-derived stimulation input `base_beta` is finite and at least1:
the pinned trainer line50 defines `DA_trans` as3 times a cumulative positive
Gaussian, and line259 adds1 to two entries of that array. The raw switch
algebra can be evaluated below1, but those inputs are outside the source
model and can produce a negative multiplier after the0.75 subtraction.
The main port therefore retains its source-domain rejection and the separate
nonnegative learning-update guard.

The first independently selected39-case suite omitted this domain distinction;
six original cases use0.4. Preserve them and the first failed replay as history.
Subsequent reports classify those six as explicit out-of-source-domain
rejections, not numerical-equation passes. Any source-valid boundary cases
are independently authored additions, kept separate from the original suite.
