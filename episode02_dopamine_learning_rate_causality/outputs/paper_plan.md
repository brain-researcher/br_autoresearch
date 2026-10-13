# EP02 paper plan — From a protocol effect to learning-rule evidence

Design/writing revision: 2026-09-30. No empirical analysis or outcome opening
has occurred. This plan reorganizes the existing contract; it does not enact
new endpoints, model families, thresholds, terminal rules, or access rights.

The subsequent October 6 scientist agreement adopts the
[authenticated article/task primary and report-only pinned-code sensitivity](source_semantics_design.md#adopted-model-choice--october-6-2026).
That variant choice is answered; provider trigger/codebook facts, meaningful
margin rationale and empirical implementation remain unfinished. The agreement
does not adopt optional rival operators or validate a confirmatory decision rule.

The [current source and implementation handoff](source_semantics_design.md)
incorporates the subsequent October 3–6 work: exact unresolved provider fields,
the unsent source request, separate article/pinned-code variants, the already
noncausal 462-label fallback, passing arithmetic components and eleven passing
checks of unadopted rival interfaces. Those checks are preparation rather than
empirical or full-model qualification. The four proposed margins remain
inactive, physical reader separation is unprovisioned and no empirical result
is available. The [execution log](experiment_log.md) is the evidence for that
progress; the results narrative below remains conditional.

## October 7: selected prospective population observation recipe

Under the scientist's delegated judgment, select the following common
observation layer **now, before empirical outcomes**, within the existing
session-binomial, training-only partial-pooling and new-mouse prediction
grammar. The earlier specification did not select a Gaussian mouse effect;
this is the explicit prospective choice, not a recovered source fact.

For fixed mechanism session logits `h_ms`, use
`K_ms | u_m ~ Binomial(n_ms, sigmoid(h_ms + b + u_m))`, with one
`u_m ~ Normal(0, sigma^2)` shared by **all** sessions of mouse m. Mouse effects
are independent across mice. One common population offset b calibrates the
observation level. The caller must supply fixed mechanism logits and cannot
simultaneously estimate another free readout intercept: their additive levels
would not be jointly identified. Mechanism/readout fitting and the actual
trial-to-session probability map still need their own explicit recipe; this
observation layer neither supplies them nor guesses source timing.

[`logistic_normal_observation.py`](code/logistic_normal_observation.py)
implements constrained marginal maximum likelihood using the eight training
mice of each fixed development fold. A caller-frozen finite nonnegative SD
grid includes zero; each point receives the same bounded common-offset solve.
All grid points must pass numerical checks. Choose the smallest SD within the
explicit numerical profile-likelihood tie tolerance of the best training
likelihood. Freeze the bounds, SD grid and numerical tolerances before
development access and use the same recipe for all mechanism families,
folds and seeds. These numerical settings are implementation work, not another
request for permission or a new scientific effect margin.

Every fit returns its complete SD likelihood profile and whether its offset
is on a supplied boundary. Estimates are constrained by those offset bounds
and the finite SD grid; an upper-grid SD estimate does not establish an
interior optimum, unique variance identification or a global continuous-SD
maximum. Selected population hyperparameters are plug-in training estimates.
The new-mouse Gaussian effect is integrated, but uncertainty in fitted
population/mechanism parameters is not a Bayesian posterior or automatically
propagated interval. Report that distinction rather than claiming complete
uncertainty coverage.

The October 9 implementation also supplies the policy's final refit after
recipe lock: `refit_final_observation_model` requires exactly the fixed nine
development controls, the same frozen recipe and an explicit resource callback.
It reuses the fold fitter's constrained profile algorithm and returns a distinct
`FinalObservationFit`, which cannot enter the leave-one-mouse-out prediction
helper. The caller must already have legitimate recipe-lock and reader-release
authority; passing arrays does not create either. This supplies an observation
refit component, not mechanism fitting, source semantics or an empirical result.
Three new focused checks and the six affected existing checks remain unrun for
this source revision.

For a new mouse, integrate the **product** of its conditional session
likelihoods against the one shared Gaussian effect, then take the logarithm.
Do not multiply separately marginalized session probabilities or insert a mean
probability into a binomial likelihood. Normalize by that mouse's total
eligible trials, average the five seed scores, and then apply the unchanged
equal-mouse minimum/median/mean ranking against the constant-rate null.
No held-mouse outcomes enter fitting or prediction construction. Scoring uses
the same frozen quadrature nodes/orders for every outcome and refuses
unresolved numerical agreement; it never refits, chooses a new order or tunes
the distribution to those outcomes. Zero SD uses the exact binomial-product
limit. Explicit zero-trial sessions contribute zero information while staying
in the session roster; every mouse must have positive total support. This
does not relax the separate endpoint requirement of at least 30 eligible
trials in both sessions 1 and 8, or authenticate missing-session semantics.

Six [focused checks](tests/test_logistic_normal_observation.py) are prepared,
**unrun**: exact zero-SD and zero-information limits, shared-effect integration
against independent scalar quadrature, known constrained optima, grouped
heterogeneity and immutable prediction, invalid/unfinished numerical cases,
and the unchanged complete-panel seed/mouse aggregation. No simulated cohort,
recovery sweep, scientific payload or empirical fit is involved. Existing
passed component checks remain reusable.

The coordinator obtained BR advisory rubric `ext_review_dir_70ff8d44da21` at
2026-10-07 09:37:57 UTC. Its repeated-measures, leakage/held-out-selection,
well-posed-parameter and fit-versus-mechanism criteria inform this choice.
It did not review these artifacts, issue a verdict or grant canonical
scientific acceptance. Source codebook/time/trigger facts, data isolation,
four unbound margins and full mechanism/readout fitting remain unfinished.

## The question a reader should remember

What can reward-time dopamine stimulation contingent on licking tell us about
the rule by which behavior is learned? In particular, does a difference
between two closed-loop protocols discriminate competing rules, or only show
that the protocols lead to different acquisition gains?

The scientific tension is between an identifiable behavioral contrast and a
latent mechanism. An algorithm can fit controls and still extrapolate poorly;
different algorithms can also extrapolate to the same qualitative pattern.
Neither the best fit nor a familiar intervention direction settles uniqueness.

## Position relative to prior work

| Prior work | What EP02 must not claim as new | Consequence for this paper |
| --- | --- | --- |
| [Coddington, Lindo and Dudman, Nature 2023](https://www.nature.com/articles/s41586-022-05614-z) | ACTR's adaptive-rate account, calibrated closed-loop manipulation and its published direction. | Source result is a benchmark; the new object is the inference supported by controlled model extrapolation and the release, not rediscovery. |
| [Wilson and Collins, eLife 2019](https://elifesciences.org/articles/49547) | Model recovery and validating a fitted model as general ideas. | Required synthetic recovery is a diagnostic, not itself a new method. |
| [Cone, Clopath and Shouval, Nature Communications 2024](https://www.nature.com/articles/s41467-024-50205-3) | Flexible temporal-representation alternatives to canonical TD and reanalysis of published dopamine recordings. | The finite active TD/ACTR panel cannot be called an exhaustive test of all value-learning theories. FLEX is not silently inserted into that panel. |
| [Mah, Golden and Constantinople, Cell Reports 2024](https://doi.org/10.1016/j.celrep.2024.114840) | RPE/learning-rate dissociations in an expert-rat task. | A naive-mouse result does not settle dopamine's role across tasks or learning stages. |
| [Dopaminergic action prediction errors, Nature 2025](https://www.nature.com/articles/s41586-025-09008-9) | A value-free dopamine teaching signal for stimulus–action associations in tail striatum. | This is a different circuit/task, not a direct replication or a model already tested by EP02's finite panel. |
| [Gong and colleagues, Science 2026](https://doi.org/10.1126/science.aeb0813) | A broad new claim that dopamine/reward magnitude controls learning efficiency or task engagement. | This release does not manipulate reward magnitude across tasks; no such generalization is available here. |

This is a bounded literature position, not an assertion that nobody has done
the exact reanalysis. A standalone paper requires actual results showing a
nonredundant discrimination or identification-limit result. If the only
deliverable is the published directional effect plus better recordkeeping,
it is a reproducibility report, not a new mechanism paper.

Current literature broadens the comparison context, not the executable panel.

## Three claims, three evidential objects

1. **Control prediction.** Leave one of nine control mice out at a time;
   compare whole-mouse predictive performance under the existing equal-budget
   mechanism panel. Never treat sessions, trials or random seeds as additional
   mice. Development model ranking is not intervention confirmation.
2. **Protocol-dependent gain.** For each primary mouse define
   `gain = P(preparatory lick, session 8) - P(preparatory lick, session 1)`.
   The unchanged primary is `delta_rate_raw = mean(gain | stimLick-) -
   mean(gain | stimLick+)`, in probability-gain units. Six and five released
   mice are the biological units. The standardized score is exactly the raw
   contrast divided by the frozen control SD, not a second estimand.
3. **Mechanistic discrimination.** Compare the already-required, locked
   predictions of the target and alternatives. If their predictions overlap,
   a target-consistent pattern is not a mechanism-unique observation. The
   current terminal assesses target pattern consistency; it does not run a
   calibrated all-rival exclusion test.

The protocol is the causal intervention object, not the number of stimulation
events actually realized. Both interventions occur at reward time and depend
on the preceding 750-ms lick state; only `stimLick+` has its documented 50%
session cap. A state-contingent intervention changes exposure as behavior
changes. Conditioning on stimulation count would alter the primary question,
and the previously exposed count diagnostic is permanently excluded anyway.

## Results narrative, conditional on future authorized evidence

### 1. What each implementation can explain without seeing interventions

Show control predictive performance relative to the constant-rate nested
null, with mouse-equal summaries and the existing influence/seed checks.
Report both successful and unsuccessful families, not only the target.
Explain which within-control differences affect the locked extrapolation.
Required model-recovery results describe finite-panel recoverability under
the specified simulations; they do not demonstrate global structural
identifiability.

### 2. Where the frozen interventions separate the hypotheses

Display both arm predictions for every locked family under the shared task
schedule and faithful trigger rules. State which pairs make distinct signs,
which differ only in magnitude, and which agree. Any intervals shown must
describe their actual simulation/fitting uncertainty, not fabricated
confidence in biology. There is no post-audit refit or best-rival swap.

This is a descriptive organization of predictions already required by the
policy. It cannot introduce a tolerance-selected equivalence class, a new
statistical primary, a calibrated rival rejection, or a new terminal label.
If the scientist wants a formal model-discrimination audit, its statistic,
uncertainty rule and evaluator output must be separately agreed and frozen
before any primary outcome access.

### 3. How much the complete-case protocol contrast resolves

Use the one authorized future aggregate evaluator packet to show the raw
contrast and its exact standardized companion, the frozen raw-unit margins,
arm-reference guards, and aggregate influence/replication states. Do not
request or draw individual audit-mouse values: the present output boundary is
aggregate-only. No speculative effect sizes or observed winners belong in
the design figure.

Unblocked enumeration of 462 labels remains a conditional-exchangeability
sensitivity, not the documented within-cohort assignment distribution. The
missing block roster and four excluded animals prevent a design-exact ITT
claim. Even after a design recovery, a sharp constant-shift test is not a
confidence bound for an arbitrary heterogeneous population-average effect.
Controls are development-only, so arm-reference guards are not randomized
causal comparisons against controls.

### 4. What remains indistinguishable or unresolved

Put the behavioral conclusion alongside, rather than inside, the mechanism
conclusion. Show whether its direction was uniquely predicted within the
tested panel. A positive gain contrast can coexist with algorithmic ambiguity.
A nonrejection in eleven mice can be weak information, not evidence of no
effect. A robust reverse pattern challenges the locked target but does not
automatically identify a different mechanism.

## Positive, ambiguous and null stories

| Evidence available in a future valid run | Defensible headline | Not defensible |
| --- | --- | --- |
| Target predicts a distinctive pattern; primary and all target guards pass | Locked rate-gate-consistent predictions survive this cohort and the declared falsifiers; describe separation within the tested panel. | Dopamine's unique learning rule has been discovered. |
| Target and a rival share the observed pattern | The protocol contrast is compatible with multiple control-qualified rules. | All rivals are rejected because the target ranked first on controls. |
| Positive arm contrast, failed arm-reference/influence guard | The released protocols differ on acquisition gain, without the full target pattern. | Bidirectional causal enhancement/suppression relative to controls. |
| Valid reverse-direction terminal | The primary pattern is inconsistent with the locked target. | Universal RPE mechanism established. |
| Valid nonrejection | The small cohort does not resolve the prespecified contrast. | No dopamine effect, exact equivalence, or absence of learning. |
| Semantics or design support remain unrecoverable | The release does not identify the intended endpoint or design-exact causal quantity. | Biological refutation based on a technical limitation. |

## Figure sequence

1. **Concept and inferential boundary.**
   ![EP02 conceptual design](ep02_conceptual_question.png)
   Three panels: fit controls and lock predictions; show reward-time lick/no-
   lick protocol triggers; separate a protocol contrast from unique learning-
   rule identification. Schematic only. The cards are examples of alternative
   explanations, not an observed tie or exhaustive family inventory.
2. **Control fit versus intervention extrapolation — planned evidence.**
   Whole-mouse development prediction summaries beside the complete locked
   arm-prediction panel. No plotting of sealed primary mice. Source values
   must come from an authorized run, not the conceptual figure.
3. **Aggregate protocol result and inference scope — planned evidence.**
   Raw/standardized aggregate contrast with the actual frozen decision fields
   and falsifier states. Distinguish design-exact inference, if recovered,
   from exchangeability sensitivity. No individual audit data.
4. **Unresolved mechanisms — planned reporting, not an added experiment.**
   Qualitative agreement/disagreement of existing locked predictions with
   the permitted aggregate packet. Required recovery and ablation results
   supplement the figure. No statistical rival-exclusion claim without an
   explicit pre-outcome amendment.

The high-amplitude cohort is a supplementary boundary diagnostic only, with
its own six-session endpoint. It cannot salvage the primary result or become
a second mechanism discovery.

## Honest paper ceiling and current next step

The present design is most plausibly a focused computational-neuroscience
reanalysis/methods note. Its value would be a concrete demonstration of what
intervention extrapolation does or does not discriminate, with biological
group independence and causal-design limits intact. It is not presently
enough for a broad dopamine discovery paper, and merely renaming the question
does not improve novelty.

Before empirical work, resolve the existing endpoint codebook and assignment/
attrition support through authorized documentation, then decide the causal
scope that can be frozen. Margins and the endpoint-faithful binding rule need
their existing pre-outcome scientific decision. No new data, contact with
authors, candidate scoring, lock or audit is authorized by this writing task.
The active access boundary remains closed. ASTRA lists existing stages; the
execution log records that none has launched.
