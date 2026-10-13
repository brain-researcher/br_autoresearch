# One possible next question after the constant W/HW blend

**UNADOPTED scientific assessment, 2026-10-04.** This document authorizes no
implementation, scoring, source access or launch. It uses released complete
evidence through comparison33 only. H2W and intersection I/R pilot scores
remain closed and were not consulted. At this evidence boundary, the blend
is the comparable incumbent with primary gain0.0026788544310999856 and
patience0. These are a dated starting point, not proposed future ordinals or
incoming counters. Root owns any later adoption and live stopping decision.

## Recommendation

If the already adopted work supplies no stronger reason to change direction,
the most defensible additional family is **one fixed linear elapsed-time
allocation of the existing W/HW prediction blend**, preserving its average
half-and-half weight. Ask whether the benefit of history is distributed
differently across the fixed reach window. This is a small, falsifiable
extension of a useful completed comparison. The evidence does not establish
that its answer will be positive or that it can close the large practical
performance gap.

Do not launch another unweighted quadratic LMP tensor under a new name.
I also do not currently recommend a quadratic-by-four-lag tensor or a robust
spatial median as the next family. The former expands a temporal flexibility
that previously hurt primary prediction; the latter discards the electrode
contrasts retained by the strongest models without observed evidence that
channel outliers are the present limitation. These are relative priorities,
not scientific rejections of either general approach.

## What the completed temporal recipe actually tested

The [adopted temporal proposal](primary_temporal_control_proposal.md),
[implementation](tools/primary_temporal.py), and
[complete temporal result](runs/adaptive_search/primary_temporal_control_summary_attempt001/primary_temporal_control.json)
already evaluated `lmp_timepoly2_ridge_v1`. It used every original current-bin
LMP channel, fitting-only channel standardization, and the three fixed
discrete orthonormal columns1, centered time, and centered time-squared on
bins0:14. Its3E product columns were fitting-centered and supplied a new
ridge head. The five original penalties were selected from the other fifteen
sessions. Its eight whole-trial training derangements preserved direction,
time and feature covariance and used the paired-selected penalty.

That complete candidate produced primary0.001177898 and identity0.003951206.
Its primary mean was−0.000424661 in Mihili and+0.002780457 in Chewie, with
only1/6 and9/10 positive sessions. Relative to stationary full-channel LMP,
primary differences were−0.001439402 and−0.000079571 in the two animals.
The correctly paired training contrast passed, so failure cannot be summarized
as no usable training alignment. It was a failure of primary performance for
that particular larger temporal map. The
[23:32 closeout](experiment_log.md#2026-10-03-2332-utc--adaptive_search-temporalcontrol-pair-complete-primary-remains-unsupported)
records the rounded values above.

Thus elapsed-time interaction is already tested, not missing initial coverage.
Adding the current-only W prior to that exact tensor would be distinct, but
the existing results give limited reason to expect the prior to repair its
Mihili loss. Expanding to12E weighted four-lag product columns additionally
raises estimation burden. Neither is the cleanest next response to the
released evidence.

## Why a constrained prediction allocation is different

The [complete blend](runs/adaptive_search/primary_history_blend_summary_attempt001/compact_evidence.json)
selects shared lambda1 and historical weight0.5 in every outer fold. Its
primary0.0026788544310999856 exceeds W by0.0005040036032114077 and HW by
0.0003375999579230016, while its identity0.004559394718773168 is below HW by
0.0010549892485697313. Both original practical margins still fail. The blend's
own animal and fixed-omission gains are positive, but its advantage over HW
is fragile in Chewie: only3/10 sessions improve and omitting20150630 changes
that animal's primary contrast sign.

This establishes a prediction tradeoff between the two saved maps, not its
time course. A constant mixture could be averaging an early/late tradeoff;
it could equally be doing ordinary error averaging with no stable temporal
structure. No per-bin score or trajectory was opened to choose between these
possibilities. Testing the latter possibility directly is more focused than
adding an unconstrained time-specific neural coefficient for every channel
and lag.

The [mean-LMP result and multichannel rationale](primary_multichannel_proposal.md)
also favor retaining the existing spatial maps for this next question. The
mean-LMP primary was approximately−8.77e−7. A median is not algebraically that
mean and remains allowed, but common referencing does not by itself show
that median aggregation would retain the useful residual information. No
observed outlier or contamination diagnosis motivates replacing the full
electrode representation now.

## A finite form suitable for a later adoption decision

Reuse only the complete original W/HW five-penalty predictions and their exact
original trials, native XY, direction/time baseline B and common denominator.
For t=0,...,14 define u(t)=2t/14−1 and

`alpha_s(t)=0.5+s*u(t),   s in {-0.5,-0.25,0,0.25,0.5}`.

At the same original shared lambda, predict

`P_s(t)=(1-alpha_s(t))*P_W(t)+alpha_s(t)*P_HW(t)`.

Every weight is between0 and1; its mean across the fifteen fixed bins is0.5.
The family changes the temporal mixing schedule while holding its average
mixing coefficient fixed. This does not fix the average historical prediction
contribution: `u(t)*(P_HW(t)-P_W(t))` can have nonzero mean and uneven amplitude.
It is not an isolated timing or constant-capacity contrast. Both ramp
directions are allowed prospectively; the
evidence does not select an early-history or late-history story. There is no
knot, interval, polynomial-degree, per-animal or per-session history-length
search. Full fifteen-bin predictions remain the scored endpoint. Both
components include the same B, which is added exactly once by the convex
expression; no standalone elapsed-time predictor is introduced.

Select shared lambda and slope jointly using the unchanged animal-balanced
primary objective from only the other fifteen sessions, and apply one choice
to every direction/trial of the held session. A later formal proposal should
freeze exact ties as stronger lambda, then smaller absolute slope, then a
declared deterministic sign order. There are25 parameter cells, not25 new
scientific recipes. The mean0.5 constraint is an explicit adaptation to the
released blend, so the entire assessment remains exposed development.

The zero-slope cell must reuse the completed half-blend arithmetic exactly.
For endpoint bins whose coefficient is0 or1, reuse the corresponding literal
W/HW values. Preserve the qualified saved-target/B layout and exact component
score replay. Score each resulting combined prediction itself; averaging
component SSEs would answer a different question. No source read or new
within-session neural fit is needed.

This is a time-dependent convex ensemble of the same two completed pipelines,
within GOAL's ensemble choice; it is not the previous3E LMP tensor refit.
Neither unchanged component penalties nor an equal average mixture weight
establishes matched effective df for the selected ensemble.

## Controls and interpretation

Keep the exact within-direction wrong-trial evaluation readout. Report full
same-D contrasts against the frozen original blend, W and HW, with both animal
means, every session, fixed-prediction omissions and the unchanged0.01/0.005
margins. Report every outer-selected slope and sign alongside lambda and the
slope-reversal contrast. Compare shared-lambda anchors descriptively where
useful; no anchor
is refit and no extra threshold is introduced.

A particularly direct timing reference is **slope reversal**: at the paired
outer-selected lambda and slope s, evaluate−s without retuning it. Reversal
preserves the mean, range and linear smoothness of the mixture weights while
exchanging early and late allocation. A stable held-session advantage over
the reversed schedule would support the selected allocation policy. If slopes
change sign across folds, it does not support one stable early-versus-late
ordering. A selected slope describes model allocation, not a measured
physiological time course. It would not identify neural phase, causal timing,
contamination or a motor-population mechanism. No phase-randomization claim is appropriate
for these stored features.

Treat this reversal as a descriptive contrast using already computed parameter
cells, with no automatic additional scientific or alternative-check charge.
It is not a literal training-pairing null. The existing HW capacity control
and earlier tensor training control do not automatically satisfy every
required finalist control for a newly selected mixture.

If every selected prediction equals the completed original blend panel exactly,
record a duplicate and zero new candidate depth. The same literal-panel rule
applies if another preserved candidate is reproduced; score closeness is
insufficient. If H2W has completed before adoption, include its fixed selected
predictions in the exact duplicate checks and same-D references, without
substituting H2W into this two-component definition. A distinct complete policy
contributes at most one candidate and can reset patience only through a strictly larger complete
comparable primary result against the incumbent actually in force then.

## Decision boundary

H2W and I/R remain the authorized next computations. I/R is nonpromotable
support sensitivity and cannot reset the comparable incumbent by design.
After their complete release, root should decide whether this narrow timing
question still has priority; neither a failed H2W result nor the availability
of more parameter cells automatically authorizes it. If adopted, a separate
prospective specification and bounded saved-artifact resource pilot should
precede a full panel. This assessment selects no run bounds or executable
command.

Retain all existing coverage, the post-initial alternative allocation,
scientific-depth and resource limits, and the adopted conditional stopping
rule. Patience12 is not an isolated hard ceiling divorced from required
coverage. Conversely, this optional proposal creates no new coverage
obligation or reason to continue past a valid stopping condition. No result
here changes the missing future-session evidence or supports promotion.
