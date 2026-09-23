# EP05 — adaptive condition-general LFP kinematics

## Status, protocol, and exposure boundary

This is the current local episode contract, governed by
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). It does not
authorize data access, compute, submission, reward, or canonical mutation.

The Dryad outcomes have been accessed previously and are development evidence.
Prior technical work did not establish scientific acceptance and cannot seed
the adaptive trial order, thresholds, or winner. The current contract inherits
no action or decision from that work.

## Scientific question

Which compact representation of the released, post-processed M1 LFP features
predicts trial-specific continuous hand velocity on held-out movement
directions beyond a training-derived direction-by-time baseline, and which
design choices remain stable when the global pipeline policy is selected using
entire held-out sessions rather than favorable trial folds?

This asks about predictive information in released features, not causal neural
coding, online decoding, raw LFP mechanisms, or population prevalence. A
pipeline policy (feature construction, history, model class, regularization
rule) is shared across sessions; fitted coefficients remain session-specific.
Cross-session transfer of fitted mappings belongs to EP07, not this episode.

## Reproduction infrastructure is not search

Before any adaptive trial, run a fixed publication-fidelity suite for the M1
spike-latent/LFP profile and M1 X/Y velocity decoder using the pinned paper
code, best-recoverable dependency boundary, and synthetic fixtures. Its output
is `pass`, `mismatch`, or `technical_failure`.

- Compatibility work, scheduler retries, and exact reproductions never count
  toward the 28 scientific trials.
- Neural outcomes may not be used to tune the reproduction target or tolerance.
- A mismatch is a reportable scientific limitation and blocks a positive
  adaptive candidate; it is not permission to optimize the benchmark.
- Search begins only after the source is technically evaluable and the
  reproduction result is frozen.

## Evidence roles and independence

- **Adaptive development/selection:** all structurally eligible M1 sessions in
  the historical Dryad release. The corpus is outcome-exposed from the earlier
  EP05 work and therefore exploratory. Candidate policies are assessed with
  nested whole-session outer folds; within each session, complete trials and
  entire directions remain grouped.
- **Selection unit:** one session estimate after collapsing X/Y, directions,
  trial folds, channels, and time bins. Animals are the biological replication
  level; Chewie implants remain nested in one animal.
- **Prospective audit:** compatible future whole sessions whose neural outcomes
  and comparison scores are sealed until lock. New sessions from an existing
  animal test session robustness; a new animal is required to strengthen
  biological generalization. No such data are claimed to exist here.

Randomly withholding trials or renaming sessions from the exposed Dryad
release does not create independent confirmation.

## Bounded scientific operator grammar

Every scientific trial is one declarative, session-invariant pipeline from:

1. **LFP block:** LMP; one authenticated stored power band; a registered
   low-frequency, high-frequency, or all-band block; or LMP plus one registered
   band block. Stored band identities come from guides, not numeric columns.
2. **Channel/electrode transform:** fixed matched-electrode mean/median,
   train-only channel PCA, train-only reliability weighting, or deterministic
   top-`k` selection from a finite `k` schedule using training trials only.
3. **Temporal representation:** current bin or causal-looking historical
   windows from a finite lag schedule. Because source features may contain
   offline/zero-phase processing, every result remains offline regardless of
   lag sign.
4. **Kinematic target:** X/Y velocity residual after a training-only cyclic
   direction-by-elapsed-time baseline; registered speed or acceleration
   targets are secondary diagnostic branches and cannot win the primary
   search.
5. **Feature transform:** train-only standardization, PCA, PLS, or a registered
   tensor product of LFP block and elapsed-time basis.
6. **Predictor:** ridge, elastic net, reduced-rank regression, PLS regression,
   linear kernel ridge, or RBF kernel ridge with a finite capacity schedule.
7. **Ensemble:** convex blend of at most two already evaluated complementary
   pipelines; weights are learned inside training sessions/folds.

Models must use the same eligible trials and score full velocity after adding
the neural residual prediction back to the baseline. Negative `R2_SSE` values
are retained. Arbitrary windows, per-session winner selection, test-direction
scaling, bin-level random splits, foundation-model training, and arbitrary code
mutation are forbidden.

## Objective and constraints

For each session, compute held-out-direction
`delta_R2 = R2_SSE(candidate full velocity) - R2_SSE(direction-by-time baseline)`
on identical complete trials. The primary objective is the animal-balanced
mean of session-level `delta_R2`, with the exact within-animal aggregation and
minimum practical gain frozen before search.

A promotable policy must:

- have a positive prespecified summary in both eligible M1 animals and the
  frozen majority of their eligible sessions;
- survive a within-direction whole-trial evaluation-pairing null with all
  fitted transforms held fixed;
- remain positive under leave-one-session influence and matched trial/channel
  budgets;
- show an increment beyond behavior-only and elapsed-time-only baselines;
- pass temporal-offset and feature-label negative controls; and
- not rely solely on 100--400 Hz power or disappear under the registered
  spike-bleed-through sensitivity.

Random-trial performance is a within-condition ceiling. Publication
`r_corr_squared`, phase atlases, speed/acceleration, and reduced-resource curves
are diagnostics and cannot rescue a failed held-out-direction objective.

## Multi-stage adaptive loop

1. **Infrastructure gate:** authenticate source/code, freeze eligibility and
   event conventions, pass synthetic index/leakage/metric fixtures, and freeze
   the publication reproduction result without search.
2. **Session-role freeze:** create immutable animal/implant/session manifests
   and nested whole-session folds. Within each outer session, all transforms
   and tuning use only the remaining sessions and that session's permitted
   training trials/directions.
3. **Coverage stage:** run direction-by-time, LMP, canonical low/high bands,
   linear regularized, reduced-rank, and one nonlinear anchor before adaptive
   exploitation.
4. **Adaptive stage:** make one mechanism-led operator change per trial;
   append hypothesis, parent, full configuration/hash, session predictions,
   objective, constraints, runtime, and failure reason. Maintain a nonterminal
   incumbent and complexity/robustness Pareto archive.
5. **Successive fidelity:** early pruning may use a frozen subset of inner
   development sessions. Every finalist is rerun across every eligible
   development session, both animals, all held-out directions, fixed seeds,
   and mandatory nulls.
6. **Stress stage:** incumbent plus at most three challengers undergo
   leave-one-session and leave-one-animal description, trial/channel matching,
   low-frequency-only, spike-bleed-through, temporal-offset, and target
   residualization ablations.
7. **Configuration lock:** choose one global policy, then hash source/split
   manifests, code, environment, operator DAG, feature guides, windows, model
   rule, thresholds, seeds, and output schema.
8. **One-shot prospective audit:** run the locked policy on all sealed audit
   sessions once. Session-specific coefficients may be fit under the locked
   training recipe; no policy choice, threshold, exclusion, or candidate may
   change after any audit score is visible.

The incumbent cannot terminate the loop early. Minimum coverage, falsifiers,
stress tests, patience, and audit readiness are separate gates.
At least 40% of valid post-coverage trials must be falsifiers, ablations,
negative controls, influence guards, synthetic recovery, or direct
replications. At least two outcome-adaptive successor cycles and two recorded
incumbent/challenger decisions are required before lock.

## Mandatory falsifiers and ablations

- fixed publication-fidelity result and method fixtures;
- direction-by-time and elapsed-time-only behavioral baselines;
- within-direction whole-trial evaluation-pairing permutation;
- whole-trial LFP/behavior shuffle and registered temporal offsets;
- LMP-only, each registered band block, low-frequency-only, high-frequency-only,
  and all-band ablations;
- matched electrode/channel and trial-count analyses;
- same-electrode/unit-intersection spike-bleed-through sensitivity;
- train-only target-residualization versus full-target fit;
- leave-one-session influence and both-animal sign report; and
- capacity-matched random features or phase-randomized surrogate permitted by
  the released representation.

A failed mandatory falsifier blocks promotion rather than becoming another
optimizable metric.

## Budget and stopping

- minimum valid scientific trials: **28**;
- maximum valid scientific trials: **64**;
- patience after the minimum: **12** valid trials without material constrained
  primary improvement;
- CPU ceiling: **2,000 core-hours**;
- GPU ceiling: **0 GPU-hours**;
- wall-clock ceiling: **120 hours** from the first scientific trial;
- finalists: at most **4** including the incumbent;
- audit openings: **1**;
- maximum parallel CPU cores: **32**;
- per-trial memory ceiling: **128 GB**;
- scratch-storage ceiling: **750 GB**.

Reproduction, structural QC, synthetic fixtures, and exact infrastructure
retries are ledgered separately and do not inflate trial depth. The search may
stop before 28 trials only for resource exhaustion, technical impossibility,
or policy violation.

## Terminal classes

- `candidate_ready`: the reproduction gate passes, the locked global policy
  meets every development constraint, and it passes the prospective one-shot
  audit.
- `closed_no_candidate`: technically valid deep search finds no feasible
  policy or the locked policy fails audit.
- `search_exhausted_no_audit`: search completes on the exposed public corpus
  but no sealed future whole-session audit exists.
- `technical_failure`: source authenticity, guides/events, publication
  compatibility, session eligibility, or executable scoring cannot be
  established after bounded recovery.
- `policy_violation`: outcome-dependent benchmark tuning, session cherry
  picking, audit leakage, undeclared operators, or post-audit adaptation.

For canonical outer status, `candidate_ready` maps to `candidate_ready`;
`closed_no_candidate` and `search_exhausted_no_audit` map to
`closed_no_candidate`; and `technical_failure` or `policy_violation` map to
`technical_failure`.

## Claim boundary

Development success supports an exploratory statement about stable predictive
information within this released M1 corpus. A successful fresh-session audit
supports the locked policy's session robustness only for represented animals
and task; population-level, cross-region, real-time, raw-LFP, causal, and
cross-session weight-transfer claims remain out of scope. A new-animal audit
is required before extending biological generalization.
