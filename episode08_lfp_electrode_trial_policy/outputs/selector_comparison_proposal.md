# EP08 next comparison: training-derived electrode selection

Prospective successor to the completed random-subset baseline, 2026-10-03.
The coordinator selected these five substantive selector/control panels under
the existing delegated development scope. They are **unrun**. The baseline
showed positive session means but nonmonotonic channel-count curves; this
comparison asks whether training-derived selection improves the same Q.
It does not activate the named removal-value, direction-value or coupling
endpoints and does not open any held session.

## Common information and fit

Preserve all eight development sessions, their existing 40 evaluation trials,
nested 32/64/96 fitting trials and four fitting-only folds. The inputs remain
three ten-bin means per physical electrode (LMP, 100–200, 200–400 Hz); the
target remains all native M1 population columns. The selected electrode counts
remain 8/16/32/96. Use the same five decoder penalties, training-only population
SD convention, true-target ridge fit, training-mean-null Q, equal 12-cell and
session/animal aggregation, and 0.001-Q search-progress resolution as the
completed baseline. No new success margin is selected.

Each selector sees only the rows available to its fit. Recompute selection,
centering and scaling in each inner fitting fold; its validation rows influence
only decoder penalty selection. Recompute selection on the complete allowed
outer fitting prefix for the final fit. This prevents validation targets from
entering a supervised feature selector before cross-validation. Nested channel
prefixes are shared across the three scarce electrode counts. Source data are
still read only by the trusted evaluator; selectors/fitter receive array
copies, direction labels and deterministic seed material, never a source path
or evaluation array. The same-UID process boundary is not an OS sandbox.

## Exact prospective selectors

Let X and Y be the currently permitted fitting rows, each column centered and
scaled by that fitting pool's population SD; exactly zero SD becomes one.
There are n rows, U target columns, and three adjacent feature columns per
electrode. All ranks use descending score; exact ties use ascending physical
electrode ID. There is no tuning of selector shrinkage: its fixed dimensionless
ridge level is one, after this standardization, independently of the decoder's
cross-validated penalty.

1. **Direction repeatability.** For each LFP feature, use the balanced eight
   direction groups of size m in the current fitting pool. Compute
   `MS_between = m*sum_d(mean_d-mean_all)^2/7` and
   `MS_within = sum_i(x_i-mean_direction(i))^2/(n-8)`.
   Define `tau2=max((MS_between-MS_within)/m,0)` and repeatability
   `tau2/(tau2+MS_within)`, with the exactly zero/zero case zero. Average the
   three feature repeatabilities per electrode. This is direction-repeatable
   LFP structure, not spike-target association, broadband artifact quality,
   line-noise suppression or within-direction individual-trial information.
2. **Marginal target association.** For each electrode block e, let
   `G_e=X_e'X_e/n`, `R_e=X_e'Y/n`. Rank
   `trace[R_e'*(I_3+G_e)^(-1)*R_e]/U`. This uses target linkage but ignores
   redundancy between retained electrodes. It is a regularized linear proxy,
   not an information-theoretic quantity.
3. **Target-agnostic diversity.** Greedily add the electrode maximizing
   `logdet(I_3 + X_e' A_S X_e/n)`, where
   `A_S=(I_n+X_S X_S'/n)^(-1)` and S starts empty. This is block D-optimal
   coverage of the observed standardized LFP, including any unhelpful noise.
   It uses no spike targets and does not stand in for unavailable geometry.
4. **Conditional target association.** With the same A_S, greedily maximize
   `trace[R_e(S)'*(I_3+G_e(S))^(-1)*R_e(S)]/U`, where
   `G_e(S)=X_e' A_S X_e/n` and `R_e(S)=X_e' A_S Y/n`.
   This is the exact improvement in the fixed regularized fitting objective
   from adding the three-column block. It is not an unbiased prediction of
   later removal loss or a newly activated named explanatory score.
5. **Target-permuted conditional selection.** Repeat selector 4 with one
   global whole-trial permutation of multivariate Y within each allowed
   fitting pool. All target columns move together; X and direction labels
   stay fixed. This breaks both direction-level and within-direction target
   linkage in the selector. The decoder and its CV errors always use the true
   targets. This specifically tests whether the selector's target linkage
   matters; it is not a permutation null for the final prediction score.

For control replicates r=0..4, Python3.12 `random.Random`, seed version2, uses
`ep08-selector-target-permutation-v1|20261003|<filename>|<training_total>|<r>|<fold>`;
fold is 0..3 or `outer`. Shuffle ascending local row indices once; reuse that
permutation across the nested electrode counts. Retain the seed convention and
selected identities. Deterministic selectors are fitted once, not five times
masquerading as replication; average the five control permutations inside each
control cell. Seeds and counts are not separate scientific evaluations.

At all 96 electrodes selection is vacuous. Fit the unchanged ordinary ridge
model once per training amount, on the common ascending channel order, and
reuse its exact model and Q across all five panels and control replicates.
Do not recompute, retune or select an all-electrode winner. The ordinary
baseline used a column permutation of the same features; mathematical ridge
prediction is invariant to that ordering. Its saved outputs remain unchanged.

## Outputs, accounting and interpretation

Freeze all five panels' fits for a session before reading that session's
reserved evaluation arrays. Keep all cell/session/animal values, outer retained
channel identities, selected decoder penalties, fit counts and selection costs.
Report paired Q gains over the existing fixed baseline; baseline results are
only evaluator/reporting inputs and cannot reach a selector or fitter. A
complete all-session panel charges one valid evaluation, so completing this
batch charges five (the shared total would become six). A failed partial batch
retains its exposure and resources; it cannot be called five completed panels.

The expected workload is 81 scarce-electrode models plus three shared full-
channel models per session, each with 20 CV fits and one refit: 14,112 decoder
fits across eight sessions. Greedy selectors additionally compute training-only
rankings; their numerical implementation uses algebraically equivalent rank-
three updates. Selected source-array volume remains about 30.1 MB plus guides.
Proposed coordinator bound is one CPU, 2 GiB, 15 minutes, numeric threads one;
actual selection cost is unmeasured. Root selects the execution command after
review. No automatic retry, threshold relaxation, cell dropping or submission
is authorized by this document.

A useful selector result would motivate further matched model/temporal tests,
not prove a joint acquisition principle. The next competing explanation is
**reach-direction means**: fit a direction-only target null on training rows,
then compare neural residual prediction with that same training-only null,
with the complete population target and common evaluation support preserved.
Within-direction trial-pair permutation would distinguish individual-trial
signal from direction-level predictability. Those are later prospective
comparisons, not outcomes of this batch.

Further stages remain open to development feedback: temporal summaries within
the existing window; reduced-rank/PLS and nonlinear matched-decoder panels;
session-cross-fitted transferable selectors without assuming neuron identity
correspondence; and trial-coverage/acquired-only replay comparisons. Geometry
and raw artifact/line-noise controls require actual compatible source metadata
or measurements and cannot be invented from electrode numbers or stored power.
Any replay branch must declare its initialization, legal action/exhaustion
rule, total observed rows/channels and untouched future-input boundary before
scoring. Named direction/removal/coupling outcomes still require their own
prospective specification. No complete 40-panel list is manufactured: later
valid evaluations must answer distinct hypotheses or required alternatives,
with the shared 40–112 allowance, 20-evaluation patience, applicable coverage,
two adaptive cycles/two challenger decisions and at least 40% post-coverage
alternative checks retained. Hyperparameter values, seeds, cells and repeated
fits are never used to inflate that count.

## Execution record boundary

The coordinator executed this fixed recipe in job46430982 step1 on October3,
2026; its exact report is
`runs/development_search/selector_comparison_attempt001/development_selectors.json`.
All three focused tests and all five panels completed. Five valid evaluations
are charged, making six with the baseline. Earlier unrun wording above records
the prospective state, not current status. See the experiment log for actual
scores, resource use and direction-confounding interpretation. This completion
does not activate the named explanatory endpoints or open held sessions.
