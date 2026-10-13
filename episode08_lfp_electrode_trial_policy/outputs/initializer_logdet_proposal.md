# Can target-free residual-input diversity select useful electrodes?

Prepared after34 completed comparisons; **root adopted at2026-10-04
06:47UTC** after root and independent scientific review passed. The fixed
implementation and bounded developmental run are authorized; held sessions
and named explanation endpoints remain closed.
The choice was recommended from the released32-comparison evidence before
high-power scores were released. The subsequent spectral result does not
choose its objective, scale, stopping rule or comparison menu.

## Scientific contrast and existing authority

Initializer target correspondence has shown modest predictive value, but
neither raw-X direction repeatability nor the target-linked greedy conditional
proxy beats the original marginal selector in both animal means. Test a
separate explanation: can diversity of the **direction-residual LFP inputs**
select useful electrodes without population-target values in the ranking rule?

The candidate is `initial_residual_logdet_v1`. It selects groups by regularized
input-volume gain, not repeatability across direction means and not target
prediction gain. All targets are still acquired for true-target decoding;
target-free selection does not imply target-free calibration or lower target
cost. A gain would support this fixed selection policy, not a removal-value
predictor, biological complementarity, causal information or sensor-dependent
next-direction benefit.

The open policy explicitly permits acquired-only redundancy/log-determinant
selection with prospective target, information, decoder and exposure choices.
The named electrode-removal, next-direction and coupling endpoints remain
unactivated. This proposal uses only the existing prediction Q and existing
resource cells; it introduces no explanatory endpoint or acceptance margin.

## Fixed support, representation and information views

Keep all eight original development sessions,96 physical electrodes,96 fitting
trials and40 reserved evaluation trials/session, original within-direction
queues, native-unit target means and three native LFP features per electrode.
Preserve ten-bin150–450ms means, retained E=8/16/32/96 and training
n=32/48/64/80/96: the same20 equal cells. No new bands, windows, target pooling,
trials, exclusions or held access.

Use the original32-row initializer and four exact24-row inner fitting views.
For each view, compute each feature's eight direction means from that view
only. Subtract its direction mean, then subtract the residual column mean and
divide by the residual **population** standard deviation. Exact zero SD becomes
one and its column remains present. This is the X part of the qualified
residual transformation; no Y value, target-derived normalization or decoder
state enters the selector API. Do not whiten within electrodes, weight bands
or use an outer view to prepare an inner view.

Let X be that standardized residual matrix with n=24 or32 and exactly288 real
columns, three adjacent coordinates per physical electrode. The implementation
must assert288 columns; the candidate returns physical-electrode indices.
All five orders/session are frozen before any post-initial source row. The
outer ordering stays fixed through subsequent acquisition; only retained
three-coordinate channels are read on future trials. There is no later
reranking, channel revision, backfill or direction-policy change.

## Exact target-free greedy rule

Start from S empty. For every still available three-coordinate group e use

`A_S = (I_n + X_S X_S' / n)^(-1)`

`G_e = (X_e' A_S X_e / n + (X_e' A_S X_e / n)') / 2`

`gain_e = log det(I_3 + G_e)`.

The regularization is fixed alpha1 under the existing mean-loss convention.
Use the natural logarithm; dividing all gains by a common constant would not
change selection, but do not add weights or tune the penalty. Select the
largest finite gain, breaking **exact** floating-point ties by ascending
physical electrode ID. Mask previously selected groups. Always select exactly
32 distinct groups, preserving one nested order for E=8/16/32. There is no
positive-gain stopping condition, post-selection reranking or optional best
prefix. Constant inputs and exact zero gains remain eligible under this rule.

A qualified rank-three update may keep `AX=A_S X`: initially AX=X. For the
chosen e, let `H=AX_e.copy()`, `M=I_3+G_e` and compute the pre-update right-hand
side `X_e' AX/n`; then set `AX <- AX - H solve(M, X_e' AX/n)`. Do not use an
already updated AX on that right-hand side. Each view owns its independent
state. No AY or target cross-covariance is needed.

Use `slogdet` on the symmetrized M, requiring positive sign and finite gain,
Gram, solve and updated state. Since the exact gain is nonnegative, define a
roundoff tolerance `128*machine_epsilon*max(1, ||G_e||_F)` in float64. A gain
below minus this tolerance is a technical failure, not a clipped score or a
reason to switch algorithms. A finite negative gain within tolerance is
retained exactly for ranking and logged; do not clip it, manufacture a tie,
redraw or stop early. This is a numerical-validity tolerance, not an effect
margin. Save selected gains, available exact-best tie counts and zero-SD input
columns, including numerical-negative occurrences. A direct-inverse fixture
must independently reproduce the choices/gains, and a zero-X fixture must
produce the first32 ascending physical IDs without early stopping.

## Matched decoder and reused references

Use the unchanged true-target direction-residual reduced-rank ridge head,
alpha grid, legal ranks, lower-rank/stronger-alpha ties and native-unit target
scaling. At32, each inner head uses its own24-row selector's columns and
unpermuted fitting/validation Y. Later conditioned folds retain every
initializer row and validate only newly acquired rows. All20 distinct target
models/session must be fixed before reading its40 evaluation trials.

Only after every new fit is complete, read the completed
`initializer_repeatability_attempt001/initializer_repeatability.json`.
Reuse its matched mean-five uniform-order panel, true repeatability panel and
original marginal panel; none is refitted or charged again. Their decoder,
target, representation,20-cell support and initializer exposure match this
candidate. Uniform selection itself does not require observing all96 initial
channels; its earlier matched experimental panel nevertheless paid exactly
that exposure, so no cheaper-uniform efficiency claim follows.

Require exact serialized animal/trial/physical-channel roles, target-unit
counts, all20 cells and physical/numeric/target exposure. Current-run unit
guides stay equal under the qualified reader; historical unit-guide IDs were
not saved, so do not assert an unavailable historical unit-ID equality.
Replay new target denominators against the recorded repeatability and uniform
panels at the existing1e-10 relative/absolute tolerance, and direction-null Q
against all three references at1e-10 absolute tolerance. The legacy marginal
panel lacks denominator/SSE fields; preserve that known absence and compare
its recorded values only. New all96 model capacity/channel records and Q,
wrong-pair Q and pairing increment must replay the corresponding complete
three-band reference records. No reference score selects electrodes or alters
normalization, capacity, seed or stopping.

Report candidate-minus-uniform, candidate-minus-repeatability,
candidate-minus-marginal and candidate-minus-direction Q for every cell,
session, animal and the equal-animal mean. Keep the existing equal20-cell,
equal-session-within-animal and equal-animal aggregation. Preserve all eight
fixed-prediction session omissions and candidate contrasts against each of the
five recorded uniform seeds. No seed/session is chosen or excluded.

Two separate prospective screens are fixed: candidate-minus-uniform Q must be
strictly positive in both animal means for useful target-free selection, and
candidate-minus-marginal Q must be strictly positive in both animal means to
pass the incumbent challenge. Exact ties fail. Repeatability and direction
contrasts remain specified descriptive comparisons; report their sign and
omission sensitivity without manufacturing an additional mechanistic endpoint.
Keep the existing0.001 search-progress resolution, with no new significance,
equivalence, biological-importance or automatic promotion claim.

## Cost, checks and charge

The policy pays `32*96+(n-32)*E` physical electrode-trials and30 actual-dtype
LFP values per electrode-trial; targets cost `n*U*10`. This is the same
physical and feature exposure as the reused matched references. Report
repeated and unique logical source bytes separately, plus actual CPU, wall,
peak memory and retained output bytes. No arrays or decoder coefficients are
serialized.

The new candidate alone fits20 models/session: five full96 checkpoints and
five checkpoints for each scarce E. Across eight sessions this is160 target
models,3,200 inner ridge fits,16,000 capacity scores,40 selector orders,
1,280 greedy selections,40 direction-null models,2,304 fitting-trial read
calls and320 evaluation trials. There are no new target-label maps, uniform
permutations, Monte Carlo models or reference charges.

The prior six-branch conditional workload completed760 models and240 greedy
orders in approximately103seconds. This one-branch X-only workload is smaller,
though its log-determinant runtime is not yet measured. Propose one complete
cohort with1CPU within4GiB,180seconds internal/210seconds external in a
coordinator-verified attended allocation with full reserve. No generic pilot,
automatic retry or resource-budget extension is activated. Root selects the
actual allocation and launch only after adoption and focused checks.

Focused new fixtures cover exact fitting-only X normalization and no-Y API,
direct-inverse logdet equivalence, masking/physical-ID ties/fixed32/negative
roundoff policy, inner-view exclusion, five-order/20-model barriers and
true-target heads, proper historical schema/full96 replay, and reference reads
after all fits. Reuse unchanged reader/decoder checks.

One complete new candidate adds **one** valid comparison,34to35. The five
uniform seeds, reused controls, cells, omissions and historical comparisons
add none. The shared40–112 allowance, coverage and post-coverage alternative
check commitments, patience and resource limits remain in force. Passing this
candidate would not complete coverage or open any held or named endpoint.
