# EP10 — adaptive residual co-projection modeling

This episode is governed by [the common adaptive protocol](../ADAPTIVE_SEARCH_PROTOCOL.md).

## Authority and history boundary

This current local contract is unregistered. It creates no canonical loop, authorizes no
data access or compute, opens no target combinations, and establishes no
scientific finding. New canonical loop, Goal, registered program, and search
policy bindings are mandatory. Prior EP09/10/11 work is exposure to
record, not admissible initialization evidence.

EP09, EP10, and EP11 can use the same SEU cells and axonal targets. Their
results are statistically correlated and cannot be advertised as independent
replications. Their group roles, outcome rules, and claims must all be locked
before the first shared audit opening; otherwise this episode is explicitly
outcome-exposed development only.

## Adaptive scientific question

After conditioning on a neuron's observed target count, target-specific
popularity, source/soma context, quality, and frozen set-geometry proxies,
which bounded residual-combination model—sparse, low-rank, hierarchical, or a
restricted hybrid—best improves the probability assigned to the complete
projection target set in unseen biological groups?

This is an adaptive comparison of complete-set probability models, not a
search for visually interesting target pairs. The baseline `Q0` and residual
model `Q1` always score identical cells over the identical exactly normalized
choice space.

## Observation and data roles

For cell `i`, freeze candidate targets `V_i`, complete observed set `S_i`,
observed count `K_i = |S_i|`, and choice space
`Omega_i = {S subset V_i : |S| = K_i}`. Every target in `V_i` must be either a
valid detection or valid non-detection; any unknown makes the primary set
incomplete. The most conservative verified animal/brain/specimen group is the
split and inference unit; random-neuron splits are forbidden.

Assign at least 12 groups to adaptive development and at least 8 groups to a
locked audit before target identities are inspected. Development groups choose
vocabulary support, establish `Q0`, search `Q1`, and calibrate the full-search
null through nested grouped folds. Audit target sets remain inaccessible until
one model family, rank/penalty, and inference procedure are locked.

## Common baseline Q0

All candidates condition on `K_i` and share:

- K-dependent target singleton/popularity terms;
- flexible soma coordinates, source parcel, layer/independent labels, and
  prospectively available quality/technical covariates;
- soma-to-target atlas distance;
- mean pairwise distance of proposed targets and soma-anchored minimum
  spanning-tree length; and
- identical supported `V_i`, `S_i`, `K_i`, `Omega_i`, group weights, folds,
  likelihood, and exact normalizer.

`Q0` pooling may be global, source-family hierarchical, or bounded partial
pooling across `K`, chosen entirely within development data. Q0 adequacy is a
hard constraint, not a weak foil for Q1.

## Bounded adaptive grammar for Q1

Residual pair features are projected away from the complete Q0 singleton and
geometry design under every supported choice space. One candidate may use:

1. **Sparse interactions:** lasso, elastic net, or group lasso over recurring
   target pairs;
2. **Low-rank interactions:** symmetric target embeddings with ranks
   `{1, 2, 3, 4, 6, 8}` and bounded Frobenius/nuclear penalties;
3. **Hierarchical interactions:** global pair effects with source-family or
   hemisphere deviations under shared shrinkage;
4. **Restricted hybrid:** a low-rank term plus at most 16 sparse residual pair
   corrections selected inside grouped development folds; and
5. **Calibration/pooling:** bounded penalty grids, target-support thresholds,
   and one of global or prespecified source-family sharing.

Exact normalization over `Omega_i` is required for Q0 and Q1. If a supported
choice space cannot be enumerated or evaluated exactly within the frozen cap,
that cell is prospectively infeasible; pseudolikelihood or post hoc vocabulary
shrinking is not allowed. Free-form pair mining, audit-frequency features,
axon-derived `Projection class`, per-source winners, `K` prediction, and
winner selection by coefficient interpretability are prohibited.

## Objective and constraints

For independent group `g`, compute

`d_g = mean_i log2[Q1_i(S_i) / Q0_i(S_i)]`,

then give groups equal weight in `Delta_bits = mean_g d_g`. Primary search
maximizes grouped out-of-fold `Delta_bits`. Secondary objectives are held-out
calibration, worst-source-family gain, sparsity/effective rank, and runtime.

A candidate is feasible only when Q0 passes frozen marginal/geometry/quality
adequacy tolerances, all probabilities normalize, no prespecified source
family loses more than `0.01` bits/cell, the gain is not dominated by one
group, and the fitted-Q0 full-search null is controlled. Freeze a smallest
useful gain `delta_combo`, group-level interval, and multiplicity rule before
audit.

## Search stages

1. **Measurement preflight:** authenticate identities/groups, full-target
   readability, vocabulary, atlas geometry, exact choice spaces, and audit
   sealing.
2. **Q0 stage:** run 8--16 valid bounded Q0 pooling/regularization trials until
   adequacy is met; every Q0 trial is ledgered and resource-metered. Freeze a
   common Q0 rule before comparing Q1 families.
3. **Coverage stage:** evaluate at least 16 Q1 anchors spanning sparse,
   low-rank, hierarchical, and hybrid families and their bounded ranks.
4. **Adaptive stage:** propose successors using the append-only development
   ledger; every trial declares a structural hypothesis, parent, and degrees
   of freedom.
5. **Null/falsification stage:** run exactly **99** fixed-seed replicates from
   full datasets simulated from fitted Q0. Each replicate starts with an empty
   ledger and reruns Q0 search, Q1 coverage/adaptation, early stop, family
   choice, and finalist selection with the identical controller, grammar,
   budgets, and promotion rules.
6. **Lock/audit:** select one Q0/Q1 specification, reproduce it from immutable
   development inputs, freeze all measurement/model/inference artifacts, hash
   the bundle, then score the sealed groups once.

## Required falsifiers and ablations

- end-to-end fitted-Q0 simulations that rerun the complete adaptive search;
- synthetic recovery of null, singleton-only, geometry-only, sparse-pair,
  low-rank, hierarchical, and misspecified-Q0 worlds;
- K- and group-preserving target-label permutation;
- proof that residual pair columns are orthogonal to the complete Q0 design;
- removal of geometry, source hierarchy, sparse term, and low-rank term;
- capacity-matched noise interactions and penalty-path stability;
- leave-one-group, leave-one-target-pair, source-family, and batch influence;
- alternate non-overlapping atlas vocabulary as a locked sensitivity; and
- lineage/exposure audit shared with EP09/EP11.

The fitted-Q0 search-null distribution calibrates the observed development
gain and the audit threshold. Simulating only the final chosen model is
insufficient because it omits search optimism. Freeze controller code,
proposal-model/version/prompt, sampling configuration, and the 99-seed manifest
before real outcomes. The Monte Carlo rule is
`p=(1 + #{null statistic >= observed}) / 100`, with `p <= 0.05` required. If
the proposal trajectory cannot be replayed deterministically, or readiness
profiling cannot fit all 99 reruns inside the total CPU ceiling, the null gate
is invalid and the episode stops before outcome search.

## Incumbent, budget, and stopping

The incumbent is a nonterminal development model. It grants no claim or audit
access. Run **8--16** valid Q0 trials and **32--72** valid Q1 trials, for
**40--88** total scientific development trials. Q1 patience is **14**
consecutive valid trials after its minimum with no feasible Pareto improvement
of at least `0.002` bits/cell. All four Q1 families, declared rank regimes, and
required falsifiers must have coverage. At most 12 engineering failures may be
retried outside scientific patience. The 99 mandatory null reruns do not count
as extra candidate trials but do count against compute and wall ceilings.

Resource envelope: CPU only; at most 1,800 aggregate CPU-hours, 144 wall-clock
hours, 64 concurrent cores, and 1.5 TB scratch, including fitted-Q0 full-search
simulations. Resource exhaustion returns `incomplete_search`, never a
scientific negative.

## Lock, one-shot audit, and terminal boundary

The lock bundle includes sources/hashes, cell/group/duplicate/exposure ledger,
atlas/vocabulary and observation rules, exact choice spaces, Q0 adequacy,
residualization matrices, grammar, full trial/proposal ledger, all grouped
predictions, fitted-Q0 full-search null, selected Q0/Q1, penalties/ranks,
`delta_combo`, interval/multiplicity code, falsifiers, environment, seeds, and
artifact hashes. Audit groups open once and can never cause sparse/low-rank/
hierarchical winner swapping or renewed search.

`candidate_ready` requires the frozen group interval above `delta_combo`, a
passed fitted-Q0 search-null criterion, acceptable Q0 adequacy and probability
calibration, concordant prespecified source-family behavior, and no decisive
falsifier or single-group dependence. Otherwise report
`closed_no_candidate`, `unresolved`, or `technical_failure` under the lock.

A positive audit means only that residual target-combination structure helps
complete-set prediction at the frozen atlas resolution, observation rule,
target count, vocabulary, and source distribution. It does not prove synapses,
functional coordination, causal routing, developmental mechanism, intrinsic
information, cell type, or independence from EP09/EP11.
