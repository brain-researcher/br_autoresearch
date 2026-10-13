# First development comparison: offline random-subset information curves

2026-10-03. The coordinator adopted this prospective per-comparison recipe;
adoption recorded at 20:35:55 UTC. **Scientific scoring and global search are
not activated.** Metadata trial roles completed at 20:35:38.059729 UTC in job
46429138 without source-MAT or neural access. This is an initial reference for open exploration, not the
named complementarity/uncertainty amendment or the optional 16-trial protocol.

## Authority and question

The approved `open_exploration_design.md` says:

> For each developmental comparison, state the fitting pool, decision-time
> information, action grammar, target, evaluation set, model-fitting procedure
> and resource accounting before scoring it. These are per-comparison validity
> commitments, not an early freeze of the entire search.

It also requires a new endpoint's stopping/improvement rule before its first
scored search. The choices below instantiate that delegated development scope.
They do not amend the fixed session split, shared allowance or single held
opening. The question for this first comparison is how a fixed, ordinary
random-electrode baseline changes with available electrodes and fitting trials.
It supplies a matched baseline for later candidates, not a test of a named
explanatory rule or evidence of sequential acquisition efficiency.

## Metadata-only trial roles

Use the existing `runs/source_support/attempt002/metadata_support.json` and its
eight assigned development sessions: Mihili 20140217, 20140218, 20140303,
20140304; Chewie 20160929, 20161005, 20161006, 20161007. Preserve all eight.
The four already-held sessions remain unassigned at the trial level and their
neural payloads stay closed.

For each development session and direction 0 through 7, sort eligible zero-based
source trial indices ascending. Shuffle with Python 3.12 `random.Random`,
`seed(version=2)`, using the exact string
`ep08-development-trial-roles-v1|20261003|<filename>|<direction>`.
The first five indices are common untouched evaluation trials. The remainder
is the declared fitting pool in that seeded order. Do not redraw a split.
There are 40 evaluation trials per session. The smallest observed direction
has 17 trials, leaving 12 fitting trials and permitting the nested balanced
sets below. This split uses no LFP/spike values or policy scores.

For total fitting amounts 32, 64 and 96, take the first 4, 8 and 12 fitting
trials in each direction. The sets are nested. Inner fold `j` holds out every
trial whose zero-based position within its direction-specific fitting prefix
is congruent to `j mod 4`; whole trials, not bins, form the four folds.
Every fold retains all directions. All cells and methods share these indices.
Remaining fitting-pool trials are not inspected by this first comparison.

`tools/prepare_development_trial_roles.py` only materializes these metadata
roles and nested indices. It neither opens source MAT files nor grants a
candidate/evaluator access boundary. Evaluation arrays must remain evaluator
inputs when a later numeric consumer is implemented; a common account or a
manifest is not an OS-level seal.

## Representation, target and matched fit

Use the complete 150--450 ms post-movement window already indexed by the
metadata inspector: ten 30-ms bins. For each selected physical electrode,
average each of LMP (guide 0/0), 100--200 Hz power and 200--400 Hz power across
those ten bins, yielding three inputs per electrode. This is one declared
temporal representation; later supported representations remain open.

The target is every source-provided M1 spike-population column, identified by
its `M1_unit_guide`, averaged over the same ten bins. Preserve source column
identities and native per-bin units: no integer-count assumption, conversion
to Hz, neuron selection, target PCA or outcome-dependent target removal.
Provider processing and guide consistency must be disclosed by the numeric
consumer; a mismatch is a technical failure, not permission to select a
better target. This initial comparison predicts the supplied population
activity representation, not newly reconstructed raw spike counts.

Within each fitting fold, standardize each input and target column using only
that fold's mean and population SD (`ddof=0`). Replace an exactly zero SD by
one, recording the affected columns; do not drop target columns. Evaluation
and inner-validation values never fit these transforms. Refit transforms on
the full selected training set for the outer prediction. All electrode cells
at the same trial amount share the same target transforms.

Fit ordinary multioutput ridge with an unpenalized intercept via centered
inputs/targets, minimizing `mean_training ||Y-XB||_F^2 + alpha*||B||_F^2`.
The fixed alpha grid is `[0.0001, 0.01, 1, 100, 10000]`. Select alpha by the
sum of held-inner-fold squared errors in that fold's training-standardized
target coordinates; all folds contain equal trial counts. Resolve an exact
tie in favor of the larger alpha. Then refit once on the full selected training
set. No evaluation score chooses alpha, scaling, rank, target or subset.

Compare nested random sets of 8, 16, 32 and all 96 physical electrodes.
For replicate `r=0..4`, shuffle the ascending eligible electrode IDs with
Python 3.12 `random.Random`, `seed(version=2)`, string
`ep08-random-electrodes-v1|20261003|<filename>|<r>`; use prefixes of the same
permutation. All 96 is identical across replicates and is fitted once per
training amount, then reused exactly. No geometry or reliability estimate is
needed, and no selector accesses evaluation measurements. Model family,
alpha grid, tuning folds, target and evaluation support match within every
electrode comparison. Electrode and trial resources remain separate axes.

The declared regime is offline development: the fitting representation may
contain all 96 electrodes. Charge that access explicitly; a small retained
subset is not a claim that only those electrodes were originally observed.
No separate pilot, next-trial action, channel switch or hidden free initializer
is introduced.

## Endpoint, allowance and stopping rule

For a session, fitting amount and random subset, compute
`Q = 1 - SSE(model) / SSE(training-mean null)` on the common 40 evaluation
trials in the outer-training-standardized target coordinates. The null predicts
the outer-training mean vector, including the intercept; it never fits the
evaluation mean. Sum SSE over whole trials and all target columns. A nonfinite
or zero denominator leaves the comparison unresolved; do not add epsilon or
drop the session. This explicitly defined predictive R2 need not equal a
test-mean-centered R2 or the optional reference's implementation.

Average the five subset replicates within each electrode/trial cell (reuse the
identical full-electrode score), the 12 cells equally within session, four
sessions equally within animal, and the two animals equally. This scalar is
the first branch's search endpoint; retain all cell/session values and paired
electrode/trial increments, including negative values. Do not infer biological
replication from bins, units, random subsets or resource cells.

A completed method panel, including all seeds and resource cells, charges
**one** valid evaluation to the shared 40--112 allowance. Cells, tuning fits,
random replicates and a fit-only cost pilot are not separate valid evaluations.
Engineering failures retain their consumed resources and use the existing
maximum of 16 engineering retries. No scored pilot may masquerade as a free
evaluation. The first panel establishes a baseline, not a scientific winner.

For subsequent candidates compared on this same endpoint/support, reset the
shared post-minimum patience counter only for an improvement of at least
`0.001 Q` over that branch's incumbent, or a new feasible utility/robustness/cost
frontier point under the existing policy. The 0.001 value is an explicitly
prospective **search-progress resolution**: one tenth of one percent of the
training-mean-null SSE on this normalized endpoint. It is not a meaningful
biological margin, a significance threshold, or an imported 0.005 reference
decision. Stop for patience only after at least 40 total valid evaluations,
20 consecutive valid evaluations without the declared progress, applicable
coverage and the required alternative-check allocation; stop incomplete at
112 or a resource ceiling if required work remains. The first baseline does
not itself satisfy this search. A different branch endpoint needs its own
prospective improvement rule, shares these counters/resources, and cannot
reset consumed allowance. A branch's first score alone is not progress.

No new scientific success rule, uncertainty claim, explanatory endpoint or
held-session release follows. The selected final protocol's endpoints,
comparisons, margins and decisions still need to be fixed before the single
shared opening of all four held sessions. The optional reference's numerical
decision is unchanged and does not transfer here.

## Bounded next execution, after coordinator review

1. Materialize the metadata roles in a fresh output directory. One CPU,
   2 GiB, at most two minutes inside the next attended allocation; expected
   seconds and only the existing metadata JSON is read. Preserve the output
   as the selected split before any numeric consumer runs.
2. The selected training-only numeric reader/fit is now prepared at
   `tools/training_cost_pilot.py`; the fixed evaluator is prepared at
   `tools/development_baseline.py` and remains unrun.
   The initial cost pilot uses only Mihili20140217's 96-trial fitting prefix,
   one electrode permutation, all 12 resource cells and the fixed nested
   tuning. It returns elapsed time, peak memory, logical array bytes accessed,
   fit counts and numerical completion only. It reads no outer-evaluation LFP
   or spikes and computes no outer-evaluation prediction/score. Its inner
   training-pool CV selects alpha internally as prescribed; its errors,
   selected penalties and coefficients are not returned as scientific feedback.
   No coefficient, fitted effect or training quality summary is returned to
   choose a method. Bound: one CPU, 2 GiB, ten minutes, numeric threads one;
   maximum reservation `1/6` core-hour, with a 540-second internal deadline.
   Stop at the bound without automatic
   retry or changing the scientific recipe.
3. Charge actual Slurm CPU allocation time and elapsed time, including failed
   attempts, against the existing 1200 CPU-hour/120-hour/750-GB limits. Project
   full cost from the observed first-session fit workload and the other
   sessions' existing dimensions; apply twofold headroom and obtain a bounded
   selected full-comparison command before scoring. Do not launch a 40-panel
   search from an unmeasured estimate.

The eight native development files total 3,315,332,161 bytes (~3.09 GiB), but
are read selectively in place; no full-source copy is proposed. The first
pilot touches one 463,317,544-byte source file and only training trial arrays.
Logical numeric input is roughly `96*10*(288+number_of_M1_units)*8` bytes for
double arrays, plus small guides. Actual compressed-chunk filesystem I/O can
exceed that logical volume. Record own-process `ru_inblock` deltas during reads
and the whole pilot; these are cache-dependent OS block-input operations, not
HDF5 logical bytes. Exact physical storage-byte measurement is unavailable in
this worker and will not be inferred from logical values. Source deserialization,
finiteness, physical-unit guide consistency and actual runtime are not yet
qualified; these estimates are not a completed pilot.

The pilot and evaluator share the pure training-only fitter in
`tools/ridge_training.py`. Its mean-loss penalty convention has an independent numerical check
in `tools/test_training_pilot_math.py`: compare the dual Gram solve with
augmented least squares and require invariance to replicating every training
row. Additional focused checks compare inner-CV/refit with direct least squares,
verify the training-mean-null score and reject invalid relative folds. These
checks have not yet run. The retained mean arrays are only
`96 x 288` and `96 x number_of_M1_units`; one native trial window is temporary,
so the selected two-GiB bound is ample for this minimal NumPy/h5py worker.
No full MAT deserialization, source copy or source-sized buffer is used.

## Prepared evaluator boundary and workload

The fixed trusted evaluator finishes all 48 session models before opening its
40 reserved evaluation rows. The pure fitter receives only copied training
arrays, training-relative folds and counters; no source/evaluation identifiers,
paths or arrays enter its interface. Scaling and alpha selection occur there.
Scoring uses the returned immutable model record in the trusted evaluator,
which never returns evaluation arrays to fitting. There are no candidate
plugins or dynamic worker callbacks. The two reader entrypoints admit only
the fixed development filenames and their declared disjoint trial sets;
evaluation guide identities must equal training identities before neural
slices. Ordinary same-UID ambient filesystem access remains a weaker boundary
than OS isolation; this implementation does not claim an independent identity
seal or authorize adversarial candidate code.

The per-session workload is three training amounts times 16 models: five
subsets each for 8/16/32 electrodes and one all-96 model reused for five
replicates. Each model has 20 inner fits plus one refit: 1,008 ridge fits per
session and 8,064 across all eight. The pilot measures 252 of these fits for
one session/one subset seed with the identical shared implementation. Full
logical neural input is approximately
`sum_sessions 136*10*(288+native_M1_unit_count)*8` bytes plus guides; physical
compressed-chunk I/O remains cache-dependent. No entire MAT is copied.
The full command's bounded wall/CPU reservation will be selected from actual
pilot evidence, including twofold headroom and shared budget accounting;
this preparation alone does not launch scoring.

## Optional reference support fact

The existing outcome-blind metadata records 17 eligible trials in one direction
of development session Chewie20161007. The unchanged balanced 128-additional
reference cell requires 16 trials per direction plus two pilot trials: 18
before any untouched evaluation reserve. Therefore the complete reference
grid is unsupported in the fixed cohort. Retain that session and the failed
reference-support fact. This does not invalidate the open exploration branch
or authorize dropping a cell, session or animal.

## Execution record boundary

This prospective recipe was executed without revision: the training-only
pilot and three focused tests passed, and the complete fixed baseline finished
on 2026-10-03 at 20:55:33 UTC in allocation46429138. See the execution log and
`runs/simple_comparisons/development_baseline_attempt001/development_baseline.json`.
Earlier statements above that the pilot/evaluator were unrun describe their
pre-launch state. One valid development evaluation is now charged. Global
search completion, named explanatory scoring and held-session opening have
not occurred. The next selector comparisons are separate prospective recipes;
the completed baseline's method and artifacts remain unchanged.
