# Adaptive validation record

This file is a historical portfolio design audit, not a launch checklist. Its
data, evaluator, runtime, and audit findings are now inputs to each episode's
own bootstrap and validation stages. `bin/codex-episode` does not read this
file or require its issues to be resolved before starting an episode.

Reissued on 2026-09-22 after the formal EP01 replacement and promotion of EP02
to a complete but blocked adaptive-search contract. This is a design,
source-acquisition-integrity, and filesystem validation only. The earlier
2026-09-21 fifteen-episode registry inventory is retained in Git history rather
than treated as current. No primary audit outcome, canonical transition, or Git
operation was opened by the EP01 replacement or EP02 promotion. No EP02
candidate search/controller ran and no primary audit role opened. A superseded,
uncommitted local Phase-0 v1 packet nevertheless emitted the per-record fields
`stimulated_trial_count` and
`declared_contingency_structurally_consistent` before that point. Their values
are conservatively treated as exposed regardless of whether anyone viewed
them. Phase-0 v2 removes both fields, and neither field may ever enter candidate
scoring, model selection, an audit terminal, or amplitude-boundary
interpretation.

## Portfolio coverage

- Nineteen current formal episode directories exist: EP01--EP17 except EP18,
  plus EP19--EP20. EP18 is the sole incomplete, uncounted, and unlaunchable
  draft.
- Each formal directory has one current `GOAL.md`, `DATASETS.md`, and search
  policy, plus retained `inputs/` and `outputs/`. EP02 uses
  `SEARCH_POLICY.json`; the other 18 formal episodes use `SEARCH_POLICY.yaml`.
- The current EP01 is `episode01_narps_deep_search`. The old NARPS trees are
  archived under `_examples/historical_prior_records/` as immutable, fully
  exposed prior records rather than current episode directories. Their one-time
  relocation changed no file content.
- The parallel intermediate portfolio and local numbered version trees remain
  removed; no alternate version tree competes with any current registry path.
- Existing input data were retained. The intentional MaleCNS read-only input
  symlink remains; no other episode input was moved or rewritten.
- EP08's scientific object is post-pilot electrode retention and sequential
  calibration-trial acquisition rather than a fixed scaling surface.
- EP13's scientific object is counterexample-guided theorem discovery rather
  than verification of a nearly supplied theorem.
- EP15's scientific object is discrimination among parcel membership, smooth
  relocation, certified isometry, and bounded stable non-isometry for reliable
  between-person Task-B-only cerebellar variation. It is not another generic
  alignment benchmark or a universal cognitive coordinate claim.
- EP16's scientific object is adjudication among operational cross-session
  failure signatures under one task-, budget-, and model-matched panel. It is
  not reconstruction of a historical online decoder or additive causal
  decomposition of hardware loss and neural drift.
- EP17's scientific object is identification of controlled pairwise visual-
  model relations under a finite linear-encoding measurement-contract family.
  Its audit is concept-disjoint and does not refit the development predictor.
  It is not a model leaderboard, cross-dataset/new-participant replication, or
  evidence for a universally best linking model.
- EP02 is a formal direct F20-17 contract with a 28--48-trial bounded grammar,
  whole-mouse development scoring, and an intended one-shot 6-versus-5 primary
  intervention audit followed only by a non-rescuing 4-mouse amplitude-boundary
  diagnostic. Its role-pack materialization integrity passed, but the handoff
  is `materialized_unsealed`: same-UID source readability and network
  reacquisition still defeat the confidentiality firewall. Generic calibration
  v2 returned `selected_rule: null` (0 of 216 eligible), so the current policy
  authorizes neither an audit opening nor a positive mechanistic terminal.

## Optional machine-readable checks

The retained readiness projections and `bin/validate-readiness` can still audit
legacy canonical-confirmation metadata, but they are optional diagnostics.
They do not determine whether an episode-managed run may start. Structural and
numeric checks needed by a scientific result are executed by the episode and
recorded with that result.

All 18 current formal YAML policies parse without duplicate mapping keys, and
the EP02 JSON policy parses as JSON. Each of the 19 policies has:

- a unique identifier from EP01--EP17 except EP18, or EP19--EP20;
- the exact minimum/maximum valid-trial, patience, CPU, GPU, and wall-clock
  budget fields;
- an explicit CPU, GPU, wall-clock, and valid-trial safety ceiling;
- a nonterminal incumbent;
- at least two adaptive successor cycles and two incumbent/challenger
  decisions;
- a numeric mandatory branch-coverage floor that precedes qualified patience;
- outcome-linked successor lineage fields that prevent a fixed grid from being
  relabeled as adaptive after execution;
- typed lock/audit records that bind every audit receipt to the immutable
  configuration-lock hash;
- a post-coverage falsifier fraction of at least 0.40;
- explicit lock, one-shot audit, no-post-audit-tuning, and terminal mappings;
- null canonical bindings and a derived compatibility projection of
  `launch_blocked: true`.

The JSON schemas parse. A PyYAML unique-key pass for YAML policies, JSON parsing
for EP02, and the `jsonschema` Draft 2020-12 validator passed all 19
registry-selected policies. All EP01--EP17 except EP18 and EP19--EP20 registry
paths resolve. No unexpected symlink is
present in the tracked EP16/EP17 contract tree; EP17's restricted raw DataLad
staging uses annex symlinks by design and is excluded, while role-filtered
handoffs must be materialized. Every EP16 and EP17 policy terminal is
represented in its Goal, and the checked files have no trailing whitespace.
EP06, EP07, and EP15 retain pre-existing Goal/policy terminal-wording gaps;
they are outside this EP17 implementation and are not counted as EP17
validation failures.

EP02's current Phase-0 v2 packet retains the validated source pin,
24-record/cohort structure, mouse inferential unit, nonlaunching semantics, and
SHA-256 replay while removing both v1 fields. Its trusted materializer then
produced exact hash-bound role packs and passed integrity/replay checks. The
superseded uncommitted v1 packet's two emitted
per-record fields remain a permanent prelaunch exposure even though v2 removes
them: `stimulated_trial_count` and
`declared_contingency_structurally_consistent` are excluded from candidate
scoring, model selection, audit-terminal logic, and boundary interpretation.
That is not a physical confidentiality seal: the current same-UID environment
can still read the provider source, and a network-enabled worker can reacquire
it. Outcome-blind generic calibration v2 evaluated 216 candidate rules and
returned `selected_rule: null`; therefore no rule may bind an audit opening or
positive terminal. Endpoint semantics, an assignment-mechanism receipt (or a presigned
conditional-exchangeability contract), endpoint-faithful calibration with an
absolute raw-unit margin, scientist signoff, an author-code port/configuration
and resource profile, a permission-separated evaluator, and canonical
registration remain unresolved.

## Trial budgets

| EP | Minimum | Maximum | Patience | CPU-core-hour ceiling | GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| EP01 | 30 | 60 | 12 | 2500 | 0 |
| EP02 | 28 | 48 | 10 | 800 | 0 |
| EP03 | 30 | 72 | 12 | 1500 | 0 |
| EP04 | 24 | 64 | 10 | 2500 | 0 |
| EP05 | 28 | 64 | 12 | 2000 | 0 |
| EP06 | 20 | 48 | 10 | 1200 | 0 |
| EP07 | 32 | 80 | 14 | 800 | 0 |
| EP08 | 40 | 112 | 20 | 1200 | 0 |
| EP09 | 40 | 96 | 18 | 1500 | 0 |
| EP10 | 40 | 88 | 14 | 1800 | 0 |
| EP11 | 48 | 120 | 18 | 1500 | 0 |
| EP12 | 36 | 96 | 16 | 1000 | 0 |
| EP13 | 16 | 64 | 12 | 256 | 0 |
| EP14 | 48 | 120 | 20 | 500 | 0 |
| EP15 | 40 | 96 | 16 | 1200 | 0 |
| EP16 | 40 | 96 | 16 | 1200 | 0 |
| EP17 | 32 | 72 | 12 | 6000 | 256 |
| EP19 | 24 | 40 | 8 | 2000 | 480 |
| EP20 | 32 | 64 | 12 | 4000 | 960 |

These maxima are safety ceilings, not targets. Qualified patience cannot fire
before required branch coverage and falsification depth are satisfied.

EP10, EP11, and EP12 additionally require 99 null replicates that restart the
entire adaptive controller from an empty ledger, not merely the selected model.
Their deterministic replay and pre-outcome compute-profile gates are design
requirements; the executors and profiles do not yet exist. EP14's exact audit
recipe uses all 3,227 training rows, 395 external rows, and a frozen 10,000-draw
hierarchical bootstrap after a single configuration lock.

EP02 currently permits development-contract work only, not an audit opening.
Its generic whole-mouse simulation is a diagnostic calibration, not an
endpoint-faithful operating characteristic: zero of 216 rules met the complete
safety and power criteria. Before any primary payload can open, the endpoint
numerator, denominator, eligibility, missingness, and time alignment must be
frozen; an absolute raw-unit margin and scale floor must be presigned; the
assignment mechanism or conditional-exchangeability basis must be recorded;
and a new bounded/discrete endpoint-specific calibration must pass independent
validation and scientist signoff. The nominal 462 label enumeration is exact
for the stated sharp hypotheses only under the documented assignment design or
the presigned exchangeability condition; it is not presented as exact for a
heterogeneous weak average-effect null.

EP15 plans 12 development and 12 audit participants, with IDs still unassigned
pending a Task-B-blind metadata/Task-A reliability balance rule and full
decision-tree simulations. Its 18 Task-B-only conditions are weighted as seven
domains. Audit anatomy and Task A may calibrate only the locked four-branch
panel, while every audit Task-B file remains evaluator-only until all
predictions are locked. The acquired ZIPs mix those roles, so the role-filtered
handoff and evaluator remain episode-internal prerequisites before EP15 opens
its held-out outcomes.

EP16 plans six development and three whole-participant audit roles. T5, T10,
and T11 are the initial forced-development set from prior/public exposure; the
three audit IDs remain unset pending an exposure-regenerated, target-schedule-
only structural support scan. After roles freeze, scorability uses immutable
128-trial packets with block-disjoint 64/64 outer folds and cannot trigger role
replacement. Its source × target panel uses near lags of 1--30 days, long lags
of at least 180 days, a nested 16/32/64 trial budget, one pre-audit recovery
method, and simultaneous transition-equivalence tests for measurement
sufficiency across degradation-admissible near and long transitions. That
candidate must also reproduce a positive admissible-pair excess contrast, and
its identity-aware versus matched-random rule participates in the main
tri-state. A component-matched `S+D` route keeps genuine local-signal reduction
testable when the normalized mismatch denominator is not ready. At least four
development participants must pass the same scorable-packet and rank gates.
Human-signed scientific margins, the finite final-`m*` recipe rule, numeric
model/rank bounds, the exact emulator and random bank, role-filtered handoffs,
and a tri-state one-shot evaluator remain episode-internal prerequisites for
EP16's terminal claim.

EP17 uses CNeuroMod-THINGS 1.0.1 only. Before neural values are read, its 720
eligible concepts must be assigned globally to 480 development, 120
support/calibration, and 120 sealed-audit roles. The final predictor and voxel
coefficients are fitted only on development concepts and applied once, without
refit, to audit concepts in the same four participants. The policy separates
correlation, held-out variance, normalized variance, and CKA scales; a reversal
requires opposite practical-margin crossings across a one-axis edge. The
37-object 128.145 GiB manifest and Zenodo provenance archive are locally
acquired and verified. The original THINGS archive and exact CNeuroMod subset
are also acquisition-verified under recorded terms but remain unextracted.
Role-filtered handoffs, the frozen taxonomy, controlled extraction and exact
image-event alignment, stage-specific B/C/D ceilings, a hashed terminal-
contract universe, numeric scientific contracts, and a trusted evaluator
remain EP17 bootstrap and held-out-evaluation tasks.

EP19 has a complete adaptive contract and seven initialized projections. Its
WAY-EEG-GAL and self-paced source archives, plus the pinned `Safe1` source, are
acquisition-verified in steward quarantine; they remain unextracted and are not
episode handoffs. Timing, onset, channel, admissible-space, margin, compute,
role-filtered handoff, and joint one-shot evaluator contracts remain EP19
bootstrap and evaluation tasks.

EP20 has a complete adaptive contract and seven initialized projections. Its
tracked documentary anchor contracts do not constitute a qualified NeuroCam
reference ensemble, executable finite design rules, replay data, or an audit
runner. It remains a virtual, paper-derived design question with no physical-
superiority claim; the gaps listed in its policy must be handled inside EP20 or
produce a technical/no-candidate terminal.

## Optional canonical-confirmation boundary

The observed canonical registry exposes no program with the required adaptive
inner-loop semantics. Every policy therefore remains `planned_unregistered`
and null-bound. The legacy `launch_blocked` boolean is inert compatibility
metadata; it no longer describes whether the local episode launcher can run. A
later canonical confirmation may still require a readiness manifest, an
independent evaluator, and fresh server state. Dataset-specific gaps in each
`DATASETS.md` remain real claim boundaries; more development search cannot
erase them.

Realized trial count, branch coverage, adaptive successor cycles,
incumbent/challenger decisions, configuration-lock receipts, audit opening,
and zero post-audit mutation are episode-internal realization requirements,
not portfolio prelaunch gates. The episode verifies the corresponding schemas,
operations, isolation, and replay during its own bootstrap and execution.

For the new EP01, the two legacy loops are prior identities only. The first was
observed `COMPLETE@3/closed_no_candidate`; the second was observed
`AWAITING_REWARD@3`, with a returned reward action that expired on 2026-08-18.
Neither is rebound, mutated, or replayed from the new workflow. Any actual
canonical prepare collision must be handled separately; the mere nonterminal
legacy record is not declared to block a new identity. The prior-lineage file
is currently a hash manifest, not a materialized input packet; `ds000005` has
no permission-separated audit handoff.

The server reported contract version `2026-07-31`, profile
`codex_autoresearch_v1`, five registered programs, and 19 canonical loops. No
loop matched Episode 17, and no registered program exposes its bound inner
search policy, trial-ledger state, incumbent/challenger controller, and
permission-separated one-shot audit. These calls were read-only; no checkpoint
or action was submitted.
