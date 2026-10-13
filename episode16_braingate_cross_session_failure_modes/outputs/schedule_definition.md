# Outcome-blind schedule and design definition

This is preparation for `structural_support`, not a participant assignment or
permission to decode neural/cursor/impedance values. The only execution input
is the existing 729-session structural JSONL. No trials are selected.

## Instantiated numerical description

The provider README defines `target_position[:,0:2]` as x/y in **task
coordinates**. It supplies neither a physical unit nor a resolution guarantee.
Exact raw coordinate pairs are therefore deduplicated without rounding. Their
unweighted centroid is subtracted and their RMS distance from that centroid
sets one positive isotropic scale. Neither rotation, reflection, separate-axis
scaling nor fitting to cursor/neural information is permitted. Coordinates are
compared through a one-to-one nearest-point correspondence. Repetition counts
do not change the geometric normalization.

The frozen arithmetic tolerance for this descriptive step is
`64 * eps_float64 * max(1, max(abs(raw_unique_xy)) / RMS_radius)` per session;
the pairwise distance limit is the sum of both session tolerances. Its units
cancel. It is a roundoff-scale descriptor, not a formal numerical error bound.
Centroid and squared-radius sums use `math.fsum` from the second attempt onward.
It does not allow provider quantization, physical
jitter, unstable layouts or a scientific tolerance chosen to increase support.
Matching components are checked for pairwise completeness: transitive chains
of approximate matches are disclosed and never called qualified equivalence
classes. Exact raw target distinctions are retained.

Per-block target coverage and transition-type counts are described. An absent
observed transition is not assumed to be a forbidden transition. A session's
pooled layout is not assumed to prove its blocks used one stable schedule.
The code does not call these descriptions eligibility decisions. The first
attempt used NumPy mean accumulation and remains preserved; the second changes
only the accumulation precision, without changing the tolerance formula.

## Conditional fixed-effect design

For each geometry group, only same-person, same-physical-array-set sessions
with the existing whole-block 64/64 capacity enter the conditional graph.
Edges are chronological 1–30-day or >=180-day pairs. Intermediate and reverse
edges do not enter the primary design. All candidate edges are retained;
none is dropped to restore rank. Connected components use physical session
nodes, and separate source/target indicator columns expose any additional
aliasing caused by the actual bipartite source-target pattern.

The design contains an intercept, orthonormal sum-zero Helmert source and
target effects, and the long-gap indicator. Each target gets equal total
weight, divided among its chronological sources. Rank uses the numerical SVD
cutoff `max(rows,columns) * eps_float64 * largest_singular_value`. The report
includes nuisance/full rank, condition number, the weighted residual long-gap
indicator norm, row hat leverage, and session incident hat-leverage sum (also
divided by total rank). These are explicitly defined diagnostics, not silently
chosen scientific information or influence thresholds. Geometry-only graphs
remain conditional on schedule/size/transition qualification; their passing
rank cannot prove full study support.

The six-near/six-long count requirement is a **held-out participant gate within
a common schedule group** (`SEARCH_POLICY.yaml`, `session_pairs`), not a
six/six requirement on each connected component. Counts may accumulate across
qualified components in that same schedule group; unrelated schedule groups
cannot be pooled to satisfy the common-group requirement. Every contributing
component independently needs both lag classes, full constrained rank and the
pre-set information/condition/leverage limits. Fit its lag coefficient
separately, then apply equal schedule-group weight and equal component weight
within each group. A shared coefficient across components must not be used to
rescue a component whose own lag contrast is unidentifiable. The held-out
six/six gate is not an additional unstated gate on every development component.

The descriptive code emits all nonempty components and never filters them on
six/six. An earlier reading of attempt02 summarized only components above that
count; it was an overconservative reporting filter, not an implemented rule.
The complete metadata report has six both-lag, full-rank components in five
people, including T11 `geometry_0037` with two near and 39 long pairs that the
summary omitted. All six are in separate geometry groups. The only groups with
multiple components are T11 `geometry_0032` (both components near-only) and
T2 `geometry_0104` (one long-only and one near-only). Neither supplies a
component-specific identifiable lag contrast by pooling its counts. These are
conditional descriptions, not final participant-support or feasibility claims.

## Eligibility still to qualify

The exact provider README says tasks include Radial-8, grids, Fitts tasks and
related variants. Neither reliable task names nor target size are in the
authorized view. The contract intentionally matches observed schedules instead
of requiring verified task identity. Full task names, controller versions and
source quantization provenance are therefore not universal eligibility gates.
Radial-looking coordinates nevertheless cannot establish size independence.
Known size-dependent schedules without the necessary sizes must be excluded;
the current metadata does not identify which session/block uses such a
protocol. A targeted annotation of size-dependent schedules can resolve that
specific exclusion without reconstructing full task identity. Uniform
"unresolved size dependence" fields in a descriptive report are placeholders,
not positive findings that all schedules depend on missing size. Whole-block
stability, transition-pattern compatibility and target coverage can be assessed
directly from the permitted structural view under a prospective rule.

Source pointers: shared release `source/README.md`, sections “Decoding dataset”
and “trials”; release `SOURCE.md`; primary associated code
<https://github.com/nptl-stanford/array-paper>; related article
<https://doi.org/10.1038/s41591-026-04530-3>. The public repository README and
code-tree metadata were inspected at commit
`93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834`. Its
[metadata.py](https://github.com/nptl-stanford/array-paper/blob/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metadata.py)
contains participant/array ordering and labels, not a session/block task map.
The inspected target/trial API in
[decoding_snr.py](https://github.com/nptl-stanford/array-paper/blob/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metrics/decoding_snr.py)
takes positions, epochs, bin size and a distance cutoff; it does not establish
task identity. This bounded inspection does not prove that no map exists in
other released source material. Repository `data/` summaries and notebook
outputs were not opened; they can contain participant outcomes. The article
has not been read for this step.

Before the dedicated storage census, source dtype was not certified by retained
metadata: the128-byte format report identifies MATLAB Level5 only, and the
selective reader places targets in a float64 array without retaining dtype tags.
[generate_decoding_summary.py](https://github.com/nptl-stanford/array-paper/blob/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/batch_summary/generate_decoding_summary.py#L104)
also casts target positions to `dtype=float` after loading them. That analysis
cast establishes no export/controller precision. The completed header-only
census now resolves all729stored target matrices as `mxDOUBLE_CLASS` and `<f8`;
a legitimate source-precision allowance still needs coordinate-generation and
export/quantization documentation. No support-driven tolerance relaxation is
selected.

`code/inspect_target_storage.py` completed the all-nine storage census at
19:59:29UTC on job46427768 in404.088s, following two passing synthetic access
checks. It decoded no numeric values and retained no raw MAT copy. Its storage
finding does not identify controller precision or task-size dependence.

## Remaining source and support route

The completed computation describes actual observed block layouts,
recurrence and directed transition identities from the completed structural
JSONL. It preserves all original trials, uses only the already-declared
roundoff descriptor for alignment, and does not activate a proposed scientific
matching tolerance or numerical support threshold. Differences in sampled
edge/target support are distinct from a proven incompatible transition rule;
one target point per trial cannot establish stability within that trial.
The exact command and attempt are in the execution log.

The [historical proposal](config/schedule_support_proposal.json) remains unchanged.
The coordinator adopted the [active numerical configuration](config/schedule_support.json)
at 20:22:11 UTC, before application. It defines the maximum one-to-one point mismatch
as the smaller of 0.001 times median intertarget spacing and 0.01 times minimum
intertarget spacing, with both spacing scales chosen symmetrically from the
two normalized layouts. These are all unordered target-pair distances, not the
nearest-neighbor median reported by the descriptive diagnostic. The angular
argument concerns a fixed cursor under one identical coordinate transform;
native-session labels and near-target exclusion are unchanged.

Near-duplicate coordinates remain exact distinctions. The new diagnostic finds
such cases in retained metadata, including T10 days67,112 and130, with minimum
normalized spacings around 1e-16. The proposal marks target distinctness
unresolved when such neighbors lie within the existing roundoff-scale
descriptor. That descriptor is not proof of an input error or physical jitter.
No rounding, automatic cluster collapse, tolerance floor or alternate spacing
statistic is used to recover support. The geometric rule must not silently
turn these unclassified distinctions into either verified equivalence or
terminal participant infeasibility.

The adopted support limits act on target-equal weights summing to one, the
existing fixed Helmert source/target basis, and the residualized long indicator.
The maximum node influence is its incident squared residual-information share,
not its generic full-hat mass divided by rank. Full-hat mass can be spread over
many nuisance directions while one session dominates the actual lag contrast.
The latter full-hat summary remains descriptive. The config fixes condition100,
residual norm0.10, retained energy0.10 and node share0.25 as operating limits,
not power/coverage guarantees. Equality remains unresolved, so strict passing
requires at least nine nodes; eight cannot all have shares below 0.25 because
the shares sum to two. Only the corresponding previously-null policy fields
were filled with these values and exact metric references before application.
No other margin was changed.

`code/apply_conditional_schedule_support.py` and all six focused checks completed
in coordinator allocation 46429138. The application took 2.821s and yielded
two numerically passing components among 32 designs, in T11 and T9, with no
nontransitive compatibility component. The
[execution interpretation](experiment_log.md#2026-10-03--structural_support-conditional-numerical-application-completed)
identifies exact contributing session/block questions and explains why this
does not qualify the cohort. It computes numerical designs
conditionally even where specific block/transition/size classification remains
pending. Nontransitive geometry components are unresolved in their entirety;
the code does not search for a convenient subclique. No role assignment or
neural access follows from a conditional numerical pass. The current paired
block/transition diagnostic is sufficient evidence only of observed support
agreement: absent sampled targets/edges remain unresolved, not proof of a
forbidden transition or changed protocol.

Where the observed view cannot answer a specific matching question, useful
**outcome-free source annotations**, keyed by participant/day/block, include:

- task/controller protocol and version, or an equivalent provider-defined
  schedule class (a familiar-looking radial/grid pattern is not sufficient);
- intended target centers, coordinate units/normalization, quantization or
  rounding steps, and whether locations vary within a block;
- schedule dependence on target size, whether sizes vary, and the source of
  any recorded size; unavailable size cannot be filled in from cursor success;
- allowed target-transition structure, whether absent observed transitions
  mean forbidden transitions or finite sampling, and any changing layouts.

These are possible documentary sources, not a mandatory full-provenance
checklist. In particular, a scientific equivalence tolerance can be fixed before
scores from a geometric error budget without alleging that it reconstructs
source/controller precision. A targeted size-dependence annotation is the
narrow remaining source question when that exclusion cannot be resolved from
available task-design evidence. No request has been sent to a third party.

Fix a unit-aware tolerance and explicit stable-layout, coverage and
transition-compatibility rule **before** qualifying any cohort.
Documented size-dependent schedules with missing target size are excluded;
unclassified schedules remain unresolved. A provider-grounded precision rule
may differ from the current strict arithmetic descriptor, but it must not be
chosen to recover a preferred participant count. Keep all same-person,
same-array chronological pairs within the qualified schedule groups; retain
every failed contributing component as unresolved instead of pruning it to
restore rank.

Then evaluate actual source/target fixed-effect graphs, numerical condition,
residual long-gap information and session leverage using prospectively fixed
scientific limits. Complete the prescribed prior-outcome exposure review and
apply the existing ordered role selection to the fully qualified metadata.
The design requires three eligible held-out people and six development roles,
at least four of them supported. T3's unconditional4-near/5-long upper bounds
still prohibit holding it out; its development role may carry the support
failure. Neither the current five conditional full-rank people nor the census
establishes the eventual qualifying count. A terminal support conclusion must
wait for the complete qualified rule and its application, without changing the
frozen design. Packet construction, onset/quality/margin choices, synthetic
checks and protected role views continue to precede neural scoring.

## Choices and exposure boundary

The transform symmetries, chronological gap classes, array/person identity,
whole-block capacity and sum-zero effects are prescribed and can be implemented
directly. Numerical SVD precision and outcome-blind descriptive arithmetic do
not require a new scientific authorization. Remaining null fields are
prospective analysis choices, not automatic permission blocks: the coordinator
can make contract-preserving choices concrete before scores and record their
scientific justification. Scientific condition/information/leverage limits,
size/transition classification, onset offset, quality rules and score margins
cannot be represented as already fixed merely because the diagnostic runs.
Changing gap/support minima, allowing rotations, ignoring unknown size
dependence, replacing participants after outcomes, or changing the six/three
design would instead amend the contract and needs the scientist's decision.

Before assignment, perform the prescribed participant-specific prior-outcome
exposure review of publications, notebooks, prior outputs and analyst notes
without opening new protected outcomes. This must include the
[2026-10-03 public-method exposure disclosure](experiment_log.md#2026-10-03--structural_support-public-method-exposure-disclosure):
the exposed reviewer received yield/within-session summaries, no identified
EP16 cross-session transport result, and cannot be treated as an untouched
evaluator. The exact nonnumeric participant/type list is retained in that log;
no new role follows automatically from it. T5/T10/T11 remain development; T3 is
ineligible for held-out assignment under the known pair-count upper bounds and
can occupy a development role with insufficient support recorded. Among
T2/T6/T7/T8/T9, apply the existing ordered selection rule only after full
structural and exposure qualification. Six development/three held-out roles
are retained, with at least four supported development participants. Protected
role views and a write seal still precede final held-out access.

## October 6 public-source update — documented fields and unresolved session annotations

The original [Dryad release documentation](https://datadryad.org/dataset/doi:10.5061/dryad.x0k6djj1h)
explicitly defines 10-ms neural bins, x/y positions in task coordinates,
zero-based starts and exclusive ends for blocks/trial intervals, electrode
impedance in kOhm, threshold crossings as counts per bin, and spike-band
power in microvolts squared. It names Radial-8, grid and Fitts task families.
The [pinned provider metadata code](https://raw.githubusercontent.com/nptl-stanford/array-paper/93beb76b4329f0e3d5b3fd5ffba9be1c7cfa2834/array_paper/metadata.py)
supplies the participant/array metadata used by the provider's simplified
analyses. These are public source definitions, not new neural observations.

The bounded review found no new per-session target-size annotation,
task/controller version, size-dependence classification or permitted-transition
rule. A task-family list or coordinate pattern cannot fill those missing
session-specific facts. Full task names and controller versions remain useful
source evidence, not new universal eligibility gates; the unchanged contract
requires observed-schedule compatibility and resolution of size-dependent
exclusions. Authentic original task records/provider clarification
remain the useful route; local analyst/publication exposure must also be
reconciled from its own history. No support calculation was repeated, cohort
requalified, tolerance relaxed or additional participant declared eligible.
The existing conditional T11/T9 passes and failed 4+3 support basis are
unchanged. This source update does not adopt a case-series redesign.

## October 6: prospective analysis and exposure definition

The coordinator authorized this preparation on October 6, 2026. The selected
[analysis definition](config/analysis_definition.json) closes the choices below
before any neural scores. It does not change the active geometry/support
configuration, qualify a participant, assign a role, activate a reader, or
select another application of the unchanged support analysis. Its small
[numerical consumer](code/analysis_primitives.py) accepts caller-supplied arrays
only and contains no source-file reader. Its three focused checks subsequently
passed in allocation 46721202; see the [execution record](experiment_log.md#2026-10-06-0529-utc--analysis_definition-focused-checks-pass).
These are implementation checks, not a new simulation campaign.

### Matching and target coverage

Keep the exact target identities, normalization, bijection, tolerance and
complete-clique rule already adopted in `schedule_support.json`. Equal observed
target sets in every nonempty block supply observed layout agreement. Different
block sets remain ambiguous between finite sampling and multiple layouts;
independent evidence of one intended layout can resolve this ambiguity, whereas
documented multiple/unstable layouts are excluded. Do not require every block
to contain every target after a common intended layout is independently known.
For the eventual fixed packet, require every target in that schedule to occur
at least once on **each 64-trial side**. This directly gives both model fitting
and evaluation support over the same target universe. Failure remains an
unresolved session: do not append trials, replace the first nested trials or
drop blocks to manufacture coverage. This rule has not been applied to the
source and was not selected by comparing candidate cohort sizes.

Equal positive directed-edge support supplies observed transition agreement.
When observed supports differ, matching independently declared permitted-edge
sets can establish that the discrepancy is sampling; unobserved edges are not
declared forbidden. A known difference in permitted transitions is a conflict.
One target point per trial never establishes absence of within-trial movement;
the rule certifies only the stated observed schedule, with any specifically
identified instability retained. Controller/task identity is not universal.
Size-independent schedules need specific supporting evidence. Known size-
dependent schedules without actual sizes are excluded; an unclassified schedule
stays unresolved. If sizes become available, their units/semantics and matching
under the existing isotropic transform must be specified before qualification;
this preparation does not invent those source quantities.

### Window, score and channel handling

Select offset **0 ms**: the first 30 released 10-ms bins beginning at the
declared trial-start index. This is an operational release-aligned window,
not an inferred physical cue or a physiological reaction-time claim. Its start
fixes the existing cursor-to-target label and near-target exclusion. A missing
or incomplete 30-bin window remains unresolved, never shortened or padded;
trial success or later cursor corrections never select the packet.

Select **summed**, not coordinate-averaged, squared error:
`Q = -sum_i w_i[(y_ix-p_ix)^2 + (y_iy-p_iy)^2]`. A zero predictor for unit labels
therefore scores exactly -1. Define eight counterclockwise half-open octants
from the positive x axis, with `w_i=1/(8*n_octant(i))` independently within the
fixed fitting/evaluation packet. All eight must occur; a missing octant makes
that packet unresolved rather than changing the estimand or replacing trials.
This applies also to the fixed first-16/32/64 fitting subsets. The two outer
scores are still averaged before subtraction or ratios. No target direction is
inferred from a target location alone.

The channel-window mean exists only if all 30 values are finite; otherwise it
is a missing channel feature, not a reason to delete that trial. Source fitting
means and population standard deviations (`ddof=0`) use only finite source
fitting-trial features. At least two are needed; zero/nonfinite scale leaves
the source channel inactive. Retain its physical-axis slot as zero. At target
evaluation, use the same source statistics; absent, invalid or inactive
coordinates are zero in source coordinates. No epsilon scale, target refit,
mask predictor or electrode-column deletion is introduced. The target-local
model can fit its own scaler only on its designated fitting side.

For the directly observed recording-state order, the later declared available
electrode set must be a subset of the earlier set, and every shared channel's
complete-window rate must be no higher, **in both folds**. New electrodes or
crossing quality directions are incomparable controls; differing fold orders
are fold-discordant controls. Ties are recorded as ties, without invented
degradation. This order identifies availability/invalid-window changes only.
Variance direction does not establish SNR, attenuation or added noise, and
optional yield/impedance absence never determines cohort membership. Compatible
attenuation/noise alternatives remain a separate internal model specification.

### Adopted anchors and genuinely unfinished choices

Adopt the existing practical anchors: adjusted and robust extra long-gap loss
must each exceed **0.10**; fractional local-signal change has upper boundary
**-0.15**; gap recovery exceeds **0.50**; relative gain exceeds **0.10**; and
normalized recording-reproduction error is below **0.10**. The robust statistic
tests the same meaningful loss as the adjusted statistic, hence the same 0.10
boundary. These are chosen effect criteria, not estimated power guarantees.
Equality remains unresolved. Corresponding policy fields are now filled; no
effect or readiness threshold was optimized on a neural outcome or simulation.

The new raw score units make further choices possible but do not scientifically
determine a raw-score decline/readiness margin. Those margins, the recording
extra-loss/random-separation criteria, compatible attenuation/noise families,
recovery ratio aggregation and lag population, literal past-only exposure,
adaptation complexity/stability and remaining joint decision uncertainty are
**internal specification work**, not unavailable provider data. They remain
explicitly unset. The original synthetic requirements remain recorded; this
preparation neither launches an expanded suite nor silently waives them.

### Participant-specific exposure review specification and present record

The permitted review records **identities, source pointers, exposure types and
reviewer access**, never fresh outcome values. Use the existing design record,
publication/notebook inventories, prior output descriptions and analyst notes;
do not reopen their neural result panels to establish a negative exposure
claim. Record each relevant source checked, what type of result the reviewer
had already accessed, and any unreviewed history. An absent recorded result is
not evidence that no analyst has seen it. A concrete analyst/source disposition
can resolve missing history; a generic assertion that the public dataset is
"clean" cannot. No message or request to an external party was sent.

| Participant | Verified existing exposure, without values | Present disposition |
| --- | --- | --- |
| T2 | Yield summaries in the October 3 disclosure | Candidate; relevant cross-session history review incomplete |
| T3 | Yield and within-session summaries in that disclosure | Cannot be held out under the existing 4-near/5-long upper bound; retained development possibility |
| T5 | PRI-T/chronic recalibration design history; disclosed yield summaries | Already fixed development |
| T6 | Disclosed yield and within-session summaries | Candidate; relevant cross-session history review incomplete |
| T7 | Disclosed yield and within-session summaries | Candidate; relevant cross-session history review incomplete |
| T8 | Disclosed yield summaries | Candidate; relevant cross-session history review incomplete |
| T9 | Disclosed yield and within-session summaries | Candidate; relevant cross-session history review incomplete |
| T10 | Prior public session-level decoding/angular-error design exposure; disclosed yield summaries | Already fixed development |
| T11 | NoMAD/MINDFUL design history; disclosed yield summaries | Already fixed development |

The authority for initial development is `DATASETS.md`, "Development and
held-out participants". The exact October 3 disclosure is in `experiment_log.md`,
"structural_support public-method exposure disclosure". It identified no EP16
cross-session transport result; that is a limited observation, not a completed
negative review of all publications, notebooks and analyst histories. Known
yield/within-session exposure alone does not automatically reassign a candidate.
Verified participant-specific EP16 cross-session outcome exposure moves that
person to development under the existing rule. The disclosed reviewer
`/root/inventory_ep08_14` cannot be represented as an untouched final evaluator.

After source/support qualification and this review, apply the unchanged ordered
six/three selection from `DATASETS.md`. T5/T10/T11 stay development and T3 cannot
be held out. Preserve all participant and failed-component dispositions; no
score-dependent replacements or extra participant are available. Preparation
of protected reader views follows actual roles, not guessed roles. The existing
structural JSONL remains the only available all-participant analysis view;
`inputs/README.md` still prohibits a development reader from accessing held-out
neural data. No new receipt/schema or standalone refusal gate substitutes for
the still-missing source evidence, scientific choices or real reader isolation.
