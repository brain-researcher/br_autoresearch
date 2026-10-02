# EP15 paper plan: location of a shared code or different task geometry?

[Current conceptual figure](ep15_conceptual_question-v2.png) · [Exact image-generation record](ep15_conceptual_question-v2-prompt.md).

Status: proposed study design, 2026-09-27. No MDTB neural result or model-class
claim is reported here.

The [2026-10-02 scope review](scope_novelty_review_20261002.md) records the
closest prior work and the conditional, statistical-account interpretation.

## The intended contribution

The paper should answer a concrete question: when cerebellar task maps differ
between people, are the same functional relationships simply placed
differently, or do the relationships among tasks themselves differ?

For a new participant, anatomy and Task A must predict maps for 18 conditions
that occur only in Task B. This makes the condition-geometry claim
prospective: the model must say which unseen condition pairs will become more
or less similar before that participant's Task B is opened.

A better average map score is only an intermediate result. Each explanation
must be adequate and must pass a signature that a competing explanation would
not naturally produce.

## What is already known

| Prior direction | Already established | What EP15 must add |
| --- | --- | --- |
| Individual cerebellar parcellation | MDTB task sets can personalize parcel boundaries and predict novel-task organization | Show whether boundary changes are sufficient after fair relocation and geometry alternatives |
| Hierarchical atlases and spatial covariance | Task A or rest can inform individual organization evaluated on Task B | Distinguish parcel membership from one smooth field and geometry-preserving remapping |
| Cross-task map prediction | Individual Task-A information can improve predictions in the other task set | Explain what stable structure carries that prediction, not simply repeat the gain |
| Condition distances and Gram matrices | Crossvalidated task geometry has been measured in MDTB | Predict a participant's signed B-only geometry deviation from Task A |

The novelty review should be updated before a paper claim, but the standard is
already clear: neither “first personalization” nor “first condition geometry”
is available. The potential new fact is a tested boundary between spatial
placement of a shared code and predictable individual change in that code.

## Study sequence

### 1. Show that reliable individual variation exists

Use independent Task-B halves to measure reproducible between-person map
variation beyond the development group-functional prediction. Resolve a
three-way signal gate: present, demonstrably below the meaningful margin, or
unresolved.

Do not rank personalized models before establishing this target. If there is
no resolvable individual signal, the mechanism question has no supported
object in these data.

### 2. Test the spatial explanations on the same maps

Fit M1–M4 with identical source maps, support, condition weights, and one
uniform Task-A-derived gain rule. Evaluate adequacy—the reliable variation
left unexplained—not just relative score.

M1 must localize its recovery to reliability-matched boundary neighborhoods;
M2 must use one field for every condition and predict local B-only residual
direction from source-map gradients. These tests make parcel and relocation
claims more specific than “the maps improved.”

Boundary and interior voxels are also matched on development-only residual
signal opportunity and source-map gradient magnitude. If reliable individual
geometry is present, M1 and M2 must predict it adequately; neither can win by
explaining maps while ignoring the central geometry target.

M3 must pass a full isometry and identifiability certificate. A certificate is
necessary for interpretation, but it is not evidence by itself; M3 must also
predict the maps adequately.

### 3. Make geometry a prospective prediction

Construct two independent crossvalidated 18-by-18 condition-geometry matrices
from four disjoint Task-B run partitions. Remove the shared global-gain
direction defined by the actual M3 source geometry.

The decisive M4 output is a participant-specific signed matrix predicted from
Task A. Score that complete matrix twice, once in each independent Task-B
geometry replicate. Do not select favorable condition pairs from audit data.

For each development-selected M4 mode, freeze the geometry pattern it is
supposed to produce. Compare full and mode-ablated predictions against both
observed B geometry replicates in that frozen projection; the ablation must
worsen observed prediction by a prespecified participant-level margin. Merely
changing the model's own output is tautological. If a mode is not identifiable
from Task A, fix it to zero; audit Task B cannot identify it after the fact.

M4 earns the geometry-changing interpretation only if it is adequate and
materially improves both map and geometry recovery over M1, M2, and the exact
paired M3. This protects the claim from generic flexibility, interpolation,
or a topographic model that happens to change voxel-weighted geometry.

### 4. Freeze a complete panel and test new people once

Development selects one certified representative of every class, rather than
one cross-class score winner. Shared quantities are refitted on all 12
development participants. The audit calibrator then uses anatomy and Task A to
write predictions for all 12 audit participants while audit Task B is
unavailable.

The one-shot evaluator opens audit Task B once and resolves signal, adequacy,
native signatures, geometry, increments, and multiplicity. A positive class
requires all of its components and resolved competitors. Two or more fully
supported non-nested explanations yield “multiple supported explanations,”
not an arbitrary simplicity winner. A model that is only map-adequate has not
earned its mechanism label.

## The discriminating predictions

| Explanation | What should be seen if it is right? | What would falsify that interpretation? |
| --- | --- | --- |
| Personalized parcels | Reliable recovery is enriched near frozen parcel boundaries after reliability matching; interiors need no target-specific profile | Gain is interior-only, explained by reliability, or requires target-specific response profiles |
| Smooth relocation | One Task-A-stable displacement field works across all conditions and its source-gradient prediction matches B-only residual direction | Condition-specific fields are needed, the field is unstable, or a matched random warp performs similarly |
| Shared geometry | Maps are adequate while participant-specific B-only geometry is equivalent to the shared geometry after global-gain removal | Reliable geometry deviation remains or bounded M4 supplies a material increment |
| Stable geometry change | Task A predicts the signed full geometry deviation in both B replicates; mode ablation removes the predicted pattern | Only map similarity improves, geometry prediction fails, modes are unidentifiable/unstable, or M1/M2 explain the same deviation |

## Result branches

| Result | Scientific reading | Next step |
| --- | --- | --- |
| One class is uniquely adequate and passes its signature | A constrained account of stable individual organization is supported within MDTB | Build the paper around that statistical account and its strongest falsifier |
| M4 predicts maps but not geometry | Flexible topographic prediction improved, but stable geometry change is unsupported | Narrow to map prediction or stop the M4 paper direction |
| M1 or M2 predicts geometry as well as M4 | Geometry prediction does not specifically support M4 over the spatial accounts | Support a spatial account only if uniquely supported; otherwise report multiple adequate explanations/non-identifiability |
| Two or more explanations are fully supported | The data support several constrained statistical accounts, including their native signatures | Report non-identifiability; do not choose by raw score or presumed simplicity |
| Maps are adequate but all native signatures fail | Some Task-A information transports, but none of the proposed mechanisms earns its interpretation | Report adequate prediction without a supported mechanism |
| Reliable signal remains after every class | Task-A-derived structure does not transport adequately to B-only variation | Report the boundary and use post-decision diagnostics to locate session/calibration limitations |
| Signal or decision bounds are wide | The 12-person audit is underpowered for the prespecified distinction | Report unresolved evidence, not equivalence or model failure |

## Figures, contingent on evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| **1. What kind of individual difference is being tested?** | Is the distinction between location and geometry understandable and testable? | Anatomy + Task A input, sealed B-only maps, four model-class cartoons, and their make-or-break predictions |
| **2. Is there reliable individual variation to explain?** | Does B-only variation repeat across run halves? | Every participant, seven domains, signal gate, group reference, and influence analysis |
| **3. Do parcels or one spatial field suffice?** | Are differences concentrated at boundaries or captured by one stable relocation? | M1 boundary/interior matching; M2 half stability, source-gradient alignment, and random-warp control |
| **4. Is condition geometry shared or person-specific?** | Does Task A predict the full B-only geometry deviation? | Predicted and two observed 18-by-18 matrices, M3 equivalence, M4 increments, and mode ablations |
| **5. Which explanations survive the sealed audit?** | Is one class uniquely supported, are several adequate, or does none transport? | All branch components, all 12 participants, simultaneous bounds, failures, and claim boundary |

The first figure is a synthetic question schematic. Later figures must show the
complete participant evidence; a selected cerebellar slice or condition pair
cannot substitute for the registered endpoints.

## Planned conceptual figure

The current three-panel schematic shows anatomy and Task A predicting unseen
Task B, the four competing accounts, and separate map/18-condition-geometry
targets. Triangle vertices denote task conditions, not cortical positions;
their side lengths illustrate representational distances. The actual endpoint
remains the full 18-by-18 geometry matrix. Native signatures and the 12/12
participant design remain in the analysis text rather than a crowded workflow
strip. One, several or unresolved accounts are illustrative possibilities.

Use synthetic cerebellar silhouettes, activity maps, and condition triangles
or matrices. Do not use MDTB participant pixels or imply observed results.

## When to deepen, narrow, or stop

- Deepen the paper when one class is adequate, passes its native signature,
  and survives its strongest matched alternative on sealed participants.
- Make M4 the central story only when Task A predicts the signed B-only
  geometry matrix and mode ablations behave as forecast.
- Narrow to topographic prediction if maps improve but geometry does not.
- Report multiplicity when non-nested explanations remain adequate; do not
  manufacture a unique mechanism with a generic simplicity preference.
- Stop the mechanism claim if there is no reliable signal, no executable fair
  panel, or no class is adequate with useful precision.

## Claim language

A successful abstract should name the supported class, the branch-specific
prediction it passed, the anatomy-plus-Task-A input, the B-only target, and the
within-MDTB participant-holdout scope.

It should not describe non-isometry as task-specific variation, because task
set and session are confounded. It should not claim a universal geometry,
literal parcels, causal function, population prevalence, or external
replication.
