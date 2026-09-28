# EP17 paper plan: predicting when an fMRI model ranking will change

Status: proposed study design, 2026-09-27. No neural score, sealed-concept
result, or model-ranking claim is reported here.

## The intended contribution

The paper should not conclude merely that model rankings depend on the fMRI
analysis. Its central contribution should be a rule that predicts which model
benefits and why.

For a controlled model pair, map where model A or model B has the larger
voxelwise held-out advantage. Then compare that map with an independently
defined map of what the measurement operation changes:

- reliability gained by moving from B→C or C→D beta estimates;
- voxel membership gained by moving from an anatomical to a
  reliability-selected support; or
- positive weight gained under noise-ceiling normalization.

The paper-level hypothesis is that a regional ranking changes when those two
maps align. The prediction must be frozen in development and tested on new
concepts in the same four people.

All operator and advantage maps are defined within participant and visual
ROI. Voxels are averaged inside each participant/ROI before equal-participant
aggregation; native voxels from different people are never pooled as
independent observations.

## Why the three operations must remain distinct

| Operation | Mathematical mechanism | Explanatory test |
| --- | --- | --- |
| Beta construction | Can change the response and therefore the A–B advantage at each voxel | Does calibration repeat-reliability gain predict the voxelwise change in advantage? |
| Voxel support | Leaves each voxel's fixed-prediction advantage unchanged but changes which voxels are averaged | Does the development inclusion decomposition forecast the audit change? |
| Ceiling normalization | Leaves every raw prediction, voxel, and within-voxel winner unchanged; positive weights alone change | Does the development weight decomposition forecast the audit change? |

Calling all three “pipeline sensitivity” would discard the most informative
part of the experiment.

## Novelty test before a paper claim

A focused review must determine what has already been shown about:

- visual-model ranking changes across GLM or denoising choices;
- reliability-based voxel selection and model comparison;
- noise-ceiling normalization as voxel reweighting; and
- prediction of a ranking reversal from voxelwise model advantage.

The intended novelty is not the size of the contract grid. It is a prospective
operator–advantage prediction that distinguishes response change, population
selection, and score reweighting. If close prior work already makes and tests
the same prediction, EP17 needs a narrower biological or methodological
contribution before execution.

## Study sequence

### 1. Build isolated comparisons

Select controlled model pairs that differ in one named property. Use concepts,
not images, as the leakage boundary. Reserve 480 concepts for development, 120
for calibration of reliability/support/ceilings, and 120 for one sealed audit.

Build the exact four-person image intersection and align every image, trial,
voxel, and response unit across B/C/D. Freeze the anatomical visual ROIs,
support size, subset bank, spatial quotas, ceiling estimator and floor before
candidate scores.

The initial five-edge family is fixed in advance: B→C and C→D on raw R² and
the same `A_all` voxels; D-stage `A_all→A_N` as a size control; D-stage
`A_N→R_N` with equal counts and quotas; and raw→normalized D-stage R² on
identical `V_NC` voxels. Every endpoint receives a named audit homolog. Only a
prespecified structural or calibration failure can make one inapplicable.

For beta edges, use one common layer/PCA/ridge selection across stages. For
support edges, fit one voxel-prediction bank and apply the two support masks
afterward. For ceiling edges, change only the positive voxel weights. A
comparison that violates these rules is composite and cannot establish a
measurement-specific reversal.

### 2. Establish a stable relation or a genuine reversal

Use raw held-out voxelwise R² as the primary score. A stable relation requires
the same signed practical-margin result across every eligible primary
contract. A reversal requires both ends of one isolated edge to cross opposite
margins under simultaneous uncertainty.

Raw and ceiling-normalized R² have separate practical margins. A
raw-to-normalized reversal controls their joint uncertainty through
studentized max-t statistics while retaining each effect in its own units; the
two score values are never averaged.

Show all four participants, all concept blocks, and all eligible contracts.
Do not treat a point sign flip, a selected ROI, or a nonsignificant endpoint as
evidence of reversal or equivalence.

### 3. Predict the measurement effect

For every edge, define the operator map without candidate-model scores. On
development concepts, estimate the voxelwise pair advantage.

For support and ceiling, first compute the development edge change directly
from the frozen development voxel advantages and membership or ceiling
weights. This exact reaggregation attributes the development result; it does
not predict new data. Then freeze a separate audit forecast calculated from
development advantages only. The evaluator compares it with an observed
reaggregation of audit advantages. Audit values cannot enter the forecast.

For beta edges, use two deterministic repeat halves to estimate calibration
reliability gain. Within each participant and ROI, cross-fit one ordinary
least-squares model of the voxelwise advantage change from standardized gain
and fixed spatial-bin intercepts. Fit on 12 development blocks and score the
other 12, then reverse. Score participant-by-block prediction errors, not
voxel rows. Refit the same equation on all 24 development blocks and freeze
its direction, predicted magnitude, absolute-error margin, and median-split
high- versus low-gain comparison.

The preferred explanation predicts both sign and magnitude on sealed
concepts. A real reversal with a failed prediction is reported as unexplained
response-geometry or readout sensitivity rather than as recovered signal.

### 4. Test new concepts without refitting

Fit the final voxel coefficients on all 480 development concepts. The trusted
evaluator applies them directly to the 120 audit concepts. No audit concept can
choose a layer, penalty, support, ceiling, margin, or model pair.

The audit is concept-disjoint but uses the same four people. Its uncertainty
crosses participants and twelve audit concept blocks; it is not a population
estimate.

A robust relation must repeat every frozen endpoint. A reversal must repeat
both opposite endpoint effects and the development-only forecast. The same
intersection of at least three participants must satisfy every signed
component, and all participant and audit-block leave-one-out means must retain
the required direction. Score-specific adequacy-width thresholds distinguish
a precise nonreplication from an unresolved audit.

### 5. Keep secondary questions secondary

Held-out correlation checks whether the result depends on R². Fixed-rank
linear CKA tests one alternative linking criterion. Preregistered ROI or
semantic interactions can support a conditional relation. None may rescue a
failed primary comparison or become an independent candidate-ready terminal,
and their numerical scales are never pooled.

## Result branches

| Result | Interpretation | Next action |
| --- | --- | --- |
| Stable relation repeats on sealed concepts | The controlled pair relation is robust within the declared measurement family | Report its scope and the contracts actually covered |
| Reversal and operator prediction both repeat | A named measurement operation predictably favors one model | Build the paper around the operator–advantage mechanism |
| Reversal repeats but prediction fails | The ranking change is real, but the proposed reliability/reweighting explanation is wrong | Retain a narrower measurement-sensitivity result or stop the independent-paper direction |
| Relation differs by preregistered ROI or semantic group | A scientific stratum, rather than a generic pipeline choice, conditions the relation | Report only if the interaction and both within-stratum effects replicate |
| Development relation fails on sealed concepts with adequate precision | The relation does not transfer to new concepts in the same participants | Report nonreplication without reopening search |
| Bounds are wide or the edge is composite | The comparison does not distinguish the explanations | Report underidentification; do not call models tied |

## Figures, contingent on observed evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| **1. Three operations, three mechanisms** | What can each measurement choice mathematically change? | One voxelwise A–B advantage map; beta changes values, support changes membership, ceiling changes weights |
| **2. Is the relation stable or reversed?** | Do both edge endpoints clear practical margins? | Every contract, participant, concept block, and simultaneous interval |
| **3. Does operator–advantage alignment predict the change?** | Is the direction and magnitude explained before audit? | Reliability-gain strata for beta; exact support and ceiling decompositions; strongest falsifier |
| **4. Does the prediction survive new concepts?** | Which relation and explanation repeat without refitting? | All 120 audit concepts in twelve blocks, all four people, prediction error and uncertainty |
| **5. Where does the rule fail?** | Is the boundary measurement-related, regional, semantic, or linking-dependent? | Prespecified negative, conditional, and CKA results without post-hoc rescue |

The first figure is a synthetic question schematic. Later figures must display
complete participant and concept-block evidence rather than only an ROI mean.

## Planned conceptual figure

The main schematic should follow the logic of EP12:

- **Panel A:** the same images and two controlled models produce a voxelwise
  map of A advantage, B advantage, or no clear difference;
- **Panel B:** three isolated operations show explicitly what changes and what
  stays fixed;
- **Panel C:** development alignment gives an exact within-development
  decomposition and, separately, a frozen signed forecast for audit; and
- **Panel D:** sealed concepts distinguish stable relation, predicted
  reversal, unexplained reversal, and unresolved evidence.

Use synthetic image tiles and cortical grids, not CNeuroMod stimulus pixels or
observed neural values. Label it as a conceptual mockup with synthetic data.

## When to deepen, narrow, or stop

- Deepen the paper only when one controlled pair has an isolated relation, a
  useful operator prediction, and a valid homologous sealed-concept test.
- Narrow the claim when the relation repeats but its explanation fails, or
  when it holds only in a preregistered ROI or stimulus stratum.
- Stop the model-ranking story when pair construction is not controlled,
  support or beta edges require endpoint-specific retuning, the audit cannot
  be isolated, or precision cannot distinguish a margin from equivalence.
- Do not compensate for a failed explanatory prediction by expanding the
  model zoo or selecting another contract after audit.

## Claim language

A successful abstract should name the controlled pair, the measurement
operation, the operator map that predicted the change, and the same-participant
held-out-concept scope.

It should not claim that one model is universally more brain-like, that four
participants establish population prevalence, that the result transfers to a
new dataset, or that a training objective causally produced the neural
relation.
