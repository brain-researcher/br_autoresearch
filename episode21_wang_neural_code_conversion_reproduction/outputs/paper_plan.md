# EP21 paper plan: can neural codes be converted across people and imaging sites without shared stimuli?

Status: adapted-reproduction paper plan, narrative refreshed 2026-10-02.
This document describes the scientific comparison and its interpretation,
not a replication result or a live execution report. This writing update grants
no allocation, protocol lock or data access. See the
[scope review](scope_novelty_review_20261002.md) for the scientific boundary
and the [execution ledger](experiment_log.md) for current authorized stages,
attempts and next actions.

## The question

![EP21 primary adapted reproduction](ep21_neural_code_conversion_design-v3.png)

Design schematic, not results: disjoint target-decoder and source-converter
training sets, a content loss through the fixed target decoder, and common
test images/readouts for the within-source, content-loss and brain-loss methods.
Only brain loss uses paired shared training responses; unequal training sets
prevent an isolated objective-effect interpretation. The no-shared claim is
about training image IDs, not clean upstream provider measurements. Twenty
directed pairs remain dependent observations of five participants. External
sites are a contingent extension; OOD remains closed.
[Image-gen prompt](ep21_neural_code_conversion_design-v3-prompt.md).

Can a converter trained only on one person's images map that person's fMRI
activity into another person's neural space well enough for the second
person's already-trained visual decoder to recover stimulus content?

[Wang et al. (2025)](https://doi.org/10.1038/s43588-025-00826-5) answered this
question with a content-loss objective: converted activity is passed through a
fixed target decoder, and the converter is optimized so that the decoded VGG19
features match the true features of the source stimulus. Target-decoder
training and converter training need no shared images. The source paper further
reported natural-image reconstruction and successful conversion across sites.

EP21 asks whether those patterns survive an adapted LAION-fMRI test. The
primary analysis uses five participants and all 20 directed source-to-target
pairs. A separately provisioned inter-site module may extend the design to
LAION↔NSD and LAION↔THINGS. The paper must make the training roles visible:

- the target decoder learns from the target participant's 4,712 subject-unique
  regular images;
- the content-loss converter learns from the source participant's different
  4,712 subject-unique regular images;
- the two training image sets have zero exact overlap; and
- only the brain-loss comparator uses shared training stimuli, on the fixed
  `tau` train split, with evaluation restricted to disjoint `tau` test images.

The contribution is an adapted reproduction of the source pattern in this
fixed LAION setting, not invention of no-shared-stimulus conversion. The
no-shared claim concerns decoder/converter training image IDs; it does not
establish independence of all upstream provider measurement estimates. A
negative or mixed result can define a reproduction boundary if the disclosed
adaptations, failures and source-participant/dyad uncertainty are reported.

## Corrected source and implementation anchor

The initial EP21 scaffold described a different paper because the wrong
proposal was supplied. The scientist corrected the proposal before any EP21
empirical access, fitting, scoring, or job submission. This is therefore a
prospective identity correction, not an outcome-responsive protocol change.
The superseded scaffold's question, models, outcomes, and adjudication rules
are not part of this paper.

The corrected source is:

> Wang, H., Ho, J. K., Cheng, F. L., Aoki, S. C., Muraki, Y., Tanaka, M.,
> Park, J.-Y., & Kamitani, Y. (2025). *Inter-individual and inter-site neural
> code conversion without shared stimuli*. Nature Computational Science, 5,
> 534–546. [https://doi.org/10.1038/s43588-025-00826-5](https://doi.org/10.1038/s43588-025-00826-5)

The implementation target is the official
[GitHub V1.0.0 release](https://github.com/KamitaniLab/InterSiteNeuralCodeConversion/releases/tag/V1.0.0)
(commit `edbff02edd6a08d3c6c829af45a957fd14bdc460`) and the paper-linked
[Zenodo archive, record 14910040](https://doi.org/10.5281/zenodo.14910040).
The expected archive MD5 is `bb80dcc1a038737d6d8fc2c95b3e5586`.
Qualification must verify the archive contents before the protocol lock; a
moving default branch is not a reproduction identity.

On 2026-09-30 the scientist adopted the V1.0.0 code-faithful choices: tagged
LeakyReLU(0.2), normalized active-block content-loss weighting, all 19
reconstruction layers and the tagged PyTorch AlexNet/227-pixel evaluator.
These disclose paper/code discrepancies rather than leave a post-outcome
choice. The existing implementation-binding record also fixes FastL2LiR0.9
semantics, prospective BdPy0.25 and the public PyTorch true-feature definition;
historical Caffe parity remains unverified. These decisions are not a passing
runtime report or permission to fit a neural model.

## Planned authorship

| Planned author | Affiliation supplied with the proposal | Contact |
| --- | --- | --- |
| Xinhui Li | Tri-institutional Center for Translational Research in Neuroimaging and Data Science (TReNDS), Georgia State University, Georgia Institute of Technology, and Emory University | [xinhuili@gatech.edu](mailto:xinhuili@gatech.edu) |
| Zijiao Chen | Stanford University | [zijiao@stanford.edu](mailto:zijiao@stanford.edu) |

This is a planning record, not a final authorship or contribution statement.
The manuscript should replace it with author-approved affiliations,
contributions, and order before submission.

## The primary LAION-fMRI experiment

### The no-shared content-loss path

For each directed pair `source -> target`:

1. Fit the target participant's VGG19 feature decoder on that target's 4,712
   subject-unique regular images.
2. Freeze the target decoder.
3. Fit the nonlinear content-loss converter on the source participant's
   different 4,712 subject-unique regular images.
4. For each source training image, map source activity into target neural
   space, decode that converted activity with the fixed target decoder, and
   compare the decoded representation with the true VGG19 representation of
   the source image.
5. Evaluate the frozen decoder–converter chain on the authenticated shared
   regular `tau` test images.

No target neural response to a source training image enters the objective. The
source converter images and target decoder images must have zero exact image-ID
overlap, and neither set may contain a `tau` test image. That separation is the
scientific core of the paper, not a bookkeeping detail.

The primary LAION analysis contains 20 directed dyads because each of five
participants serves once as source for each of the other four targets. Direction
matters: `A -> B` and `B -> A` use different decoders, converter losses, and
training responses and remain separate observations.

### The shared-stimulus brain-loss comparator

Brain-loss conversion is a distinct comparator. It predicts target activity
from source activity and therefore requires both participants to have viewed
the same training images. It does not test the no-shared claim.

The comparator is fitted only on the authenticated shared regular `tau` train
split and evaluated only on its disjoint `tau` test split. The structural
metadata authenticates variant 0 with 897 train and 224 test images.
All repetitions of an image stay on one side of the split. No test response,
decoded feature, reconstruction, or identification score may tune the mapping.

The main comparison places within-individual decoding, no-shared content-loss
conversion, and shared-stimulus brain-loss conversion on identical eligible
`tau` test images. This is a fair held-out comparison without implying that all
three methods used the same training information.

In particular, 4,712 unique-image content training versus 897 shared-image
brain-loss training compares training packages, not loss objective alone.
The optional controls below retain their separate role and do not retroactively
make the primary contrast an objective-only experiment.

If locked before scoring, a separately labeled sensitivity may fit content
loss on the source side of the same 897 `tau` train images, without target
responses, together with a fixed 897-image source-unique sample-count control.
Neither sensitivity replaces the 4,712-image primary content model or counts
as evidence for the no-shared claim.

## What each finding asks

### Finding A — does converted activity preserve VGG19 feature information?

For every directed LAION dyad and prespecified VGG19 layer, report pattern
correlation across feature units and profile correlation across test images.
Show within-individual, content-loss, and brain-loss conditions on the same
held-out support.

The source pattern has two parts: content-loss conversion should exceed the
brain-loss comparator, and it should be close to within-individual decoding.
The first is a directional contrast. The second is a comparability claim and
requires a prospectively locked non-inferiority or equivalence margin. If no
scientifically defensible margin can be fixed before scoring, report estimates
and intervals without treating confidence-interval overlap or a nonsignificant
difference as evidence of comparability.

Current disposition, adopted 2026-09-30: comparability subclaims for A, C and E
are `not_evaluable` because no outcome-independent scientific margin was
supplied. Absolute estimates and signed method differences remain reportable.
The conditional margin language below describes the evidentiary requirement,
not an available positive-comparability branch under the current contract.

### Finding B — do the natural-image reconstructions retain recognizable content?

Use the authenticated source reconstruction pipeline to reconstruct fixed
regular natural test images from within-individual, content-loss, and brain-loss
decoded VGG19 features. Show a nonselected image set chosen by an
outcome-independent rule, and publish the complete reconstruction manifest so
failed and missing images cannot disappear from the display.

Finding B is qualitative. Visual resemblance is not a quantitative replication
verdict. Unless a blinded rating or quantitative criterion is locked before any
reconstruction is inspected, the paper may describe visible examples but must
reserve inferential language for Finding C.

### Finding C — can the natural-image reconstructions be identified?

Evaluate the Finding B reconstructions with the fixed AlexNet feature hierarchy
and pairwise identification procedure. Prespecify the candidate universe, true
candidate, feature layers, correlation, averaging order, and tie rule. Report
all layers rather than selecting the layer with the highest accuracy.

Finding C is the quantitative reconstruction endpoint. Comparability with
within-individual reconstruction again requires a locked margin; performance
above chance and performance equivalent to within-individual are different
claims.

### Finding D — does the reconstruction approach transfer across sites?

If NSD and THINGS qualify, fit no-shared content-loss converters across
LAION-fMRI and each external dataset. The target decoder is always trained on
the target participant's native training images, while the converter is trained
on the source participant's native training images. Exact training-image
overlap must be absent or removed by a fixed outcome-blind rule. There is no
invented inter-site brain-loss baseline when paired shared-stimulus responses
do not exist.

Display fixed, nonselected inter-site reconstructions separately by conversion
direction and target dataset. Finding D is qualitative under the same rule as
Finding B.

### Finding E — can inter-site reconstructions be identified?

Apply the locked AlexNet identification procedure to Finding D reconstructions.
Report results separately for LAION→NSD, NSD→LAION, LAION→THINGS, and
THINGS→LAION, as well as the declared aggregate. A target-dataset or conversion-
direction effect must remain visible rather than being averaged into a single
favorable number.

Finding E is the quantitative inter-site endpoint. A statement that inter-site
performance is comparable with a native within-dataset reference requires
matched evaluation support and a locked margin.

## Reporting the biological and dyadic structure

The paper should report the data at two complementary levels.

First, show every directed dyad. A 5×5 source-by-target matrix with the diagonal
blank makes directionality, missing pairs, and heterogeneous targets visible.
Second, collapse each source participant equally over its four eligible targets
and show all five source-participant summaries. Their equal-weight mean is the
primary biological summary. This reveals whether a group-looking effect is
carried by one source participant and avoids counting the same within-individual
reference four times without disclosure.

Neither display makes the 20 dyads independent. Each participant recurs as a
source and a target. For source fidelity, a secondary descriptive analysis
follows the source paper's dyadic bootstrap structure: 1,000 resamples of
source and target identities separately, with invalid same-participant pairs
excluded and the full summary recomputed. It does not replace the five-source
primary or support population generalization. The paper reports the resampling
algorithm, effective pair count, missing-pair rule, point estimate, and 95%
interval.

Any added within-versus-converted permutation test must lock its exchangeable
unit, treatment of repeated source and target identities, sidedness, layer
family, and multiplicity correction. Images, feature units, voxels, layers, and
bootstrap draws are not biological replicates.

Findings A, C, and E receive quantitative result dispositions. Findings B and D
receive descriptive, qualitative dispositions unless a valid assessment rule
is locked prospectively. The paper reports findings separately and does not
collapse them into one episode-wide success label.

## The provisioning-contingent inter-site module

With five LAION-fMRI, four NSD, and three THINGS participants, the planned
bidirectional module contains 70 directed pairs:

```text
LAION -> NSD:     5 x 4 = 20
NSD -> LAION:     4 x 5 = 20
LAION -> THINGS:  5 x 3 = 15
THINGS -> LAION:  3 x 5 = 15
Total:                       70
```

NSD↔THINGS pairs are outside this adapted module. Seventy is a planning maximum,
not a guaranteed sample size. Each dataset, participant, direction, native ROI,
training/test role, decoder input, and license must qualify independently. The
manuscript must show the planned and realized pair tables and explain every
exclusion.

The inter-site module does not block the primary 20-pair LAION result. If no
external handoff is provisioned, Findings D and E are `not_evaluable` and the
paper becomes an inter-individual LAION reproduction with an explicitly absent
inter-site extension. Missing external data must not be replaced after seeing
the within-LAION results.

## The blocked artificial-image analogue

The proposal suggested using LAION-fMRI abstract-shape or illusion stimuli as
an analogue of the source paper's artificial shape/color tests. Those stimuli
fall within the 371-image OOD pool conditionally reserved for EP04. The analogue
is therefore non-executable under the current EP21 contract.

EP21 has no OOD input interface and may not inspect raw OOD images, derived
VGG19 or AlexNet features, neural responses, normalization or reliability
statistics, reconstructions, predictions, scores, or summaries. It also may not
read live EP04 outputs or legacy EP04 Scratch. The final paper records this
branch as `blocked_by_ep04_reservation` or
`not_evaluable_under_current_contract`; it must not quietly omit the proposed
analogue or infer OOD robustness from regular images.

Opening the branch later would require an explicit cross-episode scientific
decision, a role-isolated handoff, and a regenerated protocol lock before any
access. Contract prose alone cannot relax the boundary.

## Study sequence

### 1. Qualify sources and official assets

Authenticate the LAION release, participant/image roles, `tau` split, native
ROIs, GLMsingle beta representation, licenses, data-risk disposition, and prior
exposure. Pin the official V1.0.0 code identity and the VGG19, AlexNet, target-
decoder, converter, generator, and DISTS assets. Record verified absence rather
than substituting an unauthenticated component.

### 2. Freeze and falsify the implementation

Resolve the decoder, converter, preprocessing, reconstruction, identification,
bootstrap, permutation, multiplicity, and comparability-margin fields. Test
shape handling, split enforcement, zero-overlap checks, leakage guards, metric
direction, and interval construction on synthetic or permuted fixtures. Emit a
protocol lock before mounting a neural scoring handoff.

### 3. Fit target decoders and no-shared converters

Fit one native VGG19 decoder per LAION target on its 4,712 target-unique images.
Then fit each directed content-loss converter on its source's different 4,712
source-unique images. Preserve fitted-object and failure manifests for all 20
pairs.

### 4. Fit the brain-loss baseline separately

Fit the linear shared-stimulus comparator only on `tau` train responses. Keep
its training records and interpretation separate from the no-shared content-
loss path.

### 5. Evaluate A–C without selection

Run feature decoding on the common `tau` test support, reconstruct the fixed
natural-image display set, and compute identification for the complete declared
set. Report dyad and source-participant results before aggregate summaries.

### 6. Run D–E only if the external branch qualifies

Fit and evaluate the fixed 70-pair maximum inter-site module without using
within-LAION outcomes to select datasets, directions, ROIs, or stopping.

### 7. Integrate results and absences

The primary `laion_report` integrates qualification, deviations, A–C, the
blocked artificial analogue, external availability, prespecified sensitivities,
and claim limits without waiting for absent external inputs. If the external
branch is locked runnable, the final manuscript synthesis must additionally
consume the separately exported D and E reports and then present A–E in order.
Technical failure, data absence, qualitative evidence, quantitative
non-replication, and policy prohibition remain distinct outcomes.

## Planned figures and tables

| Item | Question it answers | Required content |
| --- | --- | --- |
| Figure 1. Design and evidence boundary | How can content-loss training avoid shared stimuli? | Target-decoder and source-converter training sets, frozen decoder, feature-space loss, separate brain-loss path, 20-pair and contingent 70-pair modules, and locked OOD branch |
| Figure 2. Finding A | Does converted activity preserve VGG19 information? | Layer-wise pattern/profile correlations, all dyads, five source summaries, within/content/brain conditions, intervals, and any locked margin |
| Figure 3. Findings B–C | What information is visible and identifiable in natural reconstructions? | Fixed nonselected reconstruction grid, failures/missingness, AlexNet layer identification, dyad/source summaries, chance and margin references |
| Figure 4. Findings D–E | Does the result extend across sites? | Four conversion directions, realized pair counts, fixed reconstruction displays, identification by target site/layer, and unavailable directions |
| Figure 5. Boundaries and robustness | Which interpretations survive the fixed checks? | Pair heterogeneity, protocol deviations, exposure, unavailable assets, blocked OOD analogue, and permitted claim summary |
| Table 1. Data-role contract | Which images train each object and which images test it? | Participant/site counts, target-decoder set, source-converter set, `tau` train/test, exact-overlap checks, and OOD exclusion |
| Table 2. Finding ledger | What happened to every planned finding? | A–E endpoint, evaluability, effect/interval, margin status, qualitative versus quantitative status, and bounded conclusion |
| Table 3. Reproducibility record | Can another team identify the exact implementation? | Code/archive IDs, checkpoints, preprocessing, ROI definitions, pair table, exclusions, seeds, failures, and deviations |

Figures are contingent summaries, not promised positive results. A failed or
non-evaluable branch remains visible in the final sequence.

## Limitations and rival explanations

### Adaptation rather than exact reproduction

LAION-fMRI differs from Deeprecon, THINGS, and NSD in participants, stimuli,
scanner/acquisition, GLMsingle preprocessing, available masks, and training
sample structure. A positive result shows transfer of a method pattern, not
numeric identity with the source paper.

The adopted primary ROI is the fixed union of 29 independent
retinotopy/fLoc/object masks per participant: adapted, incomplete visual
support, not equivalence to the source whole visual cortex. The scientist
also accepted fixed released GLMsingle v1.2 final TYPED regular-trial
measurements despite unverified upstream exclusion of tau-test/OOD influence.
That inherited dependence stays disclosed; no clean-measurement or cancellation
claim follows. All direct OOD access remains closed and EP21-fitted transforms,
voxel ranking, tuning and stopping remain training-only. This manuscript does
not itself amend provider-measurement eligibility; separately recorded scientist
amendments and operative contracts govern that decision.

### Small and dependent biological sample

Five LAION participants generate 20 directed pairs, but repeated source and
target identities prevent treating those pairs as independent people. Dyadic
resampling addresses dependence in the reported uncertainty; it does not turn
five participants into a population-representative sample.

### Prior LAION outcome exposure

Regular shared-image neural outcomes were previously opened in the EP04
lineage. EP21 is prospective relative to its corrected fixed procedure but
retrospective and correlated relative to the campaign evidence. It is not a
fresh independent confirmation and cannot validate EP04.

### Feature-space dependence

The converter is optimized through a VGG19 decoder. Success may reflect the
chosen feature hierarchy, target decoder, voxel-selection rule, or
reconstruction prior rather than a universal neural code. The manuscript must
keep claims tied to the authenticated pipeline and declared visual ROIs.

### Qualitative reconstruction bias

Reconstructions are easy to cherry-pick. Fixed image selection, complete
manifests, and separation of B/D from quantitative C/E prevent attractive
examples from substituting for measured accuracy.

### Comparability is stronger than failure to reject a difference

Overlapping confidence intervals and nonsignificant contrasts do not establish
equivalence. Without locked margins, the manuscript may say that estimates are
close in this sample but not that methods are statistically comparable or
equivalent.

### Inter-site interpretation

Site, dataset, participant pool, image distribution, preprocessing, mask, and
decoder training all change together. Even a successful 70-pair module does not
isolate scanner effects or establish universal site invariance.

### No artificial-image or OOD evidence

The EP04 reservation prevents the proposed artificial analogue. The manuscript
must state this directly; natural-image performance cannot fill that gap.

## Permitted claims

| Evidence pattern | Permitted interpretation |
| --- | --- |
| Finding A favors content loss over brain loss on the locked test set | No-shared content-loss conversion preserves more VGG19-decodable information than this shared-stimulus comparator in the evaluated LAION dyads |
| A, C or E requests formal comparability under the current no-margin disposition | The comparability subclaim is not_evaluable; report absolute estimates and signed differences without an equivalence claim |
| A/C are positive but no valid margin was locked | Converted information is measurable and its distance from within-individual performance is estimated; formal comparability is not claimed |
| B shows recognizable fixed examples | The qualitative displays illustrate retained content; quantitative support comes only from C |
| The qualified D/E module is positive in all declared directions | The content-loss approach transfers across these LAION, NSD, and THINGS participant/dataset directions under the fixed pipeline |
| D is visually encouraging but E is weak | Inter-site examples are descriptive; the quantitative identification claim is unsupported |
| A–C are mixed or null | The corresponding Wang pattern does not reproduce under this adapted LAION design; do not search the same outcomes for a replacement method |
| D/E are unavailable | The paper makes no inter-site empirical claim and reports the provisioning or qualification failure |
| Artificial analogue remains blocked | No artificial-image or OOD claim is made |

Even a uniformly favorable result does not establish exact reproduction,
population generalization, universal representational identity, scanner-only
causality, brain-to-brain communication, or OOD robustness. It also does not
constitute independent confirmation of EP04.

## Current stopping rule

The existing episode contracts and separately recorded scientist authority
govern execution; the execution ledger records stage completion and attempts.
Provisioning or a technical repair does not itself authorize a protocol lock,
neural fitting or scoring. The existing scientific and data-role gates apply
before a real scoring handoff. This paper narrative changes no runtime, code,
resource cap, stopping rule or access boundary.

If image-role separation, `tau` integrity, authentic code/assets, calibrated
inference, or the physical OOD firewall cannot be established, stop or mark the
affected branch `not_evaluable`. Do not weaken the no-shared definition, use a
moving implementation, or substitute an attractive method after outcomes are
visible.
