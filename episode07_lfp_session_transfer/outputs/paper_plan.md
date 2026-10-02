# EP07 paper: does recording history preserve task structure or same-trial coupling?

Status: integrated paper design, 2026-09-30. EP07 owns the main line; EP06 is a
recording-domain component study. No neural result, new-day replication, or
scientific acceptance is claimed. The public corpus and earlier campaign work
were already outcome-exposed.

The [2026-10-02 scope review](scope_novelty_review_20261002.md) retains the
open search and history-specific pairing question. It clarifies that failure
to support a residual or pairing increment does not identify a mean-only or
geometry mechanism.

![EP07 concept schematic, not observed results](ep07_question_imagegen-v2.png)

Residual curves are schematic and the matching rows denote fixed predictions,
not measured gains. History and today-only have matched search opportunities;
the illustrated residual comparison is an example, not a closed model menu.
Unsupported pairing does not establish mean-only or geometry-only transfer.
[Exact imagegen prompts](ep07_question_imagegen-v2-prompt.md).

## The one argument this paper should make

A new day has only a nominal 20% calibration sample. Three earlier days might
help estimate the usual response to each reach, align population coordinates,
or preserve an LFP relationship to the fluctuations of a particular trial.
These are different statements. The paper first asks whether history helps
at all, beyond both a new-day direction/time mean and an equally tuned
today-only model. Only a supported primary result permits the explanatory
story.

The candidate advance is **history-specific same-trial information**: after
learning and freezing a population bridge, the history model gains more from
correct LFP--spike trial pairing than today-only does. Average trajectory
alignment or a high absolute LFP prediction score is not that result. If the
pairing increment fails with useful precision, the paper reports a boundary
on what history preserved; wide uncertainty stays unresolved.

EP06 asks a supporting but different question: which projected population
component benefits from source-animal recording-domain frequency weights,
relative to pooled and swapped weights? Its classifier is not the manuscript's
main discovery, and its component scores cannot rescue EP07's primary test.

## What the closest primary papers already cover

| Prior work | Established overlap | Remaining candidate contribution |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470163/) | This release already shows frequency- and region-dependent LFP associations with latent population dynamics, including stability across sessions and behavioral periods. Its analysis concatenates trial activity; it is not merely an average-response comparison. | Chronological history added value over matched today-only fitting, then the incremental correct-versus-mismatched residual pairing endpoint. Do not republish the basic frequency profile as new. |
| Ahmadi et al., Scientific Reports 2021, [LFP-to-spiking inference](https://www.nature.com/articles/s41598-021-98021-9) | LFP predicts several spiking measures, with LMP particularly informative; the work also examines channel count and electrode distance. | Not another good LMP decoder. Identify what historical information survives after limited target calibration and trial-identity controls. |
| Gallego et al., Nature Neuroscience 2020, [long-term latent stability](https://www.nature.com/articles/s41593-019-0555-4) | Aligned latent dynamics support stable behavior decoding despite turnover of recorded neurons. | LFP-to-native-population residual prediction and history-specific pairing, not another demonstration of stable latent trajectories. |
| Sussillo et al., Nature Communications 2016, [historical variability](https://pmc.ncbi.nlm.nih.gov/articles/PMC5159828/) | Large historical datasets and augmented neural variability already improve robustness to recording changes. | Identify the surviving information relative to an equally tuned today-only model; history or additional training rows alone are not novelty. |
| Degenhart et al., Nature Biomedical Engineering 2020, [low-dimensional alignment](https://pmc.ncbi.nlm.nih.gov/articles/PMC7822646/) | Low-dimensional spiking alignment stabilizes an intracortical BCI. | This study is an offline, target-calibrated LFP-to-spike test, not demonstrated online stabilization. |
| Karpowicz et al., Nature Communications 2025, [NoMAD](https://pmc.ncbi.nlm.nih.gov/articles/PMC12089531/) | Dynamics-based unsupervised manifold alignment improves long-timescale behavioral decoding in monkey motor recordings. | EP07 is not a competing state-of-the-art stabilization algorithm. Its narrower endpoint asks whether LFP residuals carry history-specific trial identity. |
| Hacker et al., bioRxiv 2026, [July 31 revision](https://www.biorxiv.org/content/10.64898/2026.01.03.697516v2) | In inferotemporal cortex, spike--high-gamma agreement depends on magnitude versus distributed-pattern coding. This is a preprint, not settled universal physiology. | Do not claim first component-dependent spike/LFP agreement. EP06's task projections and EP07's chronological pairing increment ask different, bounded motor-recording questions. |

This focused comparison supports proceeding with a conditional mechanistic
question, not declaring the field unsaturated or proving the exact endpoint
has never been studied. The selected public release provides an internal test
of the distinction, not an independent replication of its own source paper.

## Evidence sequence, with the original primary contract intact

### Search broadly before freezing the final comparison

The paper question does not dictate a small model shortlist. Explore the
declared feature, alignment, weighting, regularization and mapping families
adaptively on permitted development feedback. The sixteen starts are diverse
seeds, not final model nominations; successor proposals can reject as well as
refine the favored explanation. Residual predictors need matched opportunities
and the correct residual target, not a fixed architecture chosen now.

Likewise, calibration-visible mechanism indicators have no two-indicator
exploration cap. Freeze a final model and analysis only before their applicable
held-out evaluation. Broader search does not change target-day information,
fitting/scoring roles, primary estimands or practical margins, and does not
enlarge the current evaluation or compute budgets.

### 1. Establish history added value

Keep the exact six source and six target days in [GOAL.md](../GOAL.md), the
nominal 20% calibration split, native 30-ms counts, and separate M1 movement
and PMd preparation settings. History and today-only receive identical
development evaluation budgets: 16 starts, 2--14 successor pairs, and 10 paired
alternative-explanation checks, or 28--40 evaluations per stream.

Report joint native-neuron `R2_SSE` differences by target day, average the three
days within animal, and give the two animals equal weight. The margin remains
`0.01 R2` against **both** comparators; paired whole-trial uncertainty uses
9,999 draws with one-sided Holm across the two settings. Animal agreement,
day support, each target-day omission, and all required falsifiers remain
binding. Neither search patience (`0.002 R2`) nor a component score is the
primary effect threshold.

### 2. Freeze a bridge while the anchors still exist

Different days retain different neuron sets. Training and calibration
unresidualized spikes alone define each population basis, the common
direction-by-time anchors, source-to-target maps, and target reconstruction
loading. Freeze these objects before subtracting means or fitting residual
predictors. No neuron or electrode number is treated as a stable identity.

The existing rank/support gates stay unchanged: ranks 16, 12, 8, 4 in that
order; at least 64 common direction-by-time cells and four per dimension;
full centered rank; condition number at most 30; and cross-day support ratio
at least 0.05 on every source link. With centered anchor matrix `A_d = Q_d R_d`,
the ratio is `sigma_min(Q_source' Q_target) / sigma_max(Q_source' Q_target)`.
If no rank passes, the aligned explanation is unavailable for that day/region;
do not relax a gate or omit the inconvenient source. A supported bridge is a
measurement step, not evidence that residual coupling survives.

This freeze applies separately inside each candidate/fitting fold before its
residual fit or score; it does not preselect one bridge for the entire search.
Training/calibration-only bridge fitting remains unchanged. Primary success
gates the final coupling interpretation, not exploration of residual candidates
in the matched development search.

### 3. Fit a genuinely residual-only task

Estimate each day's direction/time means only from permitted training or
calibration trials and remove them from both LFP inputs and spike targets.
Fit capacity-matched history and today-only residual predictors through the
frozen bridge. The predictor receives neither the mean template nor
direction/time labels or uncentered activity. Score native-neuron residuals;
coordinate-space scores are diagnostics.

Subtracting a mean from a completed prediction and target leaves its squared
error unchanged. That recentering is not this test. The fitted residual task
and the correct-versus-mismatch comparison are both needed.

### 4. Distinguish a shared mean error from trial identity

For frozen history (`H`) and today-only (`T`) prediction sets, define:

```text
Delta_correct = R2(H_correct) - R2(T_correct)
Delta_mismatch(p) = R2(H_paired_by_p) - R2(T_paired_by_p)
pairing_increment = Delta_correct - mean_p Delta_mismatch(p)
```

Each `p` is a whole-trial derangement within the same target day, region, and
direction. Apply the identical `p` to both models; preserve all time bins;
forbid original-identity self-pairs; refit nothing. Both terms retain the
same mean estimates, map, task condition, sample size, and today-only pairing
signal. The endpoint isolates the extra dependence on identity attributable
to history, not the history model's absolute pairing score.

Use the registered 9,999 fixed-seed mismatch bank and recompute its center
inside each of 9,999 paired whole-trial bootstrap draws. Non-derangeable draws
retain the adverse infinite tail, never a convenient redraw. The residual
gain and pairing increment both need simultaneous lower bounds above
`0.01 R2`, with the original equal-day/equal-animal aggregation and Holm rule.
All eight directions need at least six held-out trials. The full numerical
contract remains in GOAL and SEARCH_POLICY.

| Explanation | Discriminating pattern | Wording permitted |
| --- | --- | --- |
| Primary gain without an identified explanation | Primary gain, but no supported separately fitted residual increment | History helps the primary prediction task; mean-only transfer is not established. A failed or imprecise residual comparison does not identify the mechanism. |
| Residual benefit with unresolved mechanism | Residual increment without the incremental trial-identity criterion | History helps residual prediction; same-trial information remains unsupported. Aligned geometry and shared mean-estimation error are compatible explanations, not identified findings. |
| History-specific coupling | Both residual and pairing increments clear their margins | Earlier days add same-trial LFP--population information in the declared setting. |
| Spike-rich high-frequency signal | Gain is confined to 100--400 Hz or spike-rich electrodes | State the frequency/recording boundary, not a general field-potential mechanism. |
| Insufficient support or precision | Bridge unavailable, disagreement, or wide uncertainty | Unavailable or unresolved; neither absence nor equivalence. |

### 5. Use EP06 to locate the recording-domain component

EP06's primary remains its residual-only frequency-profile transfer test in
both animal directions. Its separate explanatory analysis projects genuine
held-out unresidualized activity and predictions with the same frozen
common-time, direction-by-time, and trial-residual projectors. Source-animal
frequency weights are correct-domain, pooled, or exactly swapped; target
training may fit base predictors, not the combining weights.

In the combined paper, this asks **where a domain-specific frequency rule is
useful**, not whether EP07 history transfers. Keep EP06's component-energy,
nomination, support-index, abstention, and four-reserved-session rules in its
own contract. Low-frequency-only and 100--400 Hz removal, proximity and quality
controls explain signal scope. Measured matching cannot eliminate unmeasured
array differences. M1/PMd is a recording-domain contrast; EP07 additionally
confounds region with epoch. Neither contrast proves a pure cortical effect.

This is design-level integration only. The episode splits differ, so EP06
scoring must follow EP07's full freeze and one primary opening, or an approved
compatible prospective joint-role design. No sibling outcomes, fitted weights,
or candidate feedback may enter EP07 development. Do not rename an exposed
component analysis an independent holdout or animal replication.

### 6. Predict a new day's benefit, then seek real new days

Search an open pool of calibration-visible mechanism indicators, with
source/calibration trajectory agreement and source-prediction disagreement
as examples rather than a fixed list. Compare regularized forecasting rules
using day-wise development cross-fitting; indicator selection, scaling and
tuning must all exclude the day being predicted. No numerical indicator-count
cap is imposed in advance, and no future non-calibration target enters a
feature. Search breadth is not evidence strength: six target days cannot
establish a flexible benefit boundary or population-level generalization.
Inspect selection instability and simple baselines; freeze the chosen rule
before a genuinely new-day test. A concise final explanation is desirable,
but is not a prerequisite or cap on exploration.

This benefit forecast, low-frequency specificity, and extra component
diagnostics remain **planned explanatory follow-ups**, not newly activated
primary analyses or new vetoes on the original primary result. Freeze their
full procedures before the applicable outcomes are accessed. A diagnostic
selected after results belongs to development evidence and cannot confirm
itself. Only newly sequestered days can forward-test that selected explanation;
no compatible new source or third animal is currently selected.

## Figure ownership in the combined manuscript

| Figure | Main question | Evidence owner |
| --- | --- | --- |
| 1 | Does history add value beyond both matched comparators on every fixed target day? | EP07 primary, both animals/settings and all negative scores. |
| 2 | Is the surviving information a mean/geometry result or history-specific trial identity? | EP07 frozen bridge, residual refit, and paired mismatch increment. |
| 3 | Which component and signal family benefits from a recording-domain frequency rule? | EP06 component consequence plus shared signal-boundary interpretation; no independent replication wording. |
| 4 | Can calibration-visible measurements predict benefit on genuinely new days? | Separately frozen EP07 forward study, only if suitable data become available. |

EP05's figure remains about reach deviations, and EP08's about marginal
acquisition value. Do not duplicate decoder plots into four papers or count
reuse of these recordings as four independent biological demonstrations.

## When to narrow or stop

A clean primary failure stops the history-coupling story. A supported mean-only
boundary can be reported, but a routine decoder comparison may belong as a
technical result rather than sustain a paper alone. A positive residual score
without incremental pairing stays bounded. A useful EP06 component can be
included even if it is not independently publishable; a failed EP06 component
does not negate an otherwise supported EP07 primary effect.

The strongest eventual conclusion names the exact setting, frequency scope,
information that survived, calibration regime, animals/days, and evidence role.
It remains an offline result using provider-preprocessed LFP features, not
causal LFP-to-spike influence, stable cells/electrodes, an online BCI benefit,
or generalization to a new animal. Current figures are schematic and all
result-dependent manuscript branches remain hypothetical.
