# EP08: open exploration, not a compulsory pilot study

Scientist-approved design amendment, 2026-09-30. No data acquisition, neural
analysis, held-session opening, or Git mutation is activated by this document.

## Start with the scientific question

Which properties of an electrode set and a training/calibration state predict
conditional population information and the value of another observation?
Conditional complementarity and reducible uncertainty are useful hypotheses,
not the only admissible explanations or a preselected winning method.

There is **no episode-wide requirement for a separate 16-trial pilot** and no
pilot-estimation gate before other supported developmental analyses. Explore
model families, temporal summaries, representations, priors/regularization,
training/calibration amounts and coverage, and initialization strategies on
development sessions. The linear `c_e` and `g_d` scores are candidate probes,
not universal architecture or low-rank requirements. Their fitting assumptions
apply when that particular proxy is used.

The candidate space is illustrative, not exhaustive:

- static geometry/quality and development-trained electrode rules with no
  current-session response used for initial selection;
- rules based on ordinary current-session training/calibration observations,
  without treating a separate pilot as a necessary stage;
- response-adaptive policies with varying initial sample sizes and coverage;
- transfer or hierarchical models, alternative temporal summaries, and
  nonlinear representations where the available data and shared budget allow;
- conditional electrode value, direction value, sensor-dependent learnability,
  and alternatives based on quality, redundancy, coverage or balancing.

Zero-shot **selection** is not zero-shot **decoding**. A source-trained ranking
rule does not establish a cross-session population-target correspondence or
remove the decoder's need for current-session calibration.

## Two information regimes, two different claims

| Branch | Permitted fitting information | What a result could support |
| --- | --- | --- |
| Offline information-value analysis | A declared development training/calibration pool, with fitting and evaluation separated | Predictive conditional value under the stated information state; not a limited-acquisition efficiency claim |
| Sequential acquisition/replay | Static metadata, a development-trained rule, and the channel/trial features and labels actually acquired before each action | A retrospective, cost-accounted selection result; not prospective online behavioral control |

An offline analysis may reveal what is valuable with ample training data even
if a very short initializer cannot estimate it. A sequential policy may also
use source-trained priors, but cannot inspect the unacquired trial features or
targets, the discarded channels' historical non-initialization measurements,
or evaluation outcomes. Missing direction support needs a declared prior or
exploration action; an unseen next row cannot fill it.

Channel revision is not silently enabled by removing the compulsory pilot.
The reference experiment keeps its one-shot retention rule. A different
sequential action grammar must be specified before comparison, include every
actual exposure and any switching cost, and cannot backfill channels that were
not acquired on earlier trials.

## Fair comparisons without closing the search

For each developmental comparison, state the fitting pool, decision-time
information, action grammar, target, evaluation set, model-fitting procedure
and resource accounting before scoring it. These are per-comparison validity
commitments, not an early freeze of the entire search. Candidate definitions
may evolve using development feedback; prior attempts remain in the record.

When attributing a gain to selection, match decoder family/capacity, target,
training exposure and tuning opportunity within that comparison. Model-family
exploration is welcome, but a decoder upgrade alone is not a selection gain.
Offline value labels used to train a transferable rule must be session-cross-
fitted; realized labels from the session being evaluated are not policy inputs.

Count all initializer trials and their all-channel exposure, then subsequent
channels/trials, counting each acquired trial once. Compare different
initializers at matched total resources or over explicit resource curves;
a large free pilot is not a fair comparator to no-pilot selection. Keep
electrode count and trial count in separate units. Channel–trial exposure may
be reported explicitly; it is not measured power, bandwidth or participant cost.

Keep the intended four development/two held sessions per animal, whole-session
cross-fitting, and common untouched within-session evaluation. Freeze the
selected study's endpoints, comparisons, uncertainty, meaningful margins and
terminal interpretation before opening the four internal held sessions once.
Any frozen reference and new-branch analyses using that opening must be decided
together beforehand; it is not one opening per branch. No held feedback may
choose the next information state or model. Newly selected explanations after
that opening require new evidence, not a replay of the same held outcomes.

## The original protocol is an optional reference experiment

The following stay together, unchanged, as `reference_16_trial_acquisition`:
16 all-electrode trials (two per eight directions); retain 4/8/16 electrodes
once; add 32/64/128 trials with no historical non-pilot backfill; the common
reduced-rank ridge decoder; all nine cells and four factorial comparisons;
equal-cell/animal-balanced `R2_SSE`; the `0.005` margin, both animals,
three-of-four sessions and scarce-resource conditions; and its original
controls and terminal interpretations.

Those numerical decisions do **not** become universal requirements for offline
value analysis or other initializers. A changed initializer is a different
experiment; its endpoints and terminal decision need their own prospective
specification. Missing support for the old pilot/grid makes that reference
unavailable, not the whole episode uninteresting. Conversely, the reference
result cannot be rescued by switching to a different experiment after seeing
its held outcome. The v2 concept figure depicts this reference case only.

All exploration shares the existing episode-wide evaluation allowance and
resource ceilings: 40–112 valid evaluations, 20-evaluation patience, at least
40% post-coverage alternative checks, 1200 CPU-hours, zero GPU-hours, 120 wall
hours, 48 concurrent cores, 256 GB memory and 750 GB temporary storage. No
candidate, branch or initializer gets another allowance. Reference-score
patience uses the existing `0.005` improvement or new feasible frontier point;
a different endpoint needs its stopping/improvement rule before its first
scored search, within the same allowance. This amendment supplies no new
numeric endpoint, margin or winner.

The source remains absent. Two source-exposed animals support internal tests,
not external-animal generalization; Chewie-R remains the same animal. Third-
animal confirmation, biological necessity, online control and hardware savings
are not established. Exploration stays open; claims follow the supported data.
