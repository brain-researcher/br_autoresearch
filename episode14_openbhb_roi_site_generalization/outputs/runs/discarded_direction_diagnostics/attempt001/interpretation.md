# What the discarded-direction experiment establishes

**The Desikan performance loss persists without post-projection rescaling.
The broader claim that the removed directions carry reliably transferable age
information remains unestablished.** This is exploratory evidence from exposed
development data; the original no-candidate decision and closed holdout remain
unchanged.

## A narrower explanation of the original negative finding

The prespecified ridge penalty is 1000. Across all 416 external participants
in 13 acquisition domains, with participant-balanced appearances and equal
domain weights:

| Change | Desikan Δ MAE, years [descriptive 95% interval] | Destrieux Δ MAE, years [descriptive 95% interval] |
| --- | ---: | ---: |
| Remove four site directions, without further scaling | +0.3613 [0.0274, 0.6808] | -0.0032 [-0.4472, 0.3535] |
| Add the original post-projection scaling | +0.0145 [-0.0143, 0.0436] | +0.0157 [-0.0072, 0.0415] |
| Removal without scaling: within-domain centered error | +0.1491 [-0.1013, 0.3906] | +0.1196 [-0.0822, 0.3066] |

Thus the original Desikan loss cannot be attributed solely to the second
scaling step. Raw Desikan features have lower point MAE at all five fixed
penalties: removal costs 0.3556, 0.3542, 0.3477, 0.3918 and 0.3613 years at
alpha 0.1, 1, 10, 100 and 1000 respectively. No penalty was selected from these
evaluation results. Removing directions still changes the constrained ridge
problem, so this does not rule out regularization or redundancy explanations
generally, or establish irreversible information-theoretic loss.

Destrieux's uncertain absolute contrast is not evidence of equivalence.
Likewise, one atlas's interval excluding zero while another's includes zero
does not establish an atlas interaction or a biological difference.

## The discarded-score utility claim did not clear the key comparisons

Four-score predictors were fitted separately for site, random and permuted-site
directions, with every transformation and penalty choice learned within its
fitting role. The site-score predictors do not clearly improve on the declared
age baselines across the paired uncertainty and overlap checks.

On exactly the same nine common-support eligible domains:

| Site-score predictor minus baseline | Desikan Δ MAE [descriptive 95% interval] | Destrieux Δ MAE [descriptive 95% interval] |
| --- | ---: | ---: |
| Absolute age, versus fitting-age median | +0.5346 [-0.4025, 1.5978] | +0.2197 [-1.1157, 1.4925] |
| Centered age, versus centered zero | -0.3327 [-0.7431, 0.0771] | -0.0971 [-0.3503, 0.1997] |

Some centered comparisons favor the site directions over random or permuted
controls. Those contrasts do not substitute for benefit over the baseline,
and their favorable subsets do not establish a general utility claim. This
is failure to demonstrate the proposed joint benefit, not proof that useful
signal is absent. Even successful discarded-only prediction would not show
that retained features lack redundant age information.

## Age coverage remains a substantive limitation

Training-defined overlap retains 261 of 416 external participants across all
13 domains. The centered eligibility rule leaves 226 participants in nine
domains; the matched absolute companion uses exactly that population. Fit
support is ages [5,30) in views 0 and 1, and [5,35) in view 2. Older ages do
not survive this restriction. Domain 24 alone drops from 151 participants
spanning roughly ages 20–82 to 49 spanning 20–34.

The matched Desikan removal contrast attenuates to +0.0957 years
[-0.3098, 0.5017]. Changed population and reduced precision prevent treating
that attenuation as proof of an age-composition mechanism. All original ages,
domains, roles and views remain in the full report.

The 10,000 paired hierarchical bootstrap intervals resample domains and
participant clusters and recompute domain centers in each draw. They are
conditional descriptive intervals over observed support, not independent
confirmation, multiplicity-adjusted discoveries or a population guarantee.
Centering uses evaluation ages and is not a deployable correction.

## Novelty and the next scientific question

The information-removal/regularization distinction is already established in
[Holstege, Ravfogel and Wouters (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/26dbc8565974cffe8a44731aa09b2aa8-Paper-Conference.pdf).
The [nearest-prior review](../../../site_information_diagnostic_prior.md#2026-10-04-literature-update--novelty-boundary)
also identifies MRI work on scanner/age confounding. The contribution here is
a controlled diagnosis of this fixed OpenBHB pipeline, not a new general
principle or biological aging mechanism.

The unresolved question is whether projection harms within-site age prediction
when acquisition domains have substantial overlapping age coverage and fitting
choices are matched. A separate replication with that coverage would be more
informative than further tuning on these exposed outcomes. No successor search
or holdout opening is activated by this report.

## Validation and artifacts

Four focused tests passed. All six outer fits passed raw/restored equivalence
at every fixed penalty; maximum difference was 2.33e-12 years. The full fitted
diagnostic completed with exit 0 in 1039.11 seconds in Slurm 46578298.

The strict historical prediction replay passed 11 of 12 cached comparisons.
Destrieux R3 view 2 differs by at most 1.808989665619265e-6 years. Re-running
the unchanged legacy pipeline in the current runtime reproduced the new
predictions exactly, supporting code-path equivalence for this case. The
historical numerical cause remains unidentified. The strict failed record is
preserved; its top-level maximum covers passed comparisons only, so the failed
row and diagnosis give the relevant overall maximum. No scores, tolerances or
scientific decisions were changed to hide the difference. Its magnitude is
below the report's numerical precision.

The figure was rendered directly from the completed diagnostic report after
this diagnosis, using `make_figure(read_json(run/'diagnostics_report.json'), run)`
from the plotting module; the strict replay was not relabeled as passing.

- [Full comparison tables and coverage](diagnostics_report.md)
- [All numerical results, roles, views and fixed penalties](diagnostics_report.json)
- [Strict historical replay](baseline_replay_check.json) and [same-runtime diagnosis](numerical_replay_diagnosis.json)
- [PNG figure](../../../figures/discarded_direction_diagnostics.png) and [SVG figure](../../../figures/discarded_direction_diagnostics.svg)
- [Execution log](../../../experiment_log.md)
