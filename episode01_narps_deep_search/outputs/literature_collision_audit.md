# EP01 literature collision audit

Date: 2026-09-29

## Current decision

The strongest remaining use of NARPS is not another gain/loss activation map,
range-normalization analysis, choice decoder, time-on-task analysis, or generic
pipeline multiverse.

The current paper question is:

> Does the conditional NARPS gain/loss estimand remain stable, or does the map
> shift systematically when valuation, decision-state, and response variables
> are operationalized differently?

The viable contribution combines recoverable held-out behavioral mechanisms,
a controlled cognitive-model fMRI factorial, and projection of the public
many-team maps onto frozen specification-displacement directions. Any one
layer alone is too close to prior work.

Internal qualitative triage:

| Candidate contribution | Assessment after audit |
| --- | --- |
| Behavior-only DDM decomposition | direct collision; not a paper headline |
| Standard gain/loss or behavioral-neural correlation | direct NARPS collision |
| EI/ER range normalization or OFC adaptation | direct collision |
| Generic brain-to-choice cross-validation | direct NARPS collision |
| Simple recent-offer or time-on-task drift | heavily crowded |
| 70-pipeline individual-rank instability | scientifically interesting but unsupported by public group maps |
| Recoverable mechanism decomposition + controlled fMRI estimand shift + many-team method audit | plausibly strong (`~8/10`) if all identification gates pass |

The numerical rating is a triage judgment, not a measured score.

## Direct collisions around loss-aversion mechanism

| Work | What it already establishes | Consequence for EP01 |
| --- | --- | --- |
| [Zhao, Walasek, and Bhatia 2020](https://doi.org/10.1016/j.cogpsych.2020.101331) | DDM separation of asymmetric valuation, fixed utility bias, and pre-valuation rejection bias; two preregistered behavioral experiments | Fatal collision with a behavior-only decomposition; EP01 must require held-out NARPS transport and neural/analyst layers |
| [Sheng et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7260957/) | Related valuation and response components to gaze allocation and pupil dilation | Physiological dissociation is not new |
| [Wang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11245989/) | EEG/P3 signatures of valuation and response bias, with reward- and motor-related source localization; identifies fMRI anatomy as a future test | fMRI follow-up is plausible, but NARPS BOLD cannot establish temporal before-versus-during ordering |
| [Bobadilla-Suarez, Guest, and Love 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7578785/) | NARPS subjective value and inverse decision-entropy gradients; reports an in-sample behavioral/neural loss-aversion association | Generic value/confidence mapping or pooled brain-behavior correlation is occupied |
| [Brochard and Daunizeau 2024](https://doi.org/10.7554/eLife.80979) | NARPS EI/ER behavior, OFC encoding, RSA, plastic neural networks, and range adaptation | Removes EI/ER normalization, OFC geometry, and simple local-history adaptation as headlines |
| [Soch and Haynes](https://doi.org/10.1101/2022.03.24.485588) | Direct and indirect whole-brain prediction of held-out NARPS responses across runs | Held-out choice decoding itself is not novel; it can only be a criterion for mechanism/model comparison |

## Direct collisions around the many-team layer

| Work | What it already establishes | Remaining gap |
| --- | --- | --- |
| [Botvinik-Nezer et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7771346/) | 70 teams tested nine hypotheses; unthresholded maps and binary conclusions varied; original analyses related outcomes to coarse pipeline choices and used common harmonization, clustering, and image-based meta-analysis | Did not define computational valuation/decision/response reference axes or test cognitive-estimand drift |
| NARPS public result archive v2.0.1 | Supplies concatenated team maps, decisions, confidence, rich method reports, harmonization, smoothness, and generic similarity outputs | Makes the ecological test feasible but prevents novelty claims for those preprocessing and similarity steps |
| Lefort-Besnard et al. 2025 same-data meta-analysis | Models dependent multiverse maps and derives same-data consensus evidence | Consensus aggregation is not the new question |
| HCP Multi-Pipeline and recent reliability multiverses | Show pipeline effects on individual measures and group inference with richer subject-level multi-pipeline data | NARPS cannot compete on subject-rank instability without reproducing pipelines; its unique value is the task-specific cognitive estimand plus ecological team reports |

## Exact gap that survives

The original teams were nominally asked for gain/loss effects, but their
first-level descriptions vary in event duration, RT modeling, response
handling, scaling, centering, orthogonalization, and regressor set. Those
choices can alter what a gain/loss coefficient conditions on.

EP01 can therefore test a claim not resolved by the cited work:

1. use choice and RT to identify whether valuation asymmetry and a
   value-independent bias are distinguishable in this exact task;
2. vary only decision-state and response blocks inside one fixed fMRI pipeline
   to obtain empirical estimand-shift directions; and
3. ask whether team maps load on those directions in association with their
   declared cognitive specifications after recorded sample, higher-level
   model, smoothness, software, statistic, mask, and scale controls.

This is stronger than saying that pipelines differ. It tests whether some
pipelines answer different cognitive questions.

## Conditions required for publication

The full paper requires:

1. exact-design recovery of the behavioral model and its focal parameters;
2. held-out joint choice/RT/deadline prediction that distinguishes the
   proposed mechanisms from simpler and urgency/nonlinear-value rivals;
3. estimable `S11` gain/loss regressors plus reproducible marginal decision-
   and response-omission axes across nonoverlapping run halves, with
   `S00-S11` retained as the total rather than a pure process map;
4. synthetic recovery of unitless team-map shape coordinates;
5. stability to coefficient-direction sign restoration, mask, smoothness, map
   scale, statistic harmonization, and half-specific templates; and
6. method-coordinate associations that add information beyond the frozen
   sample, group-model, and map-technical factors, or 90% intervals wholly
   inside the prespecified `0.01` nat-per-coordinate and `0.02`
   multivariate-`R^2` equivalence margins. An imprecise null is not evidence
   that cognitive operationalization was unimportant.

If behavior alone passes, the result is not yet paper-level. If behavior fails,
the BOLD and team-map decomposition stop. If the team layer fails after the
controlled fMRI layer passes, the output narrows to a controlled estimand
sensitivity paper.

## Claim constraints imposed by the audit

- A DDM starting point is a computational parameter, not direct proof of a
  temporally pre-valuation neural process.
- Strong/weak response is response strength, not an independent confidence
  judgment.
- Response-category regressors are confounded with the fixed four-finger
  mapping and cannot be labeled pure motor or pure response bias.
- EI/ER is between participants and not an independent replication.
- Team maps and templates draw from the same participant pool, subject to
  team-specific exclusions, and are not independent biological samples.
- Team methods were self-selected, so metadata associations are descriptive,
  not causal.
- Public group maps cannot support 70-pipeline individual rankings.

## Retired methods-direction audits

Before the current scientist-selected pivot, EP01 considered multiscale
smoothing and then derivative sufficiency. Those decisions remain useful
negative knowledge but are no longer active work.

### Smoothing headline

| Prior work | Collision |
| --- | --- |
| [Mikl et al. 2008](https://doi.org/10.1016/j.mri.2007.08.006) | Compared pre-GLM and post-GLM contrast smoothing over 2--30 mm and discussed OLS versus ReML/prewhitening noncommutation |
| [Hagler, Saygin, and Sereno 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1785301/) | Compared smoothing time courses with smoothing coefficient maps |
| [Worsley et al. 1996](https://pubmed.ncbi.nlm.nih.gov/20408187/) | Established neuroimaging scale-space inference |
| [Ball et al. 2012](https://pubmed.ncbi.nlm.nih.gov/21404370/) | Used 13 widths from 4--16 mm and found scale-dependent results |
| [Sacchet and Knutson 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3618861/) | Reward-domain reanalysis over 0--12 mm showed smoothing-dependent localization |

Conclusion: a `{4,6,8,10}` profile, pre/post timing comparison, or smoothing
commutator is too saturated for the headline.

### Derivative-sufficiency headline

| Prior work | Collision or boundary |
| --- | --- |
| Beckmann, Jenkinson, and Smith 2003 | Fixed multilevel GLMs can be reproduced from parameter estimates and covariances under stated conditions |
| Rohde et al. 2005 | Image transformation uncertainty depends on covariance and the operator; `K Sigma K^T` is established |
| Mumford and Nichols 2009 | Ordinary group OLS on contrast maps can be valid without first-level precision |
| Chen et al. 2012 | Precision-weighted fMRI group analysis is established |
| Maumet et al. 2016 | NIDM-Results already formalizes rich statistical-result sharing |
| Germani et al. 2025 | Heterogeneously processed contrast reuse can inflate false positives |

Conclusion: an operation-indexed derivative envelope with a useful covariance
repair remains technically defensible, but the user explicitly selected the
broader and less saturated NARPS cognitive-estimand question for formal EP01.
