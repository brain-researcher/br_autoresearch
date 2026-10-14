# What linear age adjustment changed

**This fixed intervention did not establish an age-prediction benefit beyond
the within-site shuffled-age control. It also illustrates why residualizing
the classifier's training inputs does not protect age covariance when its
directions are removed from original ROI features.** These are exposed
development results; the original no-candidate decision and closed public
holdout remain unchanged.

Each outer fit estimated ROI~intercept+linear age using training rows only,
then trained the unchanged site classifier on those residuals. Its four
directions were removed from original standardized ROIs, with no further
scaling or evaluation-age input. The sham used one fixed within-site age
permutation. Every downstream age predictor used actual training ages.

## Prediction benefit was uncertain and not specific to actual age pairing

At the fixed alpha1000 anchor, across all 416 external participants in
13 domains:

| Contrast, Δ MAE in years | Desikan [descriptive 95% interval] | Destrieux [descriptive 95% interval] |
| --- | ---: | ---: |
| Actual-age adjusted − original: absolute age | -0.1599 [-0.4286, 0.1358] | +0.0594 [-0.2917, 0.5444] |
| Actual-age adjusted − sham: absolute age | -0.1547 [-0.3981, 0.0668] | +0.0147 [-0.2500, 0.2542] |
| Actual-age adjusted − original: centered age | -0.0818 [-0.2914, 0.0827] | -0.1402 [-0.3134, 0.0374] |
| Actual-age adjusted − sham: centered age | +0.0025 [-0.1368, 0.1642] | +0.0097 [-0.1152, 0.1238] |

Both actual and sham adjustments reduced centered error at the point-estimate
level, with similar all-row changes. This does not establish a benefit
specific to the actual within-site age/ROI pairing. It is also not evidence
of equivalence or proof that useful age variation is absent.

Restricting to the same nine centered-eligible common-support domains
(226 participants) does not resolve that comparison. Actual minus sham
absolute MAE is +0.0054 [-0.1596, 0.1910] years for Desikan and
-0.0007 [-0.1694, 0.1707] for Destrieux; centered differences are
-0.0628 [-0.1722, 0.0586] and -0.0502 [-0.1826, 0.0814]. This support
contains only younger ages, below 35, and cannot answer the older-age question.
All original participants and domains remain visible in the all-row results.

## Residual training inputs did not produce age-protected directions

For original standardized fitting features Z and true fitting ages, define
c=Cov(Z,age). The diagnostic below is ||V c||²/||c||² for orthonormal
direction rows V: the squared alignment of the marginal age-covariance vector
with the removed subspace. It is **not** a fraction of age information,
explained age variance, or biological signal.

| Atlas/view | Original directions | Actual-age-adjusted directions |
| --- | ---: | ---: |
| Desikan 0 | 0.028014 | 0.030836 |
| Desikan 1 | 0.022602 | 0.027920 |
| Desikan 2 | 0.027105 | 0.029336 |
| Destrieux 0 | 0.003580 | 0.044970 |
| Destrieux 1 | 0.001860 | 0.038851 |
| Destrieux 2 | 0.011905 | 0.025644 |

Actual-age adjustment increased this alignment in every one of the six fits.
The fitted residual matrix is orthogonal to training age in participant
space, but the classifier's feature directions need not be orthogonal to c.
Applying those directions to original Z therefore can still remove its
marginal age covariance. This is a concrete counterexample to treating the
residualization step as an age-preservation guarantee. It does not establish
a new theorem, biological mechanism, or a monotonic relationship between this
alignment and prediction error. An explicitly covariance-protected projection
was not part of this comparison.

The ordinary seen-site probes also give no clear separation between actual
and sham adjustments: their balanced-accuracy contrasts are
-0.000480 [-0.014632, 0.010127] for Desikan and
+0.012483 [-0.002090, 0.035475] for Destrieux. These are unconditional site
probes on the same unrescaled representations. They cannot isolate scanner
effects from site-associated age or other cohort differences.

## Scope and verification

The single sham preserves each site's age distribution and breaks individual
within-site age/ROI pairing. It does not remove between-site age composition,
and it is an active control rather than a permutation null distribution.
The 10,000-draw paired bootstrap describes evaluation uncertainty conditional
on these fitted models, this shuffle and observed domains; it does not cover
training or shuffle uncertainty. The overlapping views and two atlases use
the same exposed cohort. Common-support centering is evaluator-only and not
a deployable age correction. All five fixed penalties are reported without
selecting one from evaluation results.

All 60 raw/original prediction comparisons against the preceding diagnostic
match exactly, with maximum absolute drift zero and identical row metadata
and support. The representative Destrieux view 0 checkpoint took 18.626 seconds;
the remaining fits and aggregation took 101.108 seconds in Slurm 46581828 on
sh04-18n31.int, with successful exit. These are process durations, not allocated
resource charges; shared allocation accounting belongs to the experiment log.

See the [complete tables](age_adjusted_report.md),
[numerical results](age_adjusted_report.json),
[reference parity](reference_parity.json),
[figure](../../../figures/age_adjusted_projection.png), and
[prospective specification](../../../site_information_diagnostic_prior.md#one-fixed-linear-age-adjustment-intervention).
