# Can simple correction shrinkage reach the blend's total score?

**At the aggregate point estimate, yes.** Every one of the sixteen
other-fifteen-session calibration folds selected amplitude **0.75** for the
frozen history-only HW correction. That policy slightly exceeds the original
current/history half-blend on total velocity prediction, while losing more
trial-identity benefit. Adding the current-bin prediction is therefore not
necessary to obtain the earlier aggregate primary improvement in these exposed
recordings. This does not establish equivalence of the predictions or a
biological explanation of the blend. The component analysis below shows that
the two policies reach similar totals through different contributions.

| Fixed or calibrated policy | Primary delta R-squared | Trial-identity increment |
| --- | ---: | ---: |
| Original HW | +0.002341254 | +0.005614384 |
| Original current/history half-blend | +0.002678854 | +0.004559395 |
| HW correction multiplied by training-selected amplitude0.75 | **+0.002732680** | +0.004210788 |

Calibration improves the primary score over HW by **0.000391425** and over the
blend by only **0.000053825**. It reduces identity benefit by **0.001403596**
relative to HW and **0.000348607** relative to the blend. This is the expected
kind of tradeoff: positive scalar shrinkage attenuates trial correspondence
linearly while its squared prediction penalty changes quadratically. The
identity score is not another measure of the total velocity prediction gain.

The coordinator's [component aggregation](decomposition_aggregate.json) makes
that difference explicit:

| Policy | Direction/time mean correction gain | Net trial-varying gain | Total delta R-squared |
| --- | ---: | ---: | ---: |
| Original HW | +0.000009316 | +0.002331938 | +0.002341254 |
| Original half-blend | +0.000136223 | **+0.002542631** | +0.002678854 |
| Calibrated HW | **+0.000405753** | +0.002326927 | +0.002732680 |

The calibrated policy obtains a larger mean-correction benefit than the
blend but a **smaller net benefit for differences between trials**. This same
mean-versus-trial ordering holds in both animal averages. Compared with HW,
calibration leaves the animal-balanced net trial-varying gain slightly lower;
its higher total score comes from the improved mean correction. Consequently,
scalar shrinkage reaches the blend's overall score without reproducing its
larger trial-varying benefit. It would be incorrect to conclude that this
calibration explains away that benefit or establishes that the two predictions
contain equivalent information. Evaluation means are only descriptive
decomposition terms; they were never used to center the calibrated prediction.

Both calibrated animal means are positive: primary **0.001608142 in Mihili**
and **0.003857218 in Chewie**; identity **0.003460226** and **0.004961350**.
Primary gain over the behavioral baseline is positive in **6/6 Mihili** and
**9/10 Chewie** sessions. The negative primary session is Chewie_20150630.

The comparisons do not improve uniformly across sessions. Relative to HW,
primary improves in6/6 Mihili and4/10 Chewie sessions; both animal means and
their fixed-prediction session omissions remain positive. Relative to the
blend, primary improves in4/6 Mihili and5/10 Chewie sessions. Its small Chewie
mean advantage reverses if either20161007 or20150629 is omitted. Identity is
lower than both references in every session. Thus similar aggregate primary
scores do not imply the same trial-specific predictive benefit or a robust
new winner over the blend.

The practical requirements remain **0.01 primary** and **0.005 identity**;
the calibrated policy fails both. The original45-comparison negative closeout
is preserved. This is one separately authorized development-only comparison,
with no automatic next candidate, promotion or future-session test.

The amplitude for each scored session was selected using only the other
fifteen sessions' primary scores, with sessions averaged within animal and
then animals equally. No identity or scored-session result selected it. The
underlying HW lambda1 representation, however, was already selected during
earlier exploration of this entire corpus, and the calibration question was
motivated by those exposed outcomes. This is an informative developmental
comparison, not independent confirmation. Ordinary shrinkage is an established
statistical device; the contribution here is diagnosing this particular
prediction tradeoff, not discovering a new biological mechanism.

The [complete report](calibration.json), [session scores](sessions.csv),
[HW contrasts](contrast_HW.csv), [blend contrasts](contrast_blend.csv),
[session decompositions](decomposition_sessions.csv) and
[all direction/bin contributions](direction_bins.csv) preserve every session
and interval. Original scores replayed exactly; selected predictions are
distinct from both references, and decomposition/identity checks passed.
No neural source data or future sessions were read and no decoder was refit.

The coordinator reports that all five focused tests passed. Execution was
Slurm46578298 on`sh04-18n32.int`,2026-10-05
05:32:18.240017–05:32:21.710750UTC,exit0,process elapsed **3.470728s**,peakRSS
**211,932KiB**. This is process wall time, not allocated core-hours. The
shared allocation completed normally after1417seconds on four CPUs. Its
1.5744444444 allocated core-hours are counted once for both episodes in the
[EP14 execution record](../../../../../episode14_openbhb_roi_site_generalization/outputs/experiment_log.md#2026-10-04--discarded_direction_diagnostics--completed).
