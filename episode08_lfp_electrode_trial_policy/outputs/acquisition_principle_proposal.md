# Proposed rule: conditional complementarity and reducible uncertainty

Design proposal, 2026-09-30. **Unactivated; no neural data, rule score, or
future-value label has been examined.** Following the scientist's open-search
amendment, these are candidate explanatory proxies, not a compulsory model or
pilot stage. The [open design](open_exploration_design.md) permits other models,
summaries and training/calibration information states. The original 16-trial
grid/decoder and `0.005` decision remain together as an optional reference;
session roles, shared budgets and held-out separation remain unchanged.

## A modest, falsifiable acquisition principle

At similar quality, retain the electrode whose allowed fitting signal adds the most
target-linked variation **conditional on what is already retained**. With that
set fixed, sample the direction expected to reduce uncertainty in the fitted
mapping, rather than the direction with the largest total error. Recomputing
the second score after a quality-matched electrode swap predicts whether the
next-direction ordering changes. Those predictions, not the optimizer's name,
are the proposed scientific object.

The wording is descriptive. Conditional sensor objectives and predictive
variance reduction are established methods, not EP08 inventions:
[Krause, Singh and Guestrin, JMLR 2008](https://jmlr.org/papers/v9/krause08a.html)
studied information-based sensor selection;
[Cohn, Ghahramani and Jordan, JAIR 1996](https://arxiv.org/abs/cs/9603104)
studied statistical active learning. Earlier LFP work already performed
[whole-electrode dropping and marginal-correlation selection](https://pmc.ncbi.nlm.nih.gov/articles/PMC3429374/),
and [active-learning decoder recalibration](https://doi.org/10.1016/j.compbiomed.2025.110231)
is not new. EP08's proposed addition is a bounded empirical test of whether
these pre-choice quantities predict **later value and sensor-dependent
trial ordering**, beyond quality, marginal target association, counts, and
noise, in this source. It is not a demonstrated general law.

## Estimator support depends on the information state

The fitting pool can be ordinary permitted training/calibration data; it need
not be a separate pilot. Compare sample amount, coverage, temporal summaries,
regularization and model families in development rather than fixing them all
now. In sequential use, observations must be acquired before the score; in
offline value analysis, a declared training pool may be available at once.
Neither future features nor evaluation labels are permissible score inputs.
Target-linked response covariance cannot be calculated from no response data;
a no-current-response selector needs a static or source-trained rule, not an
imaginary zero-shot `c_e` estimate. Missing direction support likewise needs a
declared acquired-data/prior treatment, never a look at the next trial.

For the **optional 16-trial reference example**, start with the same two
acquired reaches in each of eight directions. One simple proxy summarizes
each electrode's three existing features and
each scaled population target over the fixed 150–450 ms window, one row per
whole trial. This proxy may miss dynamic information; the final decoder still
uses the original representation and capacity for that reference. Other
developmental summaries are allowed; fit/scoring roles must be specified before
their comparison, and final summaries locked before held evaluation.

- All centering, target scaling, projections, noise estimates, and
  regularization use fitting trials only. Held evaluation targets are never
  involved. Target coordinates and scaling must correspond to the same fixed
  population target used by the common decoder.
- Any learned projection has rank at most `n_fit_trials - 1`, and must have a
  smaller development-chosen cap; a 16-trial pilot is not a license to invert
  an unregularized 48-plus-dimensional covariance. Positive ridge/shrinkage
  protects every inverse. The prior/shrinkage family is shared across sessions
  and selected only on development data.
- Cross-fitting holds out whole trials while retaining all eight directions
  in each fitting fold. The two pilot trials per direction support two
  direction-balanced folds, each with only eight fitting trials. Within-trial
  time bins are not independent pilot samples. Fold-specific projections and
  scaling are learned again inside the fold, and scalar scores are averaged,
  not pooled as if their learned coordinates matched.
- With only two repeats per direction, direction-specific residual variance
  is weakly identified. Use a prospectively chosen pooled/shrunk model, report
  ranking uncertainty and instability, and do not interpret a precise
  direction-noise decomposition as observed fact.

The bullets describe that small-sample linear candidate, not all models. The
number sixteen is an optional reference constraint, not a supported claim of
sufficient information. If predictive rankings cannot be estimated or do not
repeat, that is a limitation under this initializer, not a reason to stop other
supported information-state or offline-value exploration.

## Electrode score: conditional, target-linked complementarity

For candidate electrode `e` and retained set `S` not containing it, residualize
the three electrode features and scaled targets against `X_S` using the
regularized fitting-only proxy. A descriptive score is:

```text
c_e(S) = trace[C_Ze|S (C_ee|S + lambda_e I)^(-1) C_eZ|S]
```

Here the `C` terms are shrinkage estimates of residual cross-covariance or
covariance; fitting-pool-appropriate cross-fitting and development-only calibration
address their optimism. This is a linear explained-variance proxy, **not
Shannon mutual information, a causal effect, or an unbiased value estimate**.
Large raw variance alone is not enough: unshared sensor noise can have large
conditional variance without target-linked complementarity.

For a removal prediction in final set `S`, compute `c_e(S without e)`. Test its
ordering against the existing realized removal-loss label `V_e` using the
same calibration data and common decoder. For a quality-matched swap, make
the forward prediction before observing the swapped set's evaluation loss.
Reliability alone, marginal pilot target association, geometry, and
target-agnostic residual variance are distinct comparators; the conditional
score must add value beyond them, not simply rename the cleanest channel.

## Direction score: expected reduction in parameter uncertainty

At the current acquired calibration state and fixed retained set, use a
regularized linear proxy for each fixed scaled target coordinate `k`. Let
`A_S,k` denote its parameter covariance and `sigma_d,k^2` the pooled/shrunk
residual-noise estimate for direction `d`. Define `M_S` as a fixed,
direction-balanced second moment of selected features from a declared allowed
reference fitting/calibration pool, using fitting-only coordinate rules. For
the original reference this is the common pilot. Choose the integration
distribution before action outcomes and keep it common within the comparison;
it is not the future or held-evaluation feature distribution. A proposed score is:

Both `M_S` and the distribution of `x | d` obey the decision clock. Sequential
use excludes later calibration rows and full-session normalization. A source-
trained prior must have usable feature/target coordinates; transferring neuron
identities, electrode identities or target covariance across animals is not
automatic. Source-trained selection is not evidence of zero-shot decoding.

```text
g_d(S) = mean_k E_acquired[x | d] [
  x' A_S,k M_S A_S,k x / (sigma_d,k^2 + x' A_S,k x)
]
```

The expectation is an empirical or shrinkage distribution of direction `d`
using only already acquired trials or a prospectively declared source-trained
prior, never its next recorded LFP or spike target. In the linear Gaussian proxy the fraction is the reduction in
integrated parameter-prediction variance after one additional feature row;
the proxy integrates over possible rows because the row is not known before
requesting the direction. A full Bayesian covariance is not required by the
existing decoder; this score is an explanatory approximation whose noise and
prior assumptions must be stated. It is **not an exact prediction of the
reduced-rank ridge decoder's held-out loss change**.

Equal current errors can imply different `g_d`: a poorly estimated mapping can
offer reducible uncertainty, whereas persistent trial noise can offer little
learnable benefit. Direction counts alone may already explain the ordering;
this proposal must outperform that simpler account. At every common replay
state, predict all legal direction ranks before branching, then compare with
the existing average realized `V_direction` across prespecified orders. No
single favorable next trial establishes learnability.

## Coupling makes a stronger prediction than two wins

Hold the acquired trial identities, counts, decoder procedure, and evaluation
targets fixed. For a quality-matched swap `S -> S'`, predict both `g_d(S)` and
`g_d(S')` before branch outcomes. The principle predicts the observed change
**or stability** of next-direction ordering. It does not assert that every
swap must change the argmax. A narrow direction pair may be studied only when
its pilot/pre-choice score difference satisfies a prospectively chosen
support rule; no pair can be picked for a favorable observed reversal.

For the optional reference, the inherited 2 × 2 contrasts remain at all nine budgets.
Two positive main effects do not establish coupling. The existing contract
also requires a resolved factorial interaction for its stronger joint claim;
this proposal does not relax that rule. If scores predict electrode value
but the aggregate interaction cancels, a separately supported change in trial
value remains evidence of coupling, not a pass of the reference's stronger
joint-claim decision. Conversely, an aggregate interaction need not support
the proposed conditional-value explanation. If scores predict electrode value
but not a change in trial value, report an electrode-selection explanation.
If both choices help independently, report two methods. If the policy wins
but neither score predicts future value, report an engineering gain only.

## What is still a scientific choice, not frozen protocol

Before activating additional explanatory endpoints, choose the trial-summary
and rank cap, pooled-noise/shrinkage family and allowed development selection,
pilot/pre-choice cross-fitting, rank or paired-value score, minimum
predictable contrast, uncertainty and multiplicity plan, and the rule for
missing branch/swap support. No numerical margin is invented here. The
existing primary policy decision cannot be rescued or relabeled by these
additional results.

Development uses whole-session cross-fitting in eight sessions. The four
internal held sessions test all selected prospectively frozen analyses in one
shared opening; they cannot freshly
confirm an explanation selected after those outcomes. This exposed,
two-animal public corpus cannot establish external animal transfer. Chewie-R
is another implant in Chewie, not a third animal. No compatible independent
third-animal source is currently available. Offline recorded-trial replay
also cannot establish online behavioral response, electrode placement,
biological necessity, clinical performance, or measured hardware savings.
