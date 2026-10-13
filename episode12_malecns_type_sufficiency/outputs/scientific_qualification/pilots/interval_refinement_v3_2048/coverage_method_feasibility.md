# Coverage-method feasibility for the implied EP12 target

Date: 2026-09-27  
Scope: generated data only; no real or final connectivity access

## Implied target

For synthetic cell `c`, let `G_c` be the three-vector emitted for one fresh
whole provider type by the fixed finite source-fit, CV selection, target score,
and numerical procedure: left-to-right, right-to-left, and their equal
aggregate. The implied target is `theta_c = E[G_c]` over fresh whole types and
all generated source/target randomness at the frozen sample size.

This follows the executable held-out score and the qualification requirement to
use the actual fitter/scorer. It is distinct from a best-in-family
population-KL projection and from a finite generated bank.

## Exactness audit

If 72 outer intervals are independent and compare with one fixed known
`theta_c`, their coverage indicators are iid Bernoulli and the declared
Clopper-Pearson rule applies. At `n=72`, the two-sided 95% interval contains
the nominal value `0.95` only for 64 through 71 covered replicates. Counts at
or below 63 and the count 72 fail that compatibility rule.

The following substitutions do not retain exact inference for `theta_c`:

- one independent plug-in bank mean: exact only for conditional coverage of
  that random bank mean;
- one fresh inner bank per outer replicate: targets a noise-convolved random
  mean;
- sample splitting: removes reuse but targets the split-derived random value;
- cross-fitting or leave-one-out: reference overlap makes indicators dependent;
- QMC error bars, double bootstrap, CLT, or a Bayesian posterior: approximate.

A stochastic confidence enclosure for `theta_c` can support conservative
`certain` and `possible` coverage counts. However, it introduces an oracle
error probability, multiplicity allocation across cells/readouts, tail or
moment assumptions, and a stopping/separation rule. Those are not present in
the frozen specification.

A deterministic certified enclosure could retain the existing target and rule
if refinement continues until every coverage indicator is invariant. The
current continuous/lognormal generators, nonlinear fitter, CV/model selection,
and nonconvex U/M fits have no such certified implementation.

## Measured cost anchor

The V3 pilot used 13,518.779 process CPU seconds for 1,728 eight-type records,
or about 7.823 seconds per record and 0.978 seconds per generated whole-type
row. At that cell mix:

- a shared 576-row bank over 35 cells is about 5.48 core-hours;
- a 10,000-row bank per cell is about 95 core-hours total;
- a 100,000-row bank per cell is about 951 core-hours total.

The worst observed directional row standard deviation was about 0.312. A
normal approximation, which is not a guarantee, suggests roughly 93,000 rows
per cell for a 95% half-width near 0.002: about 888 core-hours over 35 cells
before simultaneous error allocation or the outer qualification. No bank size
or stopping rule is scientifically justified by this approximation alone.

## Boundary

The target is reasonably implied, but its exact finite validation is not. A
successor must either add and authorize controlled oracle uncertainty, produce
a genuinely deterministic certified oracle, explicitly narrow to a finite
conditional bank, or leave adequacy unresolved. No successor launch is valid
until that choice is frozen in a new additive contract/spec version.
