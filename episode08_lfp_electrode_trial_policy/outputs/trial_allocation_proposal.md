# Trial allocation: reducible direction-mean uncertainty versus total error

Adopted by the coordinator on October3,2026 after the completed acquired-only
comparison and before any allocation outcomes. **Completed in job46433944.8
at22:33:32UTC; see the actual experiment log and trial_allocation_attempt001.** The
globally fixed branch is the existing random sets, selected before trial1,
with five existing repetitions for each scarce electrode count. Its explicit
lower observation cost and simpler conditioning history motivate this choice;
it does not assert random-set superiority. Never choose separately by session
or animal. The recommendation and development-exposure limit are recorded in
`experiment_log.md`. Held access and the named coupling endpoint remain closed.

## Why this comparison and what the proxies mean

A noisy direction can have high prediction error while another observation
barely changes what is estimable. First test that distinction with transparent
finite-sample direction-mean proxies, rather than calling a model's current
error expected information gain. The final predictor remains the same LFP
population decoder. The allocation utilities below concern the estimable
**direction mean**, not uncertainty in the full LFP-to-population mapping.
A success supports this bounded allocation principle; it does not validate a
conditional sensor–trial coupling mechanism or a general information estimator.

## Common observations and legal actions

Use the same eight development sessions, exact96fitting-trial pool, immutable
within-direction queues and forty evaluation trials. Initialize with32balanced
trials (four per direction), using the fixed-random retained electrodes
8/16/32/all96 from the first observation, and never open other electrodes. Every policy
has the identical initial observations and fixed set within each comparison.
A future action selects one direction d whose acquired count n_d<12. The trusted
reader reveals only the next unobserved trial in that direction's existing
queue, with only retained LFP channels and its native population target. No
future feature, target, error or neural-derived quality flag can inform choice.
Update the utility only after the chosen observation is delivered.

Record checkpoint models at32/48/64/80/96observations. Keep each acquired trial
as one row. The three policies share initial32fits; at96all twelve rows per
direction are necessarily acquired, so their final source set and fixed-rule
fit are identical. Share that final fit or verify identity, rather than count
it as three independent models. The informative allocation differences are
at intermediate checkpoints. A final-pool difference indicates implementation
failure, not a scientific effect.

## Explicit pre-choice utilities

Freeze each native target column's population SD from the common initializer;
exactly zero becomes1, with no unit removal. Write Z_i for the acquired raw
population-mean vector divided by this fixed SD. At every choice, calculate
for each direction with acquired count n_d:

`v_d = mean_u[ sum_(i in d)(Z_iu - mean_d Z_u)^2 / (n_d-1) ]`.

Let `v_pool = sum_d[(n_d-1)*v_d] / sum_d(n_d-1)`. Apply the same prespecified
three-degree-of-freedom shrinkage to every direction:

`v_tilde_d = [(n_d-1)*v_d + 3*v_pool] / [(n_d-1)+3]`.

Three pooled degrees of freedom give equal local/pooled weight at the common
four-trial start, and local evidence gains weight with observations. This is
one fixed stabilization rule, not a shrinkage grid or an outcome-selected
parameter. It uses only acquired targets, no evaluation variance. It imposes
no assertion that real errors are Gaussian or stationary.

Compare:

- **Balanced reference:** choose a legal direction with minimum n_d.
- **Mean-value allocation:** maximize `v_tilde_d / [n_d*(n_d+1)]`.
  Under an independent stationary direction-mean model this is the plug-in
  expected reduction in squared estimation error from one more observation.
- **Mean-error allocation:** maximize `v_tilde_d * (1+1/n_d)`.
  Under the same model this estimates total future direction-mean prediction
  error, including irreducible variation. It is the matched high-error
  competitor, not a claimed estimate of the full decoder's current error.

Legal directions alone are compared. Exact utility ties prefer the smaller
acquired count, then smaller frozen direction index; balance uses direction
index after the count tie. Zero variances therefore revert deterministically
to balancing without a new fallback or source inspection. Predictions of
actual usefulness are empirical: the variance model may be wrong, nonstationary
or dominated by poorly estimated means. No future realized benefit is used to
choose actions or parameters. The two adaptive utilities are the scientific
contrast, not near-duplicate tuning values of one method.

## Respect the adaptive observation history during fitting

Ordinary later CV can be misleading if an observation held out for validation
already influenced which later observations entered the fitting set. For this
first adaptive comparison, **freeze rank and alpha at the32-trial initializer**,
using the previously qualified ordinary initial folds for fixed random sets.
Each electrode set has one
initial rank/alpha shared by all three allocation policies. Do not tune again
on adaptively selected data. The full-rank/rank grid, exact ties and mean-loss
alpha convention remain unchanged.

Resolve a selected `full` rank to the initializer's fixed structural maximum
`min(3E, U, 32−8)`; preserve selected integers1/2/4/8 literally. Store that
integer and freeze it for every later refit, so `full` cannot silently increase
capacity as trials accumulate. Keep the qualified initial coefficients exactly.
This is an algebraic capacity bound, not a numerical singular-value threshold
or a claim that the fitted coefficient matrix attains the bound. Later refits
apply the same regularized response-metric projection with this fixed rank.

At each checkpoint refit only direction means, residual scales, the fixed-rank
response projection and coefficients on that policy's observed history, with
penalty n*alpha and the initial rank. Keep native targets and fitting-only
raw-Y SD exactly as in the rank decoder. Training rows are stored in canonical
direction/queue order so policies reaching the same set yield the same fit.
This removes later tuning leakage and keeps opportunity identical; it does
not make adaptively collected fitting rows iid or turn development into an
unbiased audit. A later prequential tuning design would be a separate proposal.

## Common evaluation and honest exposure

Since policies can acquire different direction mixtures, their training means
and SDs differ. Using those differing quantities to normalize evaluation would
mix prediction changes with changes of denominator/target weights. Freeze the
**evaluation** raw-target SD and global-mean null from the common32-trial
initializer for all policies and checkpoints:

`Q_common = 1 - sum[((Y_eval-prediction)/SD_init)^2] /
                    sum[((Y_eval-mean_init)/SD_init)^2]`.

This is a prospective normalization choice for this trial-allocation comparison,
not a retroactive change to prior results. Keep the native population target,
forty evaluator rows, exact within-direction wrong-pair diagnostic, and equal
cell/session/animal weighting. New Q values cannot directly replace prior
12cell/20cell rankings with different denominators. Also report raw prediction
SSE in the same fixed coordinates so the effect is transparent. Do not normalize
by evaluation-derived or residual variance. Outer evaluation remains closed to
all policies until every checkpoint trajectory/model for the session is fixed.

At a common E,n all three policies use the same actual source budget:
report n*E LFP electrode-trials from trial1. Targets use n*U*10native values,
LFP uses3bands*10bins per electrode-trial. Count actual unique/read-sum bytes,
initializer cost, computation, direction counts, and per-choice utilities.
No trial-choice policy may inspect a direction it did not choose. Previous
full-development source exposure remains in the history; this is retrospective
algorithm evaluation with a declared information boundary, not newly blinded
or prospective collection. Same-UID callable isolation is not an OS seal.

## Reporting and scope

Return all20cell scores and every session/animal, balanced-policy contrasts and
the intermediate48/64/80profile. Initial/final equalities are shown, not hidden.
Keep both adaptive policies regardless of result. Use the existing development
progress resolution only within this common-score protocol; introduce no new
biological margin, p-value, stopping threshold or coupling acceptance rule.
The full20cell aggregate and intermediate-only descriptive summaries have
separate labels. Queue order is the existing single order: repeated-order,
reverse-animal and session-omission checks remain later required work.

Conservatively charge the **two new allocation policies** once each on complete
panels; balance is their necessary reference, not a third new method count.
Resource cells, checkpoints, random-set repetitions and utility aliases are not
additional evaluations. The adopted compute envelope is1CPU/2GiB/10minutes,
540seconds internal deadline, threads1. One trusted source handle per session
serves only the chosen next-trial/channel slices. There are128initializer CV
fits across all sessions (2560inner ridge systems and12800candidate scores),
1408actual refits, and1920model references after sharing initial/final fits.
The source-read estimate is28,672fitting-trial reads plus320evaluation rows;
unique logical source support remains the original96+40trials per session.
These are prospective counts and resource estimates, not execution evidence.
Further mapping-
uncertainty, coverage, acquired-only swaps and coupling analyses remain distinct
questions. This small prospective test is not a fabricated route to the minimum
40evaluations or an exhaustive allocator menu.
