# Acquired-row direction-grouping control for the mapping heuristic

Prepared prospectively October3,2026; **adopted after the complete mapping
result, implemented and not yet executed.** Root selected the three Monte Carlo
seed labels0/1/2 and full comparison scope before that result, then adopted
this exact control after mapping completed at23:27:17UTC. The current completed
development count is20. The original heuristic lost to balance in both animals,
all eight sessions and all four order averages; the control asks whether its
real direction grouping contributes to that loss, without a rescue claim.
No endpoint, held-session role or optional named coupling analysis changes.

## Question and precise intervention

Does the acquired feature geometry's association with **the actual requested
direction** help the `mapping_design_value` policy, beyond its global geometry,
actual count constraints and acquisition history? Compare the completed true-
grouping mapping policy with one new control,
`mapping_design_grouping_shuffle3_v1`, using three fixed randomized histories.
The intervention changes only which acquired rows contribute to each
direction's utility mean. Decoder labels and targets remain correct.

This is a policy ablation. It does not test the optional named next-direction
benefit or sensor–trial coupling endpoint, and it does not create a permutation
significance test. Its scope remains the existing acquired-LFP mapping recipe
in [mapping_variance_proposal.md](mapping_variance_proposal.md).

## Unchanged experiment and decoder

Use all eight development sessions, the exact96fitting/40evaluation trial
roles,32balanced initializer (four trials per direction), fixed random
electrodes8/16/32/all96, all five scarce-set repetitions and the single all96
set. This gives16fixed electrode sets per session. Retain all four declared
order conditions: `original`, `0`, `1`, `2`; their first four trials per
direction stay unchanged. Use native3bands×10bin means, the same population
target and all32/48/64/80/96checkpoints. No order, seed, electrode set, session,
animal or checkpoint may be selected by the observed score.

Use the same initializer-selected numeric rank and alpha, including the
resolved full-rank cap. Freeze them thereafter. At each checkpoint refit the
same reduced-rank direction-residual decoder on that control history's actual
acquired features, **true directions and true population targets**. Its normal
acquired-history direction means, residual scales, response projection and
coefficients update exactly as in the original mapping comparison; no new
tuning opportunity, decoder capacity, time window or target permutation is
introduced. Evaluation features/targets and the initializer target-SD/mean-null
normalization remain unchanged.

## Count-preserving random utility groups at each decision

For each control history, retain the original mapping rule's true-direction
initializer feature means, residual mean/scales, fixed proxy coordinates
`z_i`, initializer factor`Z0`, reference moment`M=Z0'Z0/32`, and
`eta=32*alpha_init`. Later true acquired rows update
`A_n=eta*I+sum_i(z_i*z_i')` as before. Do not recompute these objects using
pseudo-labels. For every acquired row compute the unchanged nonnegative score

```text
g_i = ||Z0 A_n^{-1} z_i||² / [32 * (1 + z_i' A_n^{-1} z_i)].
```

At total acquired count`n`, let`n_d` be the **actual** acquired count for true
direction`d` in0,…,7. Sort only the currently acquired original trial IDs by
the Unicode lexicographic order of their exact`str(original_trial_id)` tokens.
These tokens use the existing original identifiers; no numeric reordering,
acquisition-order position, new row ID or future row enters the sort. Original
tokens must remain one-to-one with the preserved trial identities.

Make the ordered label list`[0]*n_0 + [1]*n_1 + ... + [7]*n_7`. Initialize a
fresh Python`random.Random()` with`seed(literal, version=2)`, where the exact
literal is

```text
ep08-mapping-grouping-control-v1|20261003|<session_filename>|<order_id>|<E>|<subset_seed>|<mc_seed>|<n>
```

`session_filename` is the frozen source session basename, including its
existing extension. `order_id` is exactly`original`, `0`, `1` or`2`.
`E` is unpadded decimal8,16,32or96. For a scarce set,`subset_seed` is its
existing frozen subset-seed identifier as an unpadded decimal integer; for
the all96 set use the literal`all`. `mc_seed` is unpadded decimal0,1or2, and
`n` is the current total actual acquired count in unpadded decimal form.
The electrode count and subset seed are separate keys, so sets cannot share
a key accidentally. Fresh per-state initialization makes the result independent
of prior RNG call counts or incidental acquisition-row ordering.

Apply one ordinary uniform`shuffle` to the label list and pair its entries
with the sorted acquired IDs. This gives pseudo-labels`tilde_d_i` with exactly
the actual direction-count multiset. Chance unchanged labels or an unchanged
whole grouping are valid draws; there is no derangement condition, rejection,
reseed, favorable-seed choice or replacement. Duplicate labels yield a uniform
count-preserving assignment because every distinct assignment has the same
number of underlying label-position permutations.

The control utility is

```text
u_tilde_d = (1/n_d) * sum_{i: tilde_d_i=d} g_i.
```

Keep all eight groups in the shuffle, including directions already exhausted.
Then apply the existing **actual** legal-action filter`n_d<12`. Choose the
legal direction with maximal`u_tilde_d`; exact ties prefer smaller actual
`n_d`, then smaller frozen direction index. Do not use pseudo-counts for queue
availability. Read the chosen true direction's next row only after the choice,
record its actual identity/features/target, update true history and`A`, and
continue. No future LFP, target, quality flag or realized next-trial benefit is
available when labels are shuffled or utilities are evaluated.

The original grouping gives`u_d=mean_{i:true_d_i=d}(g_i)`. At a **fixed
acquired history**, scrambling preserves all acquired feature rows, every
actual count,`M`,`eta`,`A`, every`g_i`, and

```text
sum_d n_d*u_tilde_d = sum_i g_i = sum_d n_d*u_d.
```

It changes the association between those row scores and the next direction
name. This is not a feature-coordinate shuffle. Globally permuting/rotating
feature coordinates together with`M` and`A` leaves the trace and rank-one
score invariant, so it would not provide this control. Replacing every
direction utility with the pooled mean would instead reduce exactly to the
existing balanced-count tie-break and add no new comparison.

## Information and cost fairness

Each seed/order control runs its **own** acquired history. After its first
different action, its future acquired rows and hence`A` can differ from the
true-grouping policy and other seeds. The same-history matrix invariance
above must never be described as equal`A` across diverged policies. The
comparison measures the entire resulting allocation policy, not only an
isolated utility calculation on the true policy's history.

No control can access an acquired row from another policy merely because it
exists in a cache. A trusted implementation may reuse an identical read or fit
only when that row is independently legal and acquired in the consuming
history. Count each history's initializer and subsequent acquisition exposure
in full. At checkpoint`(n,E)`, each history uses`n*E`LFP electrode-trials and
`n*U*10`population-target values, exactly as the original policy. Reusing the
initializer/final computation does not make those observations free. Charge
additional proxy work, shuffling, source I/O and decoder refits separately;
three randomized histories are not a free comparison to one deterministic
history.

All histories share the same initializer and the same full96-ID endpoint.
Share their identical initializer CV/fit and final fit across all four orders
and three seeds. Keep every intermediate acquired history/refit separate.
After **all new fits** finish, open only the completed matching original
mapping and prior trial-allocation/arrival-order reports for trusted replay
and reporting. Verify exact trial roles, initial numeric rank/alpha, channels,
subset seeds, queues/orders and target normalization before reference reuse;
require the existing1e−10 initial/final Q-score replay tolerance, scale-aware
SSE/normalizer checks and exact canonical96-ID endpoint. Completed prior
reports contain scores, not original prediction arrays; score agreement does
not establish old prediction-array identity. Within the new run, require exact
canonical final X/Y/direction arrays before sharing its final fit. Preserve
the original packets and all partial
new attempts. Incomplete cells, seeds or orders stop the comparison rather
than being silently omitted or replaced.

## Reporting, interpretation and budget

For each session/set/order/checkpoint, average the three seed **SSE scores**
and corresponding wrong-pair SSEs before computing the control score with the
unchanged denominator; equivalently average the three increments. Never score
an average prediction, which would create an unrequested ensemble. Retain all
seed-level scores, actual utility-group assignments and acquired histories;
report seed spread descriptively. Give orders equal weight and retain the
existing equal-cell, equal-session and equal-animal aggregation. Show the
complete20-cell surface, intermediate profiles, every session/animal/order,
fixed-prediction influence and true-grouping-minus-control contrasts. Also
retain the existing balanced/target-only competitors as context, with their
already charged costs.

Conditional on an acquired history, each shuffled-group utility has expectation
equal to the pooled mean of`g`. Its variance depends on`n_d`: smaller groups
can produce more variable means and win the argmax through variability. Thus
the ablation changes directional association and introduces controlled policy
randomization; it does not isolate a causal information effect from every
exploration/count interaction. Three draws describe this fixed Monte Carlo
control, not an exact null distribution, significance threshold, additional
biological observations or proof of exchangeability of true directions.

A reproducible true-grouping advantage supports the usefulness of the real
direction association for this bounded heuristic. It does not prove calibrated
predictive variance, causal uncertainty reduction, repeatable branch-trial
benefit or sensor–trial coupling. Similar or better control performance would
weaken the need for that association in this policy; it would not prove absence
of direction-dependent neural information. All results remain selection-exposed
development evidence in the same two animals. No new practical margin,
promotion rule, named explanatory score or held-session decision is activated.

One complete three-seed comparison counts as **one** alternative-explanation
evaluation, not three methods or twelve order/seed methods. Mapping has now
reached20; completion of this adopted control would reach21. The40minimum,
112ceiling,20-evaluation patience, diverse-family coverage,
post-coverage40%control requirement, adaptive/challenger obligations and shared
CPU/wall/storage ceilings remain unchanged. Reaching20 evaluations alone does
not certify every family has been covered.

With shared initializer/final fits, the full control entails128initializer CVs
(2560inner systems),4864actual refits,102,400fitting-trial reads,320evaluation
reads and98,304pre-choice proxy updates/shuffles. These are prospective counts
for8sessions×16sets×4orders×3seeds, with64post-initial choices and three
intermediate refits per trajectory. They do not erase per-history initializer
exposure or prior reference costs. The complete mapping panel measured271.200377s
and273,288KiB peak RSS. Root selected1CPU/2GiB/20minutes,1140s internal/1200s
external, with one numeric thread and no retry. The new count ratios are about
2.7×refits and2.8×fitting reads, plus3×proxy updates and full group logging;
the selected bound allows overhead rather than claiming measured control cost.

If adopted, focused checks should establish count/matrix/weighted-utility
invariance at one history, RNG independence from acquisition-row order and
prior calls, legal-direction filtering after grouping, untouched true decoder
labels/targets, whole-history information isolation and mean-of-SSE aggregation.
The implementation is `tools/mapping_grouping_control.py` plus
`tools/development_mapping_grouping.py`; six scientific fixtures are in
`tools/test_mapping_grouping_control.py`. Full group mappings are streamed to
eight session JSONL records with per-choice links from the compact run report.
Tests run only in root's verified Slurm allocation. No worker ran tests, read
new source values, computed scores or submitted a job. Prior scientific code
is unchanged, and launch awaits root's review/checks.
