# Which unresolved question is worth a next comparison?

**UNADOPTED assessment, 2026-10-04.** Evidence ends at complete comparison
37, patience 4. The incumbent is the original constant W/HW blend
(primary 0.0026788544310999856; identity 0.004559394718773168). No
time-varying candidate scores were read. This note authorizes no implementation,
source access, resource request or change to coverage, accounting or stopping.

My preference, if root wants one further explanatory comparison after the
time-varying result is released, is **support robustness of the actual blend
policy**. This is optional scientific work, not a missing minimum-family
requirement or a route to manufacture patience. If the purpose is only a valid
negative closeout, neither repeating these stresses nor adding PLS is required
by the [completed coverage audit](primary_post_intersection_coverage_audit.md).
Both incumbent practical margins already fail. A different incumbent after
the pending release would require reconsideration before adopting this plan;
this note does not silently substitute a different mixing policy.

## Why HW's completed support test does not answer the blend question

The [complete I/R evidence](runs/adaptive_search/primary_intersection_sensitivity_summary_attempt001/compact_evidence.json)
shows that guide-intersection HW exceeds mean count-matched random support by
0.000463359 primary, yet loses 0.000327344 against full HW. Mean I-minus-R
effects survive all animal omissions, but Mihili's primary comparison with
individual draw 4 is negative. The primary loss against full HW also reverses
in Mihili when 20140307 is omitted. These qualifications preclude a universal
guide-support advantage.

Those are four-lag HW results. The blend's improvement can instead depend on
how current-only W and HW errors complement each other. With native residual
errors eW and eHW, a mixture has

`SSE = (1-alpha)^2 SSE_W + alpha^2 SSE_HW + 2 alpha(1-alpha) sum(eW*eHW)`.

HW support results provide neither the reduced W errors nor their cross-term
with reduced HW. Thus an HW advantage associated with recorded guide membership
does not establish that the mixture's complementary errors survive that support
restriction. This is the concrete unresolved distinction.

## One finite support family and its relevant control

If formalized, use only the already frozen all-sixteen-session intersection
and the exact eight ordinary count-matched supports from metadata attempt002.
Do not introduce trial reduction, new sets, complements, ranks or redraws.
Keep all original trials, native XY, baseline, denominator and wrong-trial
evaluation. Fit current-only W on each reduced support using its unchanged
training-only algorithm; reuse that support's completed five-penalty HW
predictions only after exact identity, baseline, score and saved-array replay.
Both components must use the same reduced support. Mixing full-W with
reduced-HW would answer a different question.

For the intersection policy, retain the original five shared lambdas and
five constant alphas, literal endpoint arithmetic, and original
other-fifteen-session equal-animal `(score, lambda, -alpha)` selection. The
control is the mean of eight random-support mixtures evaluated at that same
intersection-selected lambda/alpha in each session. Each random support has
its own W/HW states and error cross-term; average its correct/wrong SSEs,
never predictions. This controls the physical channel count while holding
the selected mixing rule fixed. It does not separately optimize each random
support or equalize covariance, anatomy, signal quality or effective df.

Report direct common-denominator contrasts against the preserved full-support
blend, mean random-support mixture, and reduced HW at the same selected lambda.
The last contrast asks whether mixing still contributes beyond the historical
component on that support. Keep both animal means, all sessions and fixed
omissions, every random-draw exception and all selected lambda/alpha pairs.
These descriptive contrasts add no thresholds or counts. Guide membership
still does not measure or remove spike contamination.

This would be at most two distinct nonpromotable support-stress recipes,
with zero literal-null credit and no incumbent reset. Exact global reuse of
a previously evaluated prediction panel charges zero independently for each
recipe; compare complete selected prediction arrays, including every control
draw, rather than similar scores. A new reduced-W endpoint is not automatically
a duplicate of full W. No endpoint is forced away to earn depth.
A formal proposal and resource-only pilot review would still be required.

## Why genuine PLS remains distinct, but lower priority

The existing [covariance-SVD code](tools/primary_covariance_svd.py) makes one
SVD of `C = Z.T @ U / n`, keeps its nonzero left singular directions (at most
two because U has two coordinates), and applies a ridge head in that fixed
subspace. It never deflates fitting X or Y. A genuinely sequential PLS
regression would construct further scores after training-only deflation;
its mapped-back regression directions need not remain in the initial
`span(C)`. Likewise `(G + lambda I)^(-1) C` for full ridge need not lie in
`span(C)`. The old negative covariance-SVD result therefore does not settle
whether covariance-adjusted supervised compression can help.

That distinction alone is weaker motivation than the blend's observed
complementarity. Full weighted models already accommodate the input
covariance, and no completed evidence identifies a particular component
budget or deflation variant as the unresolved cause of their errors. PLS
would require its own exact finite algorithm and component choices,
training-correspondence controls and capacity limitations; relabeling the old
rank-two basis would add no science. I would defer it rather than open a
second optional family now. Existing depth, conditional patience, alternative
allocation and resource ceilings remain controlling; this assessment creates
no obligation to continue after a valid stop.
