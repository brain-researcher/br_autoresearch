# Fixed M1 publication comparison

Selected October 3, 2026, before inspection of this run's numerical CCA or
decoding results. The coordinator adopted this scoped benchmark from the
published article and pinned figure code. The first session's completion,
trial count, fixture status and runtime were the only run facts disclosed.

## Scope and unchanged execution

Use all sixteen qualified M1 sessions, in `source_adapter.M1_SESSIONS` order:
six Mihili, six Chewie-L, four Chewie-R. Use every successful whole trial in
the frozen `runs/source_support/attempt002/trial_identity.csv` table. Preserve
the nineteen native bins -4:+14 relative to movement onset, square-root
spikes, 50-ms Gaussian smoothing, all-trial rank-ten spike PCA, matched
physical electrodes, scalar-band CCA, fifty independent 90/10 whole-trial
splits per representation, three past bins, correlation-squared X/Y scores,
and the pinned last-coordinate-only cubic mapping. These are reference
semantics, not the later primary's training-only transformations or SSE score.

The analysis revision remains `cbda8e2e6106f5eb5ff98e18a689c595179ac5db`;
the compatible TrialData revision remains
`d5e5eeb1af592cf88c03599df7433878f9d11bbe`. The latter is not authenticated as
the authors' original runtime. Session seed is `20261003 + zero_based_index`
in the fixed sixteen-session order. Original paper random draws are unavailable.
Completed Mihili_20140217 attempt002 is reused, never rerun to improve agreement.

## Exact published summaries

The external targets are [Figure 3 supplement 1 and Figure 4C/D](https://elifesciences.org/articles/73155)
and the [pinned figure implementation](https://github.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/blob/cbda8e2e6106f5eb5ff98e18a689c595179ac5db/Generate_all_figures.m).

| Target | Fixed reconstruction and support check |
| --- | --- |
| Figure 3 supplement 1 | Pool the matched-electrode CCA coefficients over sixteen sessions separately for each of nine bands. Expected **606** electrode/session values per band. Also retain each session's channel-mean profile. |
| Figure 4C | Pool fifty split scores × two coordinates × sixteen sessions, separately for nine LFP bands and the latent representation. Expected **1,600** values per representation. |
| Figure 4D | For each session/band, x is its channel-mean CCA and y is its mean over fifty splits and X/Y. Exclude the latent representation. Expected **144** points. Fit an intercept-bearing ordinary least-squares line; report slope, ordinary R-squared, and the intercept-only-versus-line F-test p-value. |

Both pooled captions identify their horizontal bars as medians. The pinned
figure code instead computes means (`Generate_all_figures.m` lines 207 and
251). Therefore report **both caption medians and code means**, with unambiguous
labels and no outcome-dependent choice between them. Keep the original
distribution values, not just rendered histogram approximations.

The published Figure 4D association is positive, with R-squared printed as
**0.78** and F-test **p < 0.001**. Check the slope sign and that p threshold.
Record whether R-squared is in **[0.775, 0.785)**, the declared ordinary rounding
interval for the printed value. This is published reporting precision, not an
empirical equivalence margin. A failure of that rounding check alone is not
declared material disagreement, because the original fifty split draws are
unarchived. Report the numerical difference in all cases.

Support checks are exact and diagnostic: a count mismatch must be exposed and
explained, never repaired by dropping or duplicating observations. Incomplete
sessions, missing/nonfinite coefficients or scores, wrong band identities,
or a changed source/function recipe cannot be called complete benchmark
reconstruction. Chewie-L/R are displayed separately to match the publication,
but remain one animal. The pooled F-test reproduces a published descriptive
statistic; its 144 points are not 144 independent animals.

## Frequency-profile comparison and decision

The paper describes stronger low- and high-frequency CCA, weaker intermediate
8–50-Hz CCA, and good decoding by LMP and 200–400 Hz relative to intermediate
bands. It does not supply a unique set of per-band/session inequalities,
minimum contrasts, or a numerical definition of “nearly” matching the latent
decoder. **The profile comparison is descriptive.** Present all nine bands
and the latent comparator, pooled summaries, sixteen session profiles and
the two animal summaries. State which parts of the published pattern are
recovered, which differ, and whether variation is concentrated in a session
or implant. Do not invent a 0.3 cutoff, require every pairwise band ranking,
select favorable sessions, or optimize any discrepancy away.

The final record must separately state:

1. Whether complete faithful reference execution was achieved.
2. Whether the scoped M1 frequency/decoding pattern was reproduced, differed,
   or remains unclear, with the complete descriptive evidence.
3. The exact support, association, rounding comparison and numerical differences,
   including uncertainty about their materiality.
4. The actual contract decision: whether the published M1 benchmark result is
   sufficiently established to begin the separately specified primary search,
   or which concrete discrepancy still prevents that decision.

A process exit code cannot make decision 4. A documented material mismatch
blocks a positive adaptive candidate under GOAL.md; it does not authorize
tuning this benchmark. Conversely, reproducing other brain regions, TME
surrogate significance, planning epochs, the whole article, bitwise original
outputs, or a new numerical-equivalence test is **not** an added search gate.
The later primary's own splits, model schedules and scientific margins still
must be specified before its scores.

## Optional, descriptive split-randomness diagnostic

If used, run exactly 10,000 paired-bank bootstrap replicates with NumPy
`Generator(PCG64(20261003))`. Each bank resamples fifty split indices with
replacement independently within every fixed session and LFP band, keeping
each split's X/Y pair together. Recompute the 144 y means and Figure 4D
R-squared for both banks, keeping the deterministic x CCA values fixed.
Use the 2.5% and 97.5% linear-interpolated quantiles of the bank-one minus
bank-two R-squared differences as an approximate two-run split-randomness
interval, shifted by the observed R-squared. Report its overlap with the
published rounding interval and the unshifted difference quantiles.

This is conditional Monte Carlo compatibility of the same fixed recorded
sessions, not a biological confidence interval, proof of equivalence, or a
pass rule. A wide interval does not qualify the benchmark. It does not resolve
runtime/version differences, figure mean/median ambiguity, or unpublished
preprocessing. No model refits, additional source reads or split-seed searches
are authorized by this diagnostic.
