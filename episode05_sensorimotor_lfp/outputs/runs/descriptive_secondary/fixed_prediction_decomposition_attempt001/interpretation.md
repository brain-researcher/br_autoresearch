# What accounts for the small LFP prediction gain?

The fixed current/history blend predicts individual-trial departures from the
direction/time average: about **95% of its small total gain** comes from the
net trial-varying component. Its advantage over history alone combines a
smaller trial-varying prediction penalty with a smaller improvement in the
direction/time mean correction. This is a descriptive explanation of the
existing result, not a new candidate or biological mechanism.

The [complete decomposition](decomposition.json) exactly replayed all selected
W, HW and blend scores on the original sixteen sessions and passed the new
algebraic reconstruction checks. W is current-bin reliability-weighted LMP,
HW adds the original four-bin history, and the blend is their already selected
equal mixture. Both animals and every original trial, direction and time bin
remain included. All quantities below use the original full-session native-XY
denominator, with sessions averaged within animal and then animals equally.

| Fixed prediction | Mean correction gain | Trial covariance benefit | Trial prediction energy, subtracted | Net trial-varying gain | Total delta R-squared |
| --- | ---: | ---: | ---: | ---: | ---: |
| W | +0.00001826 | +0.00338195 | 0.00122536 | +0.00215659 | +0.00217485 |
| HW | +0.00000932 | +0.00541446 | 0.00308252 | +0.00233194 | +0.00234125 |
| Blend | +0.00013622 | +0.00439821 | 0.00185557 | +0.00254263 | +0.00267885 |

For the blend relative to HW, the trial covariance benefit **falls by
0.00101625**, while the subtracted trial prediction energy **falls by
0.00122695**. Their difference improves the net trial-varying gain by
**0.00021069**. Mean correction adds **0.00012691**, giving the known total
improvement of **0.00033760**. The stronger history-only pairing readout and
the better blend R-squared are therefore compatible: the blend gives up some
trial alignment while reducing the squared magnitude of its trial-varying
corrections by more. Reduced energy here means dispersion of fixed prediction
corrections across trials; it does not estimate training-sample variance or
prove removal of physiological noise.

This distinction repeats in the two animal averages. Blend-minus-HW net
trial-varying gains are approximately **+0.00026104 in Mihili** and
**+0.00016035 in Chewie**. Mean correction improves by **+0.00025819** in
Mihili and changes by **-0.00000437** in Chewie. The original blend advantage
over HW remains small and session-sensitive; this decomposition does not
strengthen its independent replication status or certify new controls.

The calculation writes the velocity residual as `e=Y-B` and the neural
correction as `r=P-B`. Within each direction and time bin, the gain splits as:

`n*(2*mean(e)*mean(r)-mean(r)^2) + 2*sum(e_centered*r_centered) - sum(r_centered^2)`.

The first term measures direction/time mean correction; the last two measure
trial covariance benefit and its prediction-energy cost. The exact existing
wrong-trial derangement readout separately multiplies each direction's
covariance term by `n/(n-1)`. Evaluation means appear only in this accounting;
they are never used to fit or select a predictor. These are not additional
R-squared endpoints with newly selected denominators.

The registered primary margin remains **0.01**, so the best **0.002679** result
still fails it. The blend's identity increment remains **0.004559**, below
**0.005**. No new source data, spike-population model, held session, model
selection or fit was used. The 45-comparison stop and exploratory evidence
status remain unchanged. Shared motor-population dynamics, contamination,
causality and online usability remain unresolved.

[Session values](sessions.csv), [animal summaries](animal_summaries.csv),
[fixed session contrasts](session_contrasts.csv), [all fifteen bins](bins.csv)
and [all direction/bin cells and trial counts](direction_bins.csv) preserve
the complete descriptive evidence without selecting favorable intervals.

Coordinator execution: Slurm **46574496**, `sh04-18n32.int`,
2026-10-05 04:49:07.971618–04:49:18.985454 UTC; process elapsed **11.013831 s**,
peak RSS **230,816 KiB**, exit 0. Allocation accounting belongs to the
coordinator; this report does not charge the shared allocation to EP05.
