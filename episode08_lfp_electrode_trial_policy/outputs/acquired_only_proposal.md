# Next step: freeze electrodes early and count the initializer

Proposed after the residual-selector report at21:50UTC on2026-10-03.
**Adopted prospectively by the coordinator after the15-evaluation closeout.**
Implementation and both focused checks completed in coordinator job46433944
on2026-10-03; see the actual execution record in `experiment_log.md` and
`runs/development_search/acquired_only_attempt001/acquired_only.json`.
This is one acquired-only policy comparison with a
necessary fixed-random reference, before more elaborate trial allocation.

## Scientific purpose and limitation

Residual marginal selection currently has the strongest aggregate Q, while the
conditional update has not improved on it. Every completed selection comparison
used all96electrodes throughout its fitting pool. Those scores cannot show
whether an electrode set chosen early remains useful after discarded electrodes
become unavailable, or whether selection gains justify the initial observations.
Test that information constraint directly before adding trial-allocation rules.

These are retrospective development trajectories, chosen after observing prior
development results. Previously exposed outcomes are not made unseen by a new
reader. The proposed numerical policies receive only their allowed history,
and the existing evaluation trials never enter them. A final prospective claim
still requires the later frozen protocol and closed held-session opening.

## Two branches and common roles

Keep the current eight development sessions, exactly the existing96fitting
trials (twelve per direction), immutable within-direction queue order, and
forty reserved evaluation trials. Begin with the first four fitting trials per
direction: **32total**, explicitly chosen for this experiment, not a universal
mandatory initializer. Evaluate checkpoints32/48/64/80/96 with balanced queues
(four/six/eight/ten/twelve per direction), and retained8/16/32/all96electrodes.
Every observation is a whole trial, with the same ten-bin means and native
population target. No new trials/windows, held sessions or outcome exclusions.

A. **Initial-only residual marginal selection.** Observe all96LFP electrodes
and target population on the32initializer trials. Fit the already-defined
residual marginal score with initial-only direction means, residual-X SD and
raw-Y SD, then freeze one nested electrode ordering permanently. Afterward,
read only retained electrodes on each acquired trial. Never re-rank from later
trials or read discarded electrodes after initialization.

B. **Fixed random selection.** Use the five already fixed electrode permutations
and nested sets. Select the set before neural observations; read only those
electrodes on every acquired trial, including initialization. Its population
labels and trial queues match A. The5random replicates are one benchmark,
not5methods. All96 is a common identity and may share its fit/predictions.

Both use one global mean-feature reduced-rank decoder, its existing finite
rank/alpha grid, direction-residual transforms, true-Y inner loss and tie rules.
Do not choose a separate decoder or selector by session/animal. The intervention
is when electrode selection occurs and what remains observable afterward.

## Avoid selection leakage during tuning

At checkpoint32, branchA uses the already-qualified nested initial-only
selector/rank procedure: each inner selector sees only that initial fold's
fitting rows; its final ordering refits all32. BranchB uses its fixed random
set and the same four whole-trial initial folds. The final A set then stays
fixed for all later checkpoints.

At later checkpoints the initializer is part of the conditioning history.
Keep all32initial trials in **every** inner fitting set. Partition only the
newly acquired post-initial trials into four validation folds, assigning each
row by `(within_direction_post_initial_index + direction_index) mod4`.
This gives nonempty folds even at48trials (two new observations/direction).
For each fold, fit all direction means/scales, rank projection and decoder from
initializer plus the other three post-initial folds. No validation target
helped choose the now-fixed initial electrode set. Refit on all acquired rows.
Use the same rule in B so tuning opportunity is matched. This deliberately
changes later CV from the old all-prefix folds: do not pretend later scores
must reproduce prior offline curves. No evaluation observations tune anything.

## Trusted reader and actual exposure

The new trajectory reader must verify exact legal trial identities **before**
HDF5 slicing. A may request all96only for its32initializer rows; subsequent
requests include only its frozen selected channel/band indices. B may request
only its fixed set from its first row onward. Read the same native unit targets
only on acquired rows. The policy receives copied observed arrays and metadata
queues; future neural values and evaluator paths stay with the trusted reader.
All trajectory models/checkpoints for a session must be fixed before its40
reserved evaluation rows are read. Report actual union bytes read separately
from each policy's allowed exposure; reuse cannot erase a policy's acquisition
cost. Same-UID callable separation remains weaker than an OS seal.

For retained E and n acquired trials, A uses
`32*96 + (n-32)*E` LFP electrode-trials; B uses `n*E`.
Also report3bands*10bins per electrode-trial, native target-unit observations,
initializer counts, actual numerical/system cost and source bytes. A and B are
matched in retained electrodes and trial count but **not initial LFP exposure**.
Never call their raw Q difference an equal-acquisition-cost win. Do not convert
an electrode into a number of trials or a power claim. Show both resource axes
and the extra initializer cost directly. Any reduced-observation advantage is
bounded by these exact source-view measurements, not old offline source reads.

## Reporting, charge and next trial-value experiment

Report the full20cell surface, all sessions/animals and random replicates;
Q uses each branch's own acquired-training global mean/raw-Y SD denominator,
which is identical between branches at a common checkpoint because their target
trial identities are equal. Retain direction-only and exact wrong-pair pairing
readouts descriptively. New20cell aggregate scores are compared **within this
protocol**, not directly ranked against the prior12cell mean. At32, the existing
matching initial fits can be replayed as an implementation reference; later
CV differs prospectively. All96 can be shared across branches. No new biological
margin or held decision is introduced.

Conservatively charge **one complete acquired-only policy comparison**, with B
as its necessary reference, not additional charges for earlier-method replays,
seeds, checkpoints or exposure aliases. It moved15to16valid evaluations on
successful completion.
The implemented design requests one CPU/2GiB with a15-minute step/840-second
internal bound in the next coordinator-owned substantive allocation. It fits
760actual rank models across eight sessions,15,960ridge systems,76,000nested
rank/alpha scores,40initial selector fits and40distinct direction-only means.
Repeated narrow views produce13,824training read calls plus320evaluation calls;
unique logical source elements remain the original96+40trials/session. Actual
source-read bytes (including repeated reads) and the unique logical union are
reported separately. The prospective runtime estimate was several minutes;
the observed completed attempt took133.452525seconds.
No worker allocation or compute occurs. Existing tests are reused;
any focused check must address actual acquisition-boundary or initial-only-CV
logic. No source copy, generic preflight or synthetic filler is proposed.

If early selected electrodes remain useful, the next distinct question is which
trial to acquire. Compare balanced queues with a pre-choice uncertainty/value
rule under that **same frozen set and initial history**, with finite direction
capacities, no future-feature lookahead and identical total observations. A
current-error allocator should be a competing explanation because high error
may be irreducible noise. At96all twelve trials/direction are exhausted, so
policies see the same final pool; adaptive benefit must appear at intermediate
checkpoints and cannot be inferred from a different final dataset. That next
rule needs its own explicit pre-score proxy, folds, legal actions and exposure
recipe. No new direction-value or coupling acceptance score is activated here.
