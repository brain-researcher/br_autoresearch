# EP05 conceptual image

Mode: built-in imagegen skill. Final asset: `ep05_question_imagegen.png`.
Scientific status: design schematic, not empirical traces or results.
The prior SVG remains available as `ep05_question.svg`.

## Generation prompt

Use case: scientific-educational.
Asset type: EP05 neuroscience conceptual figure, landscape, publication-style scientific illustration.
Primary request: explain "Does the LFP explain why one reach was different?"
Use a clean white background, dark readable sans-serif labels, restrained teal/orange/violet, ample whitespace. All curves and neural icons are explicitly schematic, never empirical measurements.
Four compact panels left to right:
1. "Same target, different reaches": sketch a monkey hand reaching along two different curved paths to the exact same target. A dashed gray path is the direction-and-elapsed-time baseline; teal and orange paths deviate differently. Caption "Predict the deviation, not just the average".
2. "Hold out one direction": eight radial reach directions, exactly seven gray labelled "Train: 7 directions", the eighth teal labelled "Test: 1 direction". A small training-only baseline box plus stored M1 LFP feature blocks lead to "Baseline + LFP residual prediction". Baseline is fit on seven directions, not mean of test trials. No raw voltage traces or invented electrode measurements.
3. "Does trial identity matter?": two test-trial rows from the same direction. Upper solid arrows pair each fixed LFP prediction with its own reach. Lower crossed arrows swap those SAME fixed predictions across trials. Exact labels "Correct trial pairing", "Wrong trial, same direction", "Keep all fits fixed". No numerical results, winners, or significant marks.
4. "If trial-specific gain survives": a dotted optional chain "Fixed LFP features" -> "Spike-population state" -> "Same velocity correction". Label "Separate explanatory follow-up". Three neutral alternatives below: "Trial-specific information", "Direction-level correction", "Unresolved / no gain". None marked as observed.
At bottom a clearly readable overall footer "Design schematic — not data".
Additional concise caution "Offline processed features; no causal or online-control claim."
Constraints: no fabricated R2/p-values/sample counts, no causal arrow from LFP to behavior, no assumption of fresh test animals or future sessions, no label claiming biological confirmation. The whole-direction split is within session. No monkey identity claims or real anatomy. The spike follow-up cannot rescue a failed kinematic test. No logos or watermarks.

## Targeted scientific correction for the final version

The first raster draft had a baseline-to-feature arrow and crossed lines that
still linked the correct identities. This edit makes the comparison a true
wrong-trial substitution and keeps unresolved evidence distinct from absence.

Use case: precise-object-edit.
Edit the attached EP05 scientific schematic to correct ONLY the following scientific logic errors. Preserve the title, style, typography, first panel, eight-direction diagram, model labels, and the spike follow-up.
Panel 2: REMOVE the horizontal arrow from "Baseline model" to "Stored M1 LFP features". Those are independent inputs, not one producing the other. Keep both arrows/input branches feeding "Baseline + LFP residual prediction".
Panel 3, lower box "Wrong trial, same direction": preserve upper-left "Trial A: LFP prediction A" and lower-left "Trial B: LFP prediction B". On the RIGHT, keep the SAME ordering as the correct-pairing box: upper-right MUST be a TEAL trajectory labelled exactly "Reach A", lower-right MUST be an ORANGE trajectory labelled exactly "Reach B". Draw crossed arrows that connect the upper-left prediction A to the lower-right orange Reach B, and lower-left prediction B to upper-right teal Reach A. This must be a true wrong-identity pairing, A->B and B->A, not crossed arrows that still map A->A and B->B. Do not change the actual fixed LFP prediction traces.
Panel 4: In the gray outcome row "Unresolved / no gain", replace the parenthetical "LFP does not explain the difference" with exactly "No supported trial-specific claim". An unresolved result does not prove absence of neural information.
Everything else unchanged. Keep footer design schematic/no data and offline no-causality statement. No new data or extra labels.
