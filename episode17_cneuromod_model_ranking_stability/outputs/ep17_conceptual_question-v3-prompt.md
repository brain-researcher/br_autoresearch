# EP17 conceptual figure — image-generation record

Updated 2026-10-02 with the built-in image-generation tool; no CLI fallback.
Current image: [ep17_conceptual_question-v3.png](ep17_conceptual_question-v3.png). Earlier project images are retained.

The actual generated image was inspected for scientific operators, labels,
readability and outcome claims, then independently reviewed. Replaced the support-only lead with all three measurement operations; verified equal support counts and genuinely different positive weights on the same voxels, without implying a novel explanatory forecast.
No participant/stimulus payload, observed neural value or result was used.

## Initial generation — exact prompt

```text
Use case: scientific-educational.
Create a new landscape scientific concept figure for EP17, white background, large navy sans-serif type, restrained teal/orange/purple, thin quiet panel dividers, abundant whitespace. Three panels A–C. Explain a measurement question through objects and transformations, not a dense infographic.
Exact title: "When does measurement change model ranking?"
A, "The same named models": three neutral model glyphs labelled "DINO", "SwAV", "Supervised". A small shared set of fictional object drawings points to all three, then arrows lead to a simple fMRI response grid. Under models: "Fixed checkpoints, matched readout". No winner, training-causal claim, task inference or real source images.
B, "Change one measurement operation": three distinct spacious rows.
"Response estimate": three small response traces labelled "B", "C", "D"; below "Refit matched readouts".
"Voxel support": a small grid with two interleaved selected subsets of equal count; below "Same predictions, different voxels".
"Ceiling weights": identical voxel dots with different ring weights; below "Same predictions, different weights".
All are schematic, with no observed numeric values or invented result matrices.
C, "Does the relation repeat?": "Development concepts" points to "Held-out concepts"; below, equally weighted unfilled cards "Stable relation", "Changed relation", "Unresolved". Under them the single statement "Same four participants". Do not show a winning model or a forecast performance chart.
Only footer: "Measurement comparison — not a causal test of training".
Avoid locks, warning labels, score bars, synthetic empirical plots, p-values, model leaderboard styling, 'explained mechanism', and any suggestion that carrying a development weighted mean to audit is a new explanatory prediction. Keep visual text to these labels only.
```

## Targeted correction — exact prompt

```text
Use case: precise-object-edit. Correct ONLY the following items in this scientific schematic. Panel B Voxel support: replace the two dot grids with two matching grids, each exactly THREE ROWS by FOUR COLUMNS. Treat each row as one faintly outlined spatial bin. Left grid: exactly TWO teal dots per row, other two gray. Right grid: exactly TWO orange dots per row, other two gray, in different positions from the left. Thus each support selects exactly six of the same twelve locations, with equal two-per-bin quotas. Add the short line 'Equal size and spatial quotas' above the existing 'Same predictions, different voxels'. Do not imply equal supports: their selected positions differ. Panel A: replace the small label 'fMRI response' above each of the three output grids with 'Predicted fMRI'. Preserve all other text, titles, model glyphs, traces, ceiling-weight illustration, panel C, colors, layout and footer exactly. No new metrics or additional text.
```

## Second targeted correction — exact prompt

```text
Use case: precise-object-edit. Change ONLY the bottom 'Ceiling weights' row in panel B. Replace its three dot grids with TWO aligned horizontal rows of six gray voxel dots at exactly the same six x-positions, so the voxel identities are visibly identical. Label the top row 'Raw' and surround all six dots by equal medium-size thin teal rings. Label the bottom row 'Ceiling-weighted' and surround the same six dots with orange rings of visibly DIFFERENT strictly positive sizes: small, large, medium, small, large, medium. Each ring must surround the corresponding fixed dot. Thus weights genuinely differ rather than only color changing. Replace the caption below with 'Same voxels and predictions; different positive weights'. Preserve the already corrected equal-count voxel-support row, all other images, model names, predicted-fMRI labels, panels A/C, title, layout, colors and footer. No numeric weights, score bars, or other new text.
```

## Display boundary

This is a conceptual illustration, not a simulation result, empirical plot or
new scientific decision. The current GOAL caption supplies the precise scope.
No analysis, model policy, numerical criterion, resource allocation or data
access changed.
