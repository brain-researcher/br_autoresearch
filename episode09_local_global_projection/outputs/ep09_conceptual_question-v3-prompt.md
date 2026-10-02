# EP09 concept figure v3 — 2026-10-02

Mode: built-in image_gen, with EP08 `ep08_episode_series-v5.png` as a style
reference, followed by targeted drawing corrections. Saved asset:
`ep09_conceptual_question-v3.png`. Prior versions are preserved.

Visual review: panel A paths remain with their own soma; panel B keeps the
recipient and its target vector fixed while changing dendritic residual;
panel C uses same-species group icons and schematic dendritic scales. No
observed target identities, effect sizes or biological cause of equivalence
is depicted. Colors and target slots do not encode data.

## Generation prompt

```text
Use case: scientific-educational.
Asset: EP09 neuroscience concept figure, landscape, three clean panels.
Input image 1 is a STYLE REFERENCE ONLY: match its white background, readable navy headings, restrained teal/orange scientific drawings and generous spacing. Do not copy its content.
Title verbatim: "Does a neuron's own dendrite predict its distant targets?"
Panel A heading "Matched neurons". Draw two cortical neurons with distinct dendritic branching shapes but neighboring somata at the same depth. Give each an axon leading to a different subset of six generic target-region patches. Short label "Same source, layer and class". This illustrates differing exact targets within measured context, not molecular types or synapses.
Panel B heading "Own dendrite or matched donor?". Draw the SAME recipient twice, with IDENTICAL soma, axon and exact target destinations in both copies. Only the dendritic tree differs: left its own teal tree, right an orange donor tree. Short labels "Own" and "Donor". A short curved arrow above the trees indicates substitution of a matched dendritic residual. Under the pair use "Same recipient and target vector". The target vector belongs to the recipient; do not exchange axons, targets or entire neurons. Show a question mark between the alternatives, no winner.
Panel C heading "Where does an advantage repeat?". Show a small dendritic arbor with three concentric spatial bands labeled "Near", "Middle", "Outer", pointing to six distinct target patches labeled T1 through T6. Beneath place two small neutral brain/group icons and the text "Across biological groups". Use an open question mark, NOT a colored empirical heatmap or a result bar. This represents possible scale-by-exact-target structure, not observed effects.
Only small footer "Conceptual design".
Keep every label readable at 1000px display width. Main scientific drawings dominate. No outcome/decision row, shared-class conclusion, biological-cause label for equivalence, validation checklist, locks, red alerts, thick colored frame, giant arrows, fabricated scores or defensive disclaimer paragraphs. No novelty claim; no exact observed neurons.
```

## Correction 1

```text
Use case: precise-object-edit.
Input image 1 is the EDIT TARGET, an EP09 scientific concept figure. Keep the title, three-panel layout, all text, panel B, fonts and colors unchanged.
Make ONLY these scientific drawing corrections:
1. Panel A: the teal soma on the LEFT must connect by a continuous teal axon trunk to targets T1, T2 and T3 only. The orange soma on the RIGHT must connect by a continuous orange trunk to T4, T5 and T6 only. Currently the teal upper-target branches incorrectly originate from the orange soma and a lower orange route is joined to the teal soma. Remove these wrong joins. Keep each axon the same color as its own soma; let routes cross without touching if necessary. No teal-to-orange merged trunk.
2. Panel C: replace the bottom human-brain-shaped icon with a simple mouse-brain outline matching the other mouse-brain outline. Use TWO IDENTICAL mouse-brain silhouettes separated by the existing question mark. These stand for biological groups from the same species, NOT cross-species transfer.
3. Panel C: center the three Near/Middle/Outer dendritic-distance bands on the grey soma at the BASE of the dendritic tree. Use three concentric pale SEMICIRCULAR bands above the soma, expanding upward into the dendritic arbor. Do not center the bands halfway up the tree. Keep Near innermost, Middle intermediate, Outer outermost and preserve the target pathways and labels.
No other edits or extra text.
```

## Correction 2

```text
Use case: precise-object-edit.
Input image 1 is the EDIT TARGET. Preserve the title and panels B and C exactly. Replace ONLY the scientific drawing inside panel A, beneath its existing text.
Keep the panel A heading "Matched neurons" and line "Same source, layer and class".
Use a much simpler NONCROSSING arrangement: TWO neurons side by side, teal on the left and orange on the right, their somata at the SAME vertical height. Each dendritic tree extends upward. Each soma has one axon extending STRAIGHT DOWN into its own half of panel A and then splitting into a little fan of THREE branches. Under the teal neuron place three small generic target patches labeled T1, T2, T3. Under the orange neuron place three small generic target patches labeled T4, T5, T6. Teal axon and branches connect ONLY the teal soma to its own three lower patches. Orange axon and branches connect ONLY the orange soma to its own three lower patches. There are NO horizontal paths between neurons, NO axons crossing the other soma, and NO joins between teal and orange. Keep the trees visibly distinct. Do not draw cortical layers behind this simplified matched pair.
This clean paired miniature arrangement replaces ALL existing neurons and paths in panel A. No other changes.
```
