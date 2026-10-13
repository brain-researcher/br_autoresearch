# EP11 conceptual figure

Mode: built-in imagegen skill. Asset: `ep11_conceptual_question.png`.
Scientific status: design schematic, not observations or fitted predictions.
No local data or outcome image was supplied.

## Final generation prompt

Use case: scientific-educational
Asset type: EP11 paper-design conceptual figure, landscape.
Primary request: illustrate the scientific question "Do nearby somatosensory neurons use reusable output patterns?" for neuroscientists.
Style: elegant clean scientific infographic, white background, readable dark sans-serif labels, restrained teal and violet, fine-line neuron sketches, generous whitespace. No photorealism.
Composition: three connected panels left to right. Panel 1 is "Same source, overlapping positions": a small schematic cortical patch labelled "SSp-tr", with intermingled neuron somata at similar depths. Several axons route towards the same two generic boxes "Target set A" and "Target set B", with different allocation of drawn branches, no missing target implied. Panel 2 is "Two descriptions": an upper row "Flexible continuum" shows a smooth connected curve of little allocation strips changing from teal-rich to violet-rich, with a visibly connected middle. A lower row "Reusable patterns + gradients" shows two overlapping-position pattern families, each internally graded, with no hard geographic segregation. Both rows carry identical generic target-set labels, and neither row is marked the true answer. Panel 3 is "Freeze, then transfer": a development-brain icon -> a small lock -> a new brain/sample icon and a comparison of predicted allocation distributions, NOT predicted cell labels. Use abstract distribution outlines without numerical axes or real data. Label "No re-clustering". Show three equal neutral conclusion boxes at bottom: "Reusable patterns", "Continuous account adequate", "Unresolved".
Exact title: "Do nearby somatosensory neurons use reusable output patterns?"
Exact footer: "Design schematic — not data"
An additional readable caution: "Brain/sample units are not yet certified independent animals."
Constraints: no invented measurements, sample counts, p-values, significant winner, molecular cell-type identity, synapse symbols, or validated anatomy. Allocation stands for regional axon length, not terminal arbors or functional connectivity. Keep diagrams conceptually illustrative. Target set A and B are placeholders, not selected endpoints. Clearly legible labels. No logos or watermarks.

## Targeted correction used for the saved final image

The first image incorrectly conflated the development data role with biological
development. The saved final image corrects that terminology and separates the
three possible conclusions from the three explanatory panels.

Use case: text-localization.
Edit this existing EP11 scientific schematic only to correct data-splitting terminology. Preserve the title, neuroscience drawings, colors, first and second panels, distributions, and uncertainty caveat.
In panel 3 replace the paragraph "Learn output patterns in one sample, freeze them, and test whether they transfer to a new sample from the same developmental stage." with exactly "Learn allocation patterns in development data, freeze them, and test their predictive distributions in held-out replicate units."
Replace the label "Developmental sample (learn)" with exactly "Development data (learn)".
Replace the label "New sample (test transfer)" with exactly "Held-out unit (test transfer)".
Remove the three separate downward arrows that currently connect panel 1 to "Reusable patterns", panel 2 to "Continuous account adequate", and panel 3 to "Unresolved". Leave the three bottom outcome boxes in place and insert one centered neutral caption directly above them: "Possible outcomes — none is assumed".
Keep the footer "Design schematic — not data" and the caution about uncertified independent animals exactly unchanged.
No other changes. Development means a training data role, NEVER biological development or developmental stage.
