# EP21 concept refresh — 2026-10-02

Mode: built-in image_gen; generation using EP08 v5 as a style-only reference.
Target: `ep21_neural_code_conversion_design-v2.png`.
No scientific payload or empirical output was used.

## Prompt

Generated with the built-in tool on 2026-10-02; inspected and corrected twice
before saving the selected v2. No experimental or runtime contract changed.

## Targeted edit 1

Edit the supplied EP21 figure with minimal changes. Keep title, panel layout, readable typography, all of panel B's content-loss training diagram, and disjoint training-image roles unchanged.
Scientific corrections:
1. In panel A remove ONLY "(target space)" from the decoded features label; output is VGG19 features, not target neural space.
2. In panel C the three methods MUST all feed the same set of readouts. Remove the three one-to-one arrows that currently misleadingly assign Feature decoding to Within-person, Reconstruction to Content-loss, and Identification to Brain-loss. Replace the bottom half of C with ONE wide shared box headed "Readouts for all three methods", containing three small icons labelled "Features", "Reconstruction", "Identification". Join ALL THREE method paths with a horizontal common connector feeding that single box. A single question mark can sit below it.
3. The Within-person baseline is WITHIN SOURCE, so change that path's lone brain from teal to orange. Keep source-to-target paths orange then teal for the other two methods.
No extra words, results, scores, badges, warnings or other changes.

## Targeted edit 2

Final precise correction to this EP21 schematic; preserve all text, color, layout, content, and typography except these three small changes.
1. In panel C replace the natural-photo test stack with three neutral outlined gray cards with a large question mark on the front, so no test thumbnail accidentally repeats a training thumbnail.
2. Add a short vertical navy connector from the Content-loss method's brain pair down to the existing shared horizontal method connector. All three methods must visibly connect to the single shared readouts box.
3. The orange dashed training-update path in B must have only one arrowhead, pointing up into the Converter. Remove its arrowhead at the Content loss end; line originates at Content loss and ends at Converter.
No other changes.

## Initial generation prompt

Use case: scientific-educational.
Create a clean landscape scientific concept figure for an adapted reproduction of Wang et al. (2025).
Input image 1 is a STYLE REFERENCE ONLY: white background, restrained navy/teal/orange palette, clear anatomy and large sans-serif labels. Do not copy its experiment.
Title: "Neural-code conversion without shared training images"
Subtitle: "Adapted reproduction of Wang et al. (2025)"
Three panels, generous whitespace and very readable text.
A, upper left: "Train the target decoder". Show a teal stack of distinct natural-image thumbnails labelled "4,712 target images" → target person's brain labelled "Target activity" → block "VGG19 decoder". Show the decoder as learned here, with output a small feature vector; do not show a converter in A.
B, lower left and center, dominant panel: "Train the converter". An orange image stack labelled "4,712 different source images" feeds source person's brain, labelled "Source activity", which feeds an orange block "Converter" → teal vector labelled "Target-space activity" → teal block "Fixed target decoder" → vector labelled "Decoded features". The source image stack ALSO feeds a separate bottom path "VGG19" → "Image features". Decoded features and Image features converge on a small comparison labelled "Content loss", with a dashed learning arrow back ONLY to Converter. Make arrows unambiguous; no target neural response enters content loss.
Between A and B a short label: "Disjoint training image sets".
C, right: "Evaluate on shared test images". Show one shared image stack then three labelled comparison paths "Within-person", "Content-loss", "Brain-loss", ending at a common neutral question mark. Under Brain-loss only a small sublabel "Paired training images". At bottom show three simple output concepts with labels "Feature decoding", "Reconstruction", "Identification". Do not draw any reconstructed example or data scores.
Compact footer: "5 participants · 20 directed pairs · Conceptual design".
Avoid huge disclaimer banners, checklists, locks, fake plots, result claims, novelty claims, 'leakage-free', arbitrary scores, or OOD pictures. Do not show brain-to-brain communication. All images are illustrative natural-image icons, not real study data. Keep every label clear at normal screen size. Caption outside the figure will disclose fixed-provider dependence and contingent inter-site scope.
