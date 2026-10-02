# EP21 feedback-arrow correction — 2026-10-02

Mode: built-in image_gen, precise-object-edit.
Input/edit target: `ep21_neural_code_conversion_design-v2.png`.
Output: `ep21_neural_code_conversion_design-v3.png`.
The earlier generation and edit prompts are retained in
[the v2 prompt record](ep21_neural_code_conversion_design-v2-prompt.md).
No scientific payload or empirical output was used.

## Exact edit prompt

Use case: precise-object-edit. Input image 1 is the edit target: the existing EP21 scientific design figure. Make ONLY one tiny arrow correction in panel B. At the right end of the ORANGE DASHED feedback/update path, just above the left side of the Content loss pill (approximately x=966,y=707 in the 1774x887 input), DELETE the small downward-pointing orange arrowhead. This endpoint must be a plain unheaded tail joined to the Content loss box. Preserve the orange dashed horizontal return path and its label 'Update converter (via content loss)'. The only arrowhead anywhere on this orange dashed feedback path must be the existing UPWARD arrowhead entering the bottom of the orange Converter at approximately x=552,y=657. Thus the feedback path points only from Content loss back into Converter, never into Content loss. Do not modify any navy solid arrows entering Content loss. Keep all other pixels/content/layout as unchanged as possible: all panels A/B/C, every word and number, image thumbnails, brain colors, within-person source-orange brain, common readouts, locks, fixed decoder, whitespace, aspect ratio and typography. No new text, no new design elements, no restyling.

## Visual check

The second-pass reviewer found a residual reverse arrowhead at content loss in
v2. In v3 the dashed orange path has an unheaded tail at content loss and only
one arrowhead into Converter. The scientific content and layout are otherwise
retained; this changes no experimental contract or runtime state.
