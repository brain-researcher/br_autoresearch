# EP13 concept figure v2 — 2026-10-02

Mode: built-in image_gen with EP08 `ep08_episode_series-v5.png` as a style
reference, followed by targeted glyph corrections. Saved asset:
`ep13_conceptual_question-v2.png`. Prior figure is preserved.

Visual review: formulas are Y = A X P^T and tau = s^T X c; panel C routes each
latent matrix to the same observation and to its own scalar. Equality is an
open question. Internal grid-count correction was not reliable, so the final
edit removes subdivisions: matrix/vector rectangles are dimension-agnostic
symbols, not specified numerical examples. The classical row-space criterion
and zero exception are stated in the reader captions, not proof boxes.

## Generation prompt

```text
Use case: scientific-educational.
Asset: EP13 mathematical-neuroscience concept illustration, landscape, three sparse panels.
Input image1 is a STYLE REFERENCE only: white background, navy readable headings, teal/orange accents, light panels, clear scientific objects. No monkey or neurons from the reference.
Title verbatim: "Can BOLD determine one target without recovering everything?"
Small subtitle "Classical linear estimability".
Panel A heading "Observe a compressed matrix". Show one large teal rectangular grid labelled X, passing through simple small transformation arrows labelled A and P^T into a smaller blue rectangular grid labelled Y. Display ONE exact formula beneath: "Y = A X P^T". Matrix colors are abstract symbols, not measured BOLD. Do not add a brain or empirical time series.
Panel B heading "Ask for one scalar". Show a teal grid X with a short orange vector alongside its ROWS labelled s and a short orange vector below its COLUMNS labelled c. Both are weights, not a selected single element. Thin arrows from these weights and the matrix converge to one orange scalar circle labelled τ. Beneath use ONE exact formula: "τ = s^T X c". Use lowercase s, lowercase c, uppercase X; superscript T denotes transpose. Short text "One target, not all of X".
Panel C heading "Same observation, same target?". Show TWO different abstract latent matrices, labelled X₁ and X₂, side by side. Both have clean arrows converging to ONE small blue matrix labelled "Same Y". Separately, below X₁ put an orange scalar symbol τ₁, and below X₂ an orange scalar symbol τ₂, each clearly connected to its OWN matrix only by a vertical orange arrow. Between the two scalar symbols put the exact expression "τ₁ = τ₂ ?" as the open scientific question. Arrange routes without crossing the blue observation and orange target arrows. This asks whether every observationally indistinguishable state has the same target; it does not assert that equality or inequality always holds.
Footer only "Exact-model illustration".
Mathematical accuracy and readable formulas are critical. Fewer than55 words beyond title. No long theorem box, decision tree, proof certificates, checkmarks, crosses, limits paragraph, empirical claims, result bars or disclaimer banner. Row-space criterion belongs in the accompanying caption, not in this artwork.
```

## Correction 1

```text
Use case: precise-object-edit.
Input image1 is the EDIT TARGET, EP13 classical linear estimability illustration.
Correct ONLY the grid dimensions. Preserve all text, equations, arrow routing, positions, colors, overall layout and style exactly.
Use a CONSISTENT SIX-BY-SIX latent matrix in every panel: panel A X, panel B X, panel C X₁, and panel C X₂ must EACH show exactly6 rows and6 columns (36 cells). This is the same latent shape throughout.
In panel B the vertical orange vector s must show exactly6 cells, matching the6 rows of X. The horizontal orange vector c must show exactly6 cells, matching the6 columns of X. Ensure both vectors clearly align with their matching grid dimension.
Use a CONSISTENT THREE-BY-THREE observation matrix in every panel: panel A Y and panel C Same Y must EACH show exactly3 rows and3 columns (9 cells).
Do not alter any formula, especially "Y = A X P^T" and "τ = s^T X c". Do not alter the two distinct routes from each latent matrix in panel C: grey to Same Y, orange to its own scalar. Do not add any labels, dimensions, results or text.
```

## Correction 2

```text
Use case: precise-object-edit.
Input image1 is the EDIT TARGET. Preserve every word, formula, label, arrow and all panel positions exactly.
Remove ALL decorative internal grid lines from ALL matrix and vector glyphs. Replace them with simple continuous color fields:
- Every X, X₁ and X₂ glyph becomes a plain teal rectangular matrix symbol with subtle continuous shading, NO cell boundaries, NO rows/columns drawn, NO countable squares.
- Every Y glyph becomes a plain smaller blue rectangular matrix symbol with subtle continuous shading, NO internal lines.
- The s vector becomes a plain solid orange vertical bar. The c vector becomes a plain solid orange horizontal bar. NO subdivisions or cells.
These are dimension-agnostic symbolic matrix/vector glyphs. Keep their outer outlines, labels, arrows and scalar circles unchanged. The exact equations carry the mathematics.
Do not draw any internal horizontal or vertical lines inside any of these colored rectangles. Do not add dimensions, ellipses, caveats or any text. Keep Y = A X P^T and τ = s^T X c exactly unchanged and legible.
```
