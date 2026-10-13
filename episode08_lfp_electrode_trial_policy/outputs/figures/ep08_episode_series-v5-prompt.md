# EP08 episode-series figure — generation prompt

Built-in imagegen, with EP05 and EP12 supplied as visual references, followed
by focused panel-B corrections to candidate identity and information-region labels.

Reference images:

- `episode05_sensorimotor_lfp/outputs/ep05_question_imagegen.png`
- `episode12_malecns_type_sufficiency/outputs/ep12_conceptual_main_figure-v2.png`

```text
Use case: scientific-educational
Asset type: EP08 scientific concept figure, a coherent sibling to the TWO PROVIDED EPISODE FIGURES.
Input images: Image 1 (EP05) is a STYLE AND TASK-ILLUSTRATION reference, not an edit target. Image 2 (EP12) is a STYLE AND PANEL-COMPOSITION reference, not an edit target. Make a NEW EP08 figure. Borrow their navy typography, teal/orange scientific color coding, concrete biological drawings, crisp linework, readable pale panel headers and scientific-question storytelling. Do not reproduce their text, study claims, footers or experimental protocols.
Primary request: A polished landscape 2-by-2 scientific figure with the exact overall question "What makes an electrode or a trial useful?" Use EP12's balanced four-panel rhythm and EP05's concrete monkey-reaching scene. The scientific images must do most of the explaining. White background; quiet pale-blue panel headings and very thin gray panel edges, not oversized decorative cards. Restrained navy, teal, orange and gray. Short labels only.
A, heading "Recording during reaching". Show a small elegant lateral-view macaque illustration reaching toward one of eight center-out targets. Beside it show a flat physical electrode-array schematic connected to three small LFP traces and a compact neural-population trace/raster. Essential labels "Electrodes", "LFP", "Population activity". This is an experimental context, not a surgical illustration: no invasive procedure or fabricated anatomical localization.
B, heading "Which electrode adds information?" Use two compact side-by-side target-linked information diagrams grounded in small electrode arrays. The SAME orange candidate electrode e is added to two equal-size teal retained sets, S1 and S2. Represent the target-linked information with simple translucent overlapping regions: the candidate mostly overlaps S1, but adds an additional extent beside S2. Labels "Same candidate", "Overlap", "Complement". Show one shared small "Population information" label linking the diagrams. This is contextual target-predictive value, not clean-looking traces or arbitrary signal variance. Do not imply one electrode maps to exactly one neural component.
C, heading "Which trial teaches us more?" Tie two small recording-state illustrations to the same eight-direction reaching motif. Left: several plausible mapping curves fit the same few schematic calibration points, making the mapping uncertain. Right: a stable mapping with equally few schematic points showing trial variability around it. Equal visual size, no numeric axes, no measured confidence intervals. Short labels "Mapping uncertainty" and "Trial variability". Each has a small reach-direction motif underneath; emphasize the same candidate direction in both. Show no before/after improvement, winner or guarantee of learning.
D, heading "Do electrode and trial choices interact?" Two equal-size electrode sets S and S' differ by a single orange electrode while shared retained electrodes stay teal. Beneath them show the same small strip of calibration-trial symbols, labelled once "Same calibration trials". Each set is linked to an eight-direction motif: one has a teal candidate direction, the other has neutral directions and a small orange question mark. This visually asks whether changing electrodes changes the next useful direction; do not show a measured reversal or a successful policy.
Constraints: Match the references' scientific objects and panel language, not a software flowchart. No giant arrows, locks, gears, document stacks, status pills, budget grids, freeze/holdout instructions, compulsory pilot, numeric results, success/failure routes, fake statistical evidence, long paragraphs, defensive disclaimers or oversized footer. Keep typography and object scale consistent with both reference images. This is a conceptual comparison, not data. Convey that through the scientific illustration, without printing a disclaimer in the artwork.
```

## Panel-B correction prompt

```text
Edit ONLY panel B of this figure. Preserve all other panels, overall title, fonts, colors, framing and whitespace exactly.
1. The orange candidate electrode e must occupy the SAME grid position in both 4-by-4 arrays: row 2, column 3 (counting from the upper left). In the right array, make its current row-4-column-3 orange dot gray, replace its row-2-column-3 teal dot with orange, and make row 3 column 2 teal, so each array still has exactly four teal retained electrodes plus one orange candidate. Leave the left array's dot colors unchanged.
2. In each small information-region diagram, label the orange region "e", NOT "S1 + e" or "S2 + e"; keep the teal regions labelled S1 and S2. Both orange e regions should have the same elliptical shape, size and orientation. In the left diagram the teal S1 region should strongly overlap that orange e region, nearly covering it. In the right diagram the teal S2 region should occupy a different leftward extent so a substantial part of e is outside S2. This clarifies redundancy versus complementary population information. Retain the labels "Overlap", "Complement" and "Population information".
Do not modify anything outside panel B, add any text, or alter the same-size electrode swap in panel D.
```

## Final local correction prompt

```text
Make exactly two local drawing corrections in PANEL B, LEFT HALF ("Add e to S1"), and leave every other pixel, panel, word and right-hand drawing unchanged.
1. In the left 4-by-4 electrode grid in panel B, change the dot at ROW 3, COLUMN 1 from teal to gray. The left grid must have four teal retained electrodes, one orange candidate at row 2 column 3, and eleven gray electrodes. Do not change the right grid.
2. In the left population-information diagram only, redraw the blue region labelled S1 so it contains essentially the entire orange ellipse labelled e. Make S1 a slightly larger ellipse with the same tilt and center as the orange e ellipse. Keep the orange e shape and its position unchanged. Both remain lightly translucent, with blue visible around the orange outline. Keep S1 and e labels readable and retain the arrow and word "Overlap". This left example must clearly show near-complete redundant overlap, unlike the existing right "Complement" example.
Do not change panel A, C or D, the title, any text, the right half of panel B, colors elsewhere, dimensions or style.
```
