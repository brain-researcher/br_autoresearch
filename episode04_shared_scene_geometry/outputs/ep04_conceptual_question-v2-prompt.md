# EP04 imagegen v2 — 2026-10-02

Built-in imagegen, no source stimuli or outcomes supplied. Accepted final inspected at 1672 × 941: shared image bus, two independent model-to-fMRI RSA links, symmetric schematic RDMs, one selected VLM and within-category OOD comparisons. A/B/C cards are abstract placeholders, not actual OOD images/classes. Initial draft's serial RSA chain, mixed-category photographs and asymmetric neural RDM were corrected. Earlier figure retained.

## Generation

```text
Use case: scientific-educational
Asset type: publication-style neuroscience question figure, wide 16:9 raster, opaque white background.
Primary request: A restrained readable scientific schematic titled "Does VLM–brain alignment survive an image shift?" Four lettered panels in a 2-by-2 composition. Dark navy sans-serif labels, thin lines, modest pale-blue heading strips, teal model accents and blue neural accents, generous white space. No glossy infographic, huge caveat ribbons, gate/check/lock badges, scores, statistical stars, bar-chart winners or fake findings.
A heading "The same images, three representations". Show three small INVENTED generic picture thumbnails labelled I1, I2, I3. Every thumbnail connects to one shared vertical input bus, and the bus branches to all three destinations: "Native VLM embeddings", "DINOv2 embeddings", and "fMRI patterns". Under the VLM destination only, short text "OpenCLIP · PEcore · SigLIP2". Under fMRI only, five small equal person icons labelled "Same 5 participants". Do not imply different image subsets across branches. No caption text fed to VLM, learned brain mapping or encoding regression.
B heading "Compare image relationships". Three same-sized schematic 3-by-3 RDMs in identical image order I1,I2,I3, labelled "VLM", "DINOv2", "fMRI". Each matrix must be exactly symmetric with uniform dark zero-distance diagonal; off-diagonal colors may differ between matrices. Two neutral comparison connectors lead from VLM RDM to fMRI RDM and from DINOv2 RDM to fMRI RDM. Label both "Spearman RSA". Model geometry is cosine distance; neural geometry is correlation distance. Those definitions may be one small line each, not paragraphs. No identical matrices or suggested winning model.
C heading "Select on regular images". Small collection of generic image tiles → three equal model name labels "OpenCLIP", "PEcore", "SigLIP2" → one neutral model outline with a question mark labelled "One selected VLM". A fixed gray DINOv2 icon sits alongside as comparator throughout, not as another selectable VLM. Caption "Image-disjoint development". No numerical sample count, effect size or chosen VLM.
D heading "Test within OOD categories". This is the scientific focal panel. Show a separate invented image collection with three category groups labelled "Category 1", "Category 2", "Category 3". Each group has three miniature varied image tiles and pairwise comparison lines ONLY inside that group, none across categories. Above the groups write "Same selected VLM · same DINOv2". Below them, two concise lines "Within-category RSA" and "Does the VLM advantage repeat?" No ID-minus-OOD subtraction or retention percentage. Same five participants, new image distribution; no new cohort implied. The category thumbnails are schematic placeholders, not actual dataset stimuli.
Only a small bottom footer "Study design". Low-level controls and selection/audit conditions belong in the caption, not warning panels. Preserve direct native embedding–fMRI geometry comparison and equal-status candidates. No superiority, training-causal, shared-code or mechanistic result is asserted. Make all visible text large and concise; avoid additional invented labels.
```

## Connector and category correction

```text
Use case: precise-object-edit.
Input image: the supplied EP04 figure is the edit target. Correct three precise scientific depiction errors; preserve the entire overall layout, title, typography, panel A shared image bus, panel C model selection, and all unrelated text.
1. Panel B comparisons: DELETE the arrow from VLM to DINOv2. VLM must compare DIRECTLY with fMRI, never with DINOv2. Draw one clear curved connector from the VLM matrix around the TOP of the central DINOv2 matrix to the fMRI matrix, labelled "Spearman RSA". Keep a separate short connector from DINOv2 to fMRI labelled "Spearman RSA". Both connectors terminate at the same fMRI RDM. No serial VLM→DINOv2→fMRI chain.
2. Panel B matrices: make each 3x3 RDM EXACTLY symmetric about its diagonal. All three diagonal cells in each matrix have the same darkest shade. Off-diagonal cell pairs (1,2)/(2,1), (1,3)/(3,1), (2,3)/(3,2) must have exactly matching colors inside each matrix. Use flat cells, no gradients. The three matrices may have different off-diagonal colors and should not look identical. Preserve I1,I2,I3 order. No numeric measurements.
3. Panel D categories: REMOVE all nine photorealistic thumbnail pictures. Instead draw simple abstract image-card placeholders, not source stimuli. Category 1 contains three teal cards marked A1, A2, A3; Category 2 three amber cards marked B1, B2, B3; Category 3 three violet cards marked C1, C2, C3. Within each group use one consistent faint category motif (for example one circle, one triangle, one square respectively), with slight position variations across the three exemplars. Preserve the three comparison lines ONLY within each group and none between groups. This makes repeated exemplars within one placeholder category clear. Keep all panel headings and "Within-category RSA" and "Does the VLM advantage repeat?".
No extra content, invented data, claimed winners, result values or warning banners. Keep generous whitespace and readable labels.
```

## Final neural-RDM correction

```text
Use case: precise-object-edit.
The provided EP04 image is the edit target. Change ONLY the small RIGHTMOST fMRI matrix in Panel B to correct its symmetry, plus its missing first column label. Preserve every other part of the image exactly.
The fMRI matrix has three rows and three columns in I1,I2,I3 order. Keep its three identical darkest diagonal cells. Its upper-triangle colors already define the distances. COPY each upper-triangle cell color to its mirror location:
- row2 column1 must be EXACTLY the same medium blue as row1 column2;
- row3 column1 must be EXACTLY the same pale light blue as row1 column3;
- row3 column2 must be EXACTLY the same blue as row2 column3.
Do not keep the current cyclic color ordering. All six off-diagonal cells must form matching symmetric pairs. Use uniform flat color fills, no gradients. Retain the upper-triangle colors. Add the missing column label "I1" above the first column, aligned with existing I2 and I3. Keep both RSA comparison connectors directly ending at fMRI and keep all other matrices, card placeholders, words, title, layout, dimensions and colors unchanged.
```
