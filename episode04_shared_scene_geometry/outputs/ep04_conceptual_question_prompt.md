# EP04 conceptual-figure prompt

Mode: built-in imagegen; new image; opaque white background. No source image,
brain response, stimulus payload, or outcome was supplied to the generator.

The exact generation prompt follows.

```text
Use case: scientific-educational
Asset type: EP04 scientific concept figure for a neuroscience research design, not a results figure.
Primary request: Create a clear publication-style conceptual schematic answering "Which image geometry aligns with fMRI?" Show that the SAME IMAGE IDENTITIES are used to derive frozen model image-embedding geometry and measured fMRI response-pattern geometry, then compared directly by representational similarity analysis (RSA). This is VLM-brain alignment, NOT reproduction of a prior analysis and NOT a voxel-prediction diagram.
Style/medium: High-resolution raster scientific illustration with a clean vector-like finish, white background, crisp dark navy sans-serif text, ample whitespace, restrained teal for VLM branches, blue for the brain branch, slate for DINOv2, and amber for an image-distribution boundary. Wide landscape canvas. No branding or decorative textures.
Composition: Four labeled panels in a balanced two-by-two grid. Main geometry comparison occupies the upper row; fixed-panel scientific contrast and honest image-shift scope occupy the lower row.
Text (verbatim) for title: "Which image geometry aligns with fMRI?"

Panel A heading: "A  SAME IMAGES, DIFFERENT REPRESENTATIONS"
Draw four invented generic thumbnail images labeled I1, I2, I3, I4. Reuse their identities and order consistently in every branch. Branch those thumbnails to: a simple model icon labeled "Frozen VLM image embeddings"; a small schematic MRI/brain icon labeled "fMRI response patterns"; and a neutral comparator icon labeled "DINOv2 image embeddings". Show rows of embedding vectors and rows of ROI response patterns as abstract colored strips, not numerical measurements. Under the VLM branch write "OpenCLIP · PEcore · SigLIP2". Under the brain branch write "5 participants". Do not draw captions or text prompts being fed into a model, learned model-to-brain mappings, training, fine-tuning, or predictions of voxel activity. Release-supplied image embeddings are the primary model input.

Panel B heading: "B  COMPARE SAME-IMAGE GEOMETRY"
Draw a VLM model representational dissimilarity matrix (RDM), a DINOv2 model RDM, and an fMRI response RDM, each symmetric with zero-distance diagonals and rows/columns I1-I4. The invented off-diagonal colors should not be identical across matrices and should contain no numbers. Label the model matrices "L2 + cosine distance" and the brain matrix "Correlation distance". Draw comparison links from EACH model RDM to the SAME brain RDM, with the label "Spearman RSA". Compare pairwise image relationships, not axes or coordinates of neural/model spaces. Write a concise line "Image-disjoint evaluation; no neural fit to model geometry". Keep the matrix ordering explicitly identical. Do not imply that all embeddings have the same dimension or equal similarity values.

Panel C heading: "C  FIXED-PANEL SCIENTIFIC CONTRAST"
Show the three VLM names as equal-status rows, each compared with one fixed DINOv2 comparator. Do not highlight or rank a winner. Put the exact formula "Δz = zρ(VLM, brain) − zρ(DINOv2, brain)" in a readable central box, where zρ denotes Fisher-transformed RSA. Below it put "Required low-level geometry control" and "Ventral · lateral · dorsal visual cortex". Show five equal-weight participant icons and the line "5 participant effects, equal weight". Include the caution "Unmatched checkpoints: not a causal language effect". No observed scores, error bars, significance stars, or plotted empirical advantage. No matched-architecture or matched-training-data claim.

Panel D heading: "D  DOES ALIGNMENT SURVIVE AN IMAGE SHIFT?"
Draw a regular-image collection separated from a visibly different new-image collection by an amber boundary. Label the development side "4,712 subject-unique images per participant" and "Image-disjoint folds; selection-exposed". Label the audit side "371 shared OOD images" and "One locked model; same 5 participants". Add "Within-category RSA; only if previously sealed" under the audit side. Use a small pale-gray crossed-out tag "1,121 exposed regular-shared images: excluded". The boundary changes images, NOT people; do not depict a held-out participant, new participant cohort, or independent replication. No result arrow meaning the audit has passed. Conclude with the question "Robust alignment, no gain, or unresolved?" rather than choosing an outcome.

Footer text (verbatim): "Scientific schematic · no observed data · similarity is not mechanism"
Scientific constraints: Every thumbnail, heat-map entry, icon and colored strip is illustrative, not source data. Never claim VLM superiority, neural specificity, a shared code, a Platonic representation, a causal benefit of language or multimodal training, participant-population generalization, or independent replication. Represent only the current fixed-panel native-geometry design. The five people are biological units; images and RDM edges are not extra participants. Keep the key labels large and legible. Render the requested text exactly, with no added numbers or scientific claims.
```

## Targeted connector edit

The initial image was inspected before this built-in edit. Its Panel A
connectors ambiguously suggested that different representations received
different image subsets. The exact edit prompt follows.

```text
Use case: precise-object-edit
Asset type: EP04 scientific concept figure.
Input image: the provided image is the edit target.
Primary request: Make ONE targeted correction ONLY in Panel A: the image-thumbnail connectors currently appear to partition I1/I2 versus I3/I4. Replace all connectors between the four thumbnail images and the three representation boxes with one clearly continuous shared vertical bus spanning ALL FOUR thumbnail rows I1, I2, I3, I4. Each of all four thumbnails connects to that SAME bus. From that same bus, exactly three distinct branch arrows lead to "Frozen VLM image embeddings", "fMRI response patterns", and "DINOv2 image embeddings". Make it unmistakable that ALL FOUR identical image identities enter EVERY branch, not different subsets of images for different models.
Constraints/invariants: Change only the connector lines and arrow positions in Panel A. Preserve all thumbnails, boxes, embedding strips, all text, the title, panel layout, palette, all other panels B/C/D, RDM matrices, comparison formula, numerical planning counts, unmatched-checkpoint caveat, and the no-observed-data footer exactly. Do not introduce source images, outcomes, statistics, new participants, caption inputs, or a trained mapping. Keep the shared input connector neutral dark gray, with teal/blue/slate branch arrows matching the existing representation boxes. No other changes.
```
