# Memory

EP21's initial Doerig draft was superseded on 2026-09-28 before empirical
access or execution. The active source is Wang et al., *Inter-individual and
inter-site neural code conversion without shared stimuli*, DOI
`10.1038/s43588-025-00826-5`.

Durable scientific decisions:

- The primary LAION analysis uses all 20 ordered source-to-target pairs.
- For each pair, the target VGG19 decoder is trained only on the target's 4,712
  regular subject-unique images, while the content-loss converter is trained
  only on the source's 4,712 regular subject-unique images. Exact and
  near-duplicate identity between those training universes must be audited
  under an outcome-blind rule; exact overlap must be zero.
- The 1,121 regular shared images are evaluation-only for the no-shared path.
  A brain-loss converter necessarily uses paired shared responses and is
  isolated to an authenticated training partition, with evaluation on a
  disjoint shared test partition.
- Findings A, C, and E are quantitative feature-decoding or identification
  claims. Reconstruction panels B and D are qualitative displays unless an
  outcome-independent quantitative rating rule is separately locked.
- “Comparable” requires a prespecified non-inferiority or equivalence margin;
  overlapping confidence intervals or a nonsignificant difference is not
  evidence of comparability.
- A LAION-to/from-NSD and THINGS module contains 70 directed pairs if all
  proposed 5, 4, and 3 participants qualify. It is independently provisioned
  and may be `not_evaluable` without blocking the within-LAION branch.
- EP04 and EP21 share regular LAION evidence and are correlated. Live EP04
  outputs and legacy Scratch are forbidden inputs.
- The complete 371-image OOD payload is closed. The proposed artificial-image
  analogue is `not_evaluable` under the current allocation.

The V1.0.0 tag, its commit, and paper-linked Zenodo record are pinned as source
identities, but archive contents, external datasets, DUA and risk status,
image roles, VGG19, AlexNet, converter discrepancies, reconstruction,
noise-ceiling, margin, resampling, and multiplicity details remain unresolved.
`PROTOCOL_TABLE.yaml` must resolve them outcome-blind or lock the affected
branch `not_evaluable` before any neural score.
