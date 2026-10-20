# EP09 stage A: n_coverage, dendrite boundaries, cortical axis, donor support (2026-10-17)

**October 18 — scoped donor-population amendment proposed, unadopted.** The
[reviewable proposal](companion/donor_population_amendment_proposed.md) defines
primary/benchmark populations and targets, prospective support filtering,
effective animal counts, pseudo-audits and the 1,000/500 µm tradeoff. It binds
the completed stage-A artifacts without changing them. The original companion
draft below remains unadopted; no axon access or execution is activated.

**Status:** stage-A values are frozen in
[stage_a_freeze_manifest.json](stage_a_freeze_manifest.json) (SHA-256
`a12f4bf8…ab0d`). The companion model/donor contract is a draft awaiting the
scientist's decisions. No axon geometry has been read, and stage B has not
started.

Codex (gpt-6-astra, max effort) drafted all four tools and the companion
draft. I reviewed each one and made the corrections noted below. I wrote
`ep09_recipient_support.py` and `ep09_donor_support_census.py` myself, reusing
the reviewed precheck functions.

## Frozen values

| Output | Value | Job | File |
| --- | --- | --- | --- |
| **n_coverage** | **6** | 48311775 | [ncoverage/ncoverage_report.json](ncoverage/ncoverage_report.json) |
| R_cap | 600 µm | 48311776 | [dendrite_boundaries/dendrite_boundaries.json](dendrite_boundaries/dendrite_boundaries.json) |
| Radial shell edges | 50, 100, 150 µm | 48311776 | same |
| Branch-order ranges | 1–3, 4–5, > 5 | 48311776 | same |
| Cortical depth and frames | v2 field; all 1,703 somata | 48312305 | [laplace_v2/](laplace_v2/) |

### n_coverage

The simulation follows contract v1.2: 108 scenarios per n, 10,000 replicates
each, and seed 9009.

| n | Worst error SD | Worst P95 | Pass |
| --- | --- | --- | --- |
| 3 | 0.0668 | 0.126 | no |
| 4 | 0.0587 | 0.111 | no |
| 5 | 0.0517 | 0.102 | no |
| **6** | **0.0474** | **0.0889** | **yes** |
| 14 | 0.0308 | 0.060 | yes |

- **Binding scenario:** prevalence 0.5, no heterogeneity, full target
  dependence, predictions 0.2 vs 0.8. This is the scenario the analytic
  calculation predicted, which gives an SD of 0.0471 at n = 6.
- **Analytic guard:** passes on all 648 checked scenarios (ratios
  0.977–1.023).
- **Count gate:** V_6 = 666.
- **Robustness:** the binding scenario has no pool effects, so the 24-cell
  relabel cannot change the selected n.

### Dendrite boundaries

These come from all 1,114 development dendrites; none was excluded and no
node was unreachable.

- **Raw quantiles:**
  - radial: 59.4, 91.8 and 137.8 µm;
  - branch-order thirds: 3 and 5;
  - cell maximum radius at the 99th percentile: 575.4 µm.
- **Length shares across the branch-order ranges:** 33%, 38% and 29%.

### Cortical axis: v1 defect and v2 correction

- **v1 (job 48311786) is superseded.** Its pial boundary counted only L1
  faces next to background.
  - At the medial wall, the hemispheres' L1 layers touch: 52–63% of the outer
    L1 faces of ACAd, ACAv, PL and ILA.
  - The orbital L1 touches olfactory structures: 98–99% for ORBl and ORBvl.
  - Those surfaces were therefore no-flux. L2/3 somata got implausible depths
    (ACAv 0.72, ILA 0.50, PL 0.43), and the two hemispheres solved as one
    component.
- **v2, reviewer correction (implementation only):**
  - L1's whole outer surface is pial, except faces into another region's
    layer 1 (sheet continuation, e.g. ENTl1) or into fibre tract.
  - The hemispheres are decoupled at the midline.
  - The soma gradient never differences across the midline.
- **Result:**
  - two components converge, each in 601 iterations;
  - median L2/3 depth: ACAv 0.225, PL 0.135, ORBl 0.128. Medial-wall axes
    point medially;
  - lateral areas are unchanged;
  - layer medians: L1 0.03, L2/3 0.14, L4 0.33, L5 0.39, L6a 0.74, L6b 0.97.
- The diagnostic is in [logs](logs/) and in the v2 report's
  `pial_face_categories_v2`.

## Recipient and donor support (outcome-free)

**Recount (job 48311488).** This applies the contract's coordinate relabel to
the 24 catalog-versus-coordinate area disagreements
([relabeled_cells_qc.csv](recipient_support/relabeled_cells_qc.csv)).

- 853 of 1,114 development cells and 406 of 589 audit cells sit in exact
  pools of ≥ 5 cells (≥ 4 non-self donors).
- The animal with the fewest such cells has 6.
- Over donor-supported cells only, V_n is 631, 623, 619 and 616 for n = 3–6,
  and 0 from n = 7.
- The all-cell recount reproduces the precheck exactly.

**Caliper census (job 48314384)**, in
[donor_support_census_variants.json](donor_support_census/donor_support_census_variants.json).
Support requires ≥ 4 caliper donors and a full donor assignment with no cell
assigned to itself.

| Calipers | Supported cells, dev / audit | Dev animals ≥ 4 | Audit animals ≥ 4 | Animals with none | Telencephalic V6 targets meeting the floor |
| --- | --- | --- | --- | --- | --- |
| Exact pool only | 853 / 406 | 27 | 16 | 0 | 127 / 170 |
| + same hemisphere | 834 / 403 | 27 | 16 | 0 | 125 / 170 |
| + depth ≤ 0.10 | 786 / 366 | 26 | 16 | 1 | 120 / 170 |
| + distance ≤ 1,000 µm | 785 / 371 | 25 | 16 | 2 | 123 / 170 |
| + depth + distance ≤ 1,000 µm | 733 / 345 | 24 | 16 | 3 | 119 / 170 |
| Draft: + depth + distance ≤ 500 µm | 610 / 241 | 21 | 13 | 9 | 121 / 170 |

- The floor is ≥ 4 donor-supported A_pre recipients in every animal that has
  donor support. A_pre is an upper bound on observability.
- The 1,000 µm width is the uncapped animal-balanced 90th percentile of the
  distance to the fourth-nearest eligible peer (987 µm, rounded up). The draft
  capped it at 500 µm.
- **Under the draft's every-animal floor, no V6 target qualifies in any
  variant** (job 48312584), because of the animals with no support. With the
  floor applied only within supported animals, every variant keeps ≥ 12
  development and ≥ 8 audit animals.

## Companion model/donor contract: draft v0.1, unadopted

[companion/companion_contract_v0_1_draft.yaml](companion/companion_contract_v0_1_draft.yaml)
(SHA-256 `61b42b27…5ec9`); Codex's notes are in
[codex_notes_v0_1.txt](companion/codex_notes_v0_1.txt). It specifies the
following outcome-free and with numeric rules:

- donor pools, calipers, the support graph and 512 constrained donor plans;
- diagnostics;
- residualization (ridge location and scale on M0);
- M0 encoding, where the impossible 2027 injection dates become missing;
- M1 feature extraction;
- the learner search (40–48 trials);
- the nuisance bank;
- separate-tail inference calibration;
- margin-derivation procedures, with no numbers;
- the 12-block explanatory template;
- hierarchy shuffles;
- three 12/15 pseudo-audits.

It lists decisions D01–D12 for the scientist.

**Review:** D01 and D02 as drafted are infeasible (see the census above). I
recommend:

- C_g is defined for animals with ≥ 4 donor-supported recipients, with the
  policy's ≥ 12 development / ≥ 8 audit animal minimums;
- the per-target floor applies within those animals, and a failing target
  leaves the C target set instead of voiding the primary;
- calipers: same hemisphere, depth ≤ 0.10, and the uncapped 1,000 µm width.

## Next

- Resolve the [scoped population/matching proposal](companion/donor_population_amendment_proposed.md)
  before treating the reviewer's recommendation as adopted. It is not blanket
  approval of D01–D12.
- Complete the remaining companion specifications, donor qualification and
  effective-group inference/precision work at their authorized stages.
- The adopted stage-B technical and later development/audit access boundaries
  remain in force. This proposal activates no axon access.
