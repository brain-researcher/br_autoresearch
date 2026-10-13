# Lean Gao MOp qualification — 2026-09-26

**Decision:** Gao MOp passed the released brain/specimen-count and outcome-blind
soma common-support gates, but failed the prospectively frozen development
structural/observation gate: 5/200 cells were globally eligible versus the
required 190, and none of 12 per-group minima passed. This run must revise or
stop before candidate search. The 8 final and 56 unassigned units remain
unopened; all 15 EP11-overlap units remain unassigned.

## Official source and access boundary

- Publication: [Gao et al., *Neuron*](https://doi.org/10.1016/j.neuron.2025.10.019)
- Provider portal: [whole-cortex projectome](https://mouse.digital-brain.cn/projectome/whole-cortex)
- Public archive: [Zenodo record 16909126](https://zenodo.org/records/16909126)
- Provider neuron metadata:
  `https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/mouse.neuron.info.json`
- Provider sample metadata:
  `https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/sample.info.json`
- Per-neuron SWC endpoint:
  `https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/neuron_download/<file>`

The first pass downloaded the two public metadata JSON files to transient
scratch. MOp was selected solely by the provider soma acronym beginning with
`MOp`. The retained fields were provider ID, SWC filename, source acronym,
hemisphere, reconstruction type, fMOST filename prefix, sample ID, project,
and transgenic line. The JSON transport also contains an outcome-derived
`class` key; its values were not queried, emitted, retained in an artifact, or
used in selection.

Slurm job `45396282` later requested bytes 0--4095 from each of the 837
selected SWC endpoints and consumed only the first validated canonical soma
root row. It did not retain the remainder of any response or open axon-tree,
target, branch, terminal, bouton, voxel, or final-group outcomes at that stage.
The later frozen development-only execution is recorded below.

## Metadata support

| Check | Observation | Interpretation |
| --- | ---: | --- |
| MOp cells | 837 | Source-label selection only |
| Distinct fMOST filename prefixes | 76 | Conservative released brain/specimen units, not certified animals |
| Exact sample joins | 76/76 | Each prefix joins once to a distinct nonempty provider sample ID |
| `Axon_and_dendrite` | 833 | Provider reconstruction scope; not a per-cell completeness score |
| `Axon_only` | 4 | Sensitivity unavailable under frozen roles: 0 development, 1 final, 3 unassigned; never open for development |
| Hemisphere | 226 left; 611 right | Coarse metadata coverage only |
| Layer | MOp1 27; MOp2/3 322; MOp5 330; MOp6a 151; MOp6b 7 | Provider labels retained as anatomy/design variables |
| Exact soma roots | 837/837 | Canonical root rows only; no axon-tree rows opened |

The sample records expose `fMOST_id` and `sample_id`, but no animal, mouse, or
donor identifier that certifies one distinct animal per released unit. The
76-unit count therefore clears the necessary 20-group arithmetic gate for a
12-development/8-final design only under the contract's conservative
brain/specimen-unit interpretation. Cross-animal replication is not yet
established.

The provider describes the release as complete axon reconstructions, while the
cell metadata distinguish only `Axon_and_dendrite` from `Axon_only`. EP10 froze
its own full-tree structural QC and terminal-bearing target rule before testing
it on complete trees. Unknown or incomplete observation cannot be assigned a
reconstruction-level zero; for an eligible cell, zero means only
`not_observed_in_the_eligible_released_reconstruction`.

## Outcome-blind soma geometry and role freeze

Slurm geometry job `45398661` placed all 837 soma coordinates in the Allen
CCFv3 geometry; all 76 fMOST/sample units remained represented. The primary
screen used the 833 `Axon_and_dendrite` cells, folded hemispheres, measured
tangential separation by 26-neighbor pial-surface geodesic, and used normalized
pia-to-soma depth. It tested 48 prespecified conditions without emitting a role
witness or opening projection outcomes. Forty rows were feasible; 28 of 36
finite-depth rows were feasible and no solver row was indeterminate.

The finite exact-polyline feasibility frontier has two incomparable minimal
conditions: 500 um with depth tolerance 0.20 (825 cells with at least two
candidate neighboring units), and 750 um with tolerance 0.10 (828 cells). EP10
froze 500 um/0.20 by a radius-first lexicographic rule before any outcome was
opened. The LineString3D sensitivity convention gives the same feasibility
result there.

Role-freeze job `45401314` then selected 12 development and 8 final whole
fMOST/sample units using only identity, line, soma geometry, and a fixed-seed
metadata priority. The development role contains 209 primary cells, of which
200 (95.69%) meet the frozen support rule; the final role contains 54 primary
cells and all 54 meet it. Fifty-six units remain unassigned. The
[76-unit frozen-role ledger](gao_mop_frozen_roles.csv) and
[summary](gao_mop_role_freeze_summary.json) record the assignment. This is a
cross-fMOST/sample-unit design, not a claim of 20 verified independent animals.

## Cross-episode overlap

Fifteen of the 76 MOp fMOST units also occur in EP11's outcome-blind Gao
SSp-tr inventory: `221224`, `221227`, `221299`, `221366`, `221428`, `221471`,
`221478`, `221479`, `221481`, `221512`, `221513`, `221726`, `221728`,
`221752`, and `233794`. EP11 has not frozen roles or opened projection
outcomes. All 15 units are hard-fixed `unassigned` in EP10, so this freeze does
not consume an EP11 development or audit option. Their row-level disposition
is recorded before any projection outcome is opened.

The 76-row [source/group support table](gao_source_group_support.csv) records
the conservative unit, sample join, observed layer/hemisphere strata,
reconstruction-scope counts, provisional independent label, and EP11 overlap.
That table records the pre-freeze qualification state; the frozen-role ledger
is authoritative for current EP10 roles.

## Development structural-gate result

The [target vocabulary and observation rule](projection_observation_contract.yaml)
are frozen, including laterality, boundaries, terminal-bearing arbor versus
passing cable, unknown handling, and the 16-coordinate primary exact panel.
The [200-file development manifest](gao_mop_development_manifest.csv) contains
only frozen common-support cells and no unsupported, final, unassigned, or
EP11-shared rows. Outcome-blind Slurm job `45412269` reproduced that manifest
and completed the synthetic exact-normalizer throughput benchmark. Service job
`45438445` then acquired exactly those 200 development trees; constructor job
`45438680` found 5 globally eligible cells and 195 unknown/excluded cells. The
exclusions were 141 tree/parser failures, 22 annotation-zero axons, 6
outside-volume axons, and 26 nongrey terminal leaves. All 12 group minima
failed.

Diagnostic job `45439110` showed that all 141 parser failures were disallowed
transitions out of provider types 10, 11, or 12. Those tags occur internally
in the provider trees, while the frozen contract defines them as terminal-only
markers. This is a resource/contract incompatibility, not a constructor crash.
Even if all parser failures were ignored, the maximum possible eligibility
would be 146/200, still below 190. The frozen action is therefore revise or
stop without replacement, promotion, tag reinterpretation, rule or threshold
relaxation. Candidate search was not started, and final-role morphology and all
final projection outcomes remain unopened.
