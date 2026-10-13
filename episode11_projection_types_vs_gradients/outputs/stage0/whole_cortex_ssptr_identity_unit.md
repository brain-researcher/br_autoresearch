# Gao SSp-tr analysis-unit assessment

**Assessment date:** 2026-09-26  
**Identity-unit result:** pass  
**Distinct-animal certification:** not available in the public release

## What the released records establish

The primary analysis unit is the unique
`(fMOST_id, provider_sample_id)` whole-brain/sample pair.

- Gao et al. Table S1 contains 221 rows and 221 unique `fMOST_id` values.
  Every one of EP11's 36 selected `fMOST_id` values occurs exactly once.
- The provider's live `sample.info.json` contains 382 sample records. Each of
  the 36 selected `fMOST_id` values occurs exactly once, the corresponding 36
  internal `sample_id` values are unique, and the mapping between the two
  identifiers is one-to-one.
- The 214 downloaded SSp-tr reconstructions group without ambiguity under
  those 36 pairs; the 206-cell primary reconstruction scope retains all 36.

This establishes 36 distinct imaged brain/sample units and rules out accidental
duplication of a released fMOST or provider sample identifier within EP11.

## What the released records do not establish

The official Table S1 workbook contains a single sheet; its headers include
`fMOST_id`, sex, line, age, timing, injection, and repository fields, but no
`animal_id`, `mouse_id`, or donor key. Neither it nor `sample.info.json`
contains an animal crosswalk. The released methods and tables do not explicitly state that two
different fMOST/sample pairs can never derive from the same mouse. It is
biologically natural to infer one source mouse per whole-brain specimen, but
that inference is not an explicit provider-certified identity crosswalk.

The 36 units also must not be described as 36 independent experimental
batches. They share injection dates, delivery intervals, laboratories, and
operators. Those variables remain adjustment, balance, and falsification
variables rather than replicate identifiers.

## Operational decision

Model development may use the 36 unique whole-brain/sample pairs as clustered
replicate units. Until the provider confirms one pair per mouse, EP11 will call
transfer to held-out units **cross-brain/sample-unit reproducibility**, not
cross-animal replication. A final cross-animal claim requires either:

1. a provider/author assertion that each pair derives from a distinct mouse;
   or
2. an official animal-to-sample crosswalk showing a one-to-one mapping.

This limitation does not imply that the units are repeated animals; it records
the maximum identity claim directly supported by the public metadata.

## Official sources

- Gao et al. Table S1: <https://ars.els-cdn.com/content/image/1-s2.0-S0896627325008001-mmc2.xlsx>
- Gao et al. article: <https://doi.org/10.1016/j.neuron.2025.10.019>
- Provider sample metadata: <https://mouse.digital-brain.cn/projectome/2/srv/info/mouse/cortex/sample.info.json>
