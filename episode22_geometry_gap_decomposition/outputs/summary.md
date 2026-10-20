# EP22 result summary

## 1. Question

Why can released VLM representations lag DINOv2 in native RSA on LAION-fMRI high-level visual cortex while linear encoding is tied?
Do spectral shape, eigenvector geometry, stimulus selection and development-fitted spectral correction help explain this discrepancy?

## 2. Data and evidence roles

- Five participants: sub-01, sub-03, sub-05, sub-06 and sub-07; these are the development participants.
- Primary sectors: laion-ventral, laion-lateral and laion-dorsal. Retinotopic EVC is descriptive.
- D0–D3 used development images already exposed in EP04 and are exploratory.
- OOD used 371 sealed images in one procedurally sealed opening, governed by a write-once lock established before OOD access.
- Locked confirmatory categories: cropped (60 images), relations (72), selfmade (30) and unusual (64).
- illusion-natural (8 images) and synthetic types are descriptive.
- RSA values are Fisher z of Spearman RSA; gaps and decomposition terms use that scale. D0 reports logEDratio.
- Aggregation weights folds (development) or categories (OOD) equally, then sectors, then participants.
- Let a and b denote native VLM and DINOv2 RSA; c uses VLM eigenvectors with DINOv2 eigenvalues, and d uses DINOv2 eigenvectors with VLM eigenvalues.
- Definitions: gap = a − b; E_val = ½[(a − c) + (d − b)]; E_vec = gap − E_val.

## 3. Development findings (exploratory)

D1 identifies negative spectral contributions across all three VLMs, with differing eigenvector contributions:

| Model | Native gap | E_val | E_vec |
| --- | ---: | ---: | ---: |
| CLIP | -0.035 | -0.045 | 0.011 |
| PEcore | 0.001 | -0.038 | 0.039 |
| SigLIP2 | -0.039 | -0.038 | -0.001 |

- D0 pooled LAION−THINGS logEDratio (n=128): CLIP 0.206, PEcore 0.291, SigLIP2 0.195.
- D2 gaps were negative at R1/R2 and positive at R3: CLIP 0.013, PEcore 0.019, SigLIP2 0.024. DINOv2 RSA increased from 0.129 at R0 to 0.532 at R3.
- D3 own-selection effects: DINOv2 -0.039, CLIP 0.011, PEcore -0.008, SigLIP2 0.004. These remain exploratory selection diagnostics.

## 4. Sealed OOD confirmation

C2 was rewritten post hoc before the lock, following a pre-lock dev-test on pseudo-OOD development categories.
The original within-category swap closure ≥50% prediction became C2a (within-category E_val < 0); C2b all-pairs closure became descriptive because development did not support it.

| Locked prediction or check | Result | Key numbers, Fisher z unless stated |
| --- | --- | --- |
| OOD gate | Passed | DINOv2 aggregate 0.187; null threshold 0.033; 10,000 permutations |
| C1: CLIP native deficit | Indeterminate | Mean gap -0.023; participant gaps have both signs |
| C1: PEcore native deficit, secondary | Indeterminate | Mean gap -0.027; participant gaps have both signs |
| C2a: CLIP spectral penalty | Passed | Mean E_val -0.020; negative in every participant |
| C2a: SigLIP2 spectral penalty | Passed | Mean E_val -0.023; negative in every participant |
| C2b: all-pairs closure | Descriptive; not evaluable | Native gaps are positive: CLIP 0.074, SigLIP2 0.021 |
| PEcore secondary criterion | Failed | Mean E_val -0.017; mean E_vec -0.010 |
| C3: CLIP combined-transform transfer | Passed | Mean transformed-minus-native RSA 0.022 |
| C3: PEcore combined-transform transfer | Passed | Mean transformed-minus-native RSA 0.017 |
| C3: SigLIP2 combined-transform transfer | Passed | Mean transformed-minus-native RSA 0.035 |

The superseded original C2 also failed on OOD: its participant rule held in 0/5 for CLIP and 2/5 for SigLIP2.
DINOv2 also improved under C3, with mean change 0.045; this was not a locked prediction.
[Confirmation figure](figures/ood_confirmation/20261013-opening01/ood_confirmation.png).

## 5. What is and is not established

Within this evaluated set, the revised spectral-penalty predictions passed for CLIP and SigLIP2.
C3 supports transfer of the combined development-fitted transform; all three VLM predictions passed.
The [implementation](code/ood_confirmation.py) projects onto the leading 256 development PCA axes and applies spectral weights using the development-fitted exponent α.
Its comparator is the full native representation, so the reported improvement does not separate PCA truncation from reweighting or isolate the exponent's contribution.
C1 remains indeterminate, and neither the failed PEcore secondary criterion nor unevaluable C2b establishes its intended claim.
DINOv2 also improves under this combined transform; the observed benefit is therefore not specific to VLMs.
These findings establish neither population effects nor causal effects of training objectives, and do not establish a universal VLM native-geometry deficit.

## 6. Post hoc hypotheses

- **Hypothesis 1 — stimulus selection:** CLIP is near DINOv2 in cropped (0.153/0.158), relations (0.163/0.155) and unusual (0.217/0.221), while selfmade differs (0.123/0.213); pairs report CLIP/DINOv2 native RSA. This pattern is compatible with CLIP-space stimulus selection inflating the development deficit.
- **Hypothesis 2 — scale dependence:** OOD E_vec changes from within-category to all-pairs values of -0.003→0.087 for CLIP, -0.029→0.041 for SigLIP2 and -0.010→0.068 for PEcore. E_val remains negative at both reported OOD scales. This motivates a between-category eigenvector advantage, although all-pairs measurements do not isolate between-category structure.

## 7. Limitations

- The same five participants contributed development and OOD data; the seal is procedural, and the revised C2 followed development exposure.
- Released stages differ: DINOv2 is a layer-23 patch-token mean, whereas the VLM representations are projection heads.
- Categories, sectors and folds do not supply additional independent participant replications.
- Descriptive categories and retinotopic EVC cannot strengthen the locked primary confirmation.
- The OOD pool is now consumed and cannot serve as fresh confirmation.

## 8. Implications for a theory-level follow-up episode

- Separate spectral-shape, within-/between-category structure and stimulus-selection hypotheses into explicit competing predictions.
- Compare matched representation stages where feasible, and contrast CLIP-space selection with selection independent of the evaluated model.
- Freeze development fitting, estimands, aggregation, margins and decision rules before accessing a fresh sealed stimulus pool.
- Evaluate participant generalization with new participants; causal claims about training objectives require evidence beyond comparisons of released checkpoints.
