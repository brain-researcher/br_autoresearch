# Proposed model-pair registry and exposure qualification

**Prospective feature definition adopted by root at22:51UTC, 2026-10-04,
after independent review; source feature execution is not yet selected.**
This record fixes the feature definition for the three already
nominated pairs and the required random-network falsifier. It uses the episode
contracts, the existing checkpoint-acquisition record, and official model
documentation. No checkpoint tensors, stimulus pixels, neural/private arrays,
features or model scores were opened to prepare it. It neither activates a
feature worker nor declares checkpoint exposure clean.

The operative [GOAL](../GOAL.md), [DATASETS](../DATASETS.md) and
[SEARCH_POLICY](../SEARCH_POLICY.yaml) remain unchanged. In particular, the
October 2 amendment permits a comparison of fixed representations under an
isolated measurement operation; it does not license a causal training-objective
claim. Three required pairs, five initial edges, the 32–72-trial search,
matched readout opportunities and the one-shot audit all remain required.

## Fixed assets and pair signs

Reuse the three assets and compatibility checks already recorded in the
[October 3 acquisition entry](experiment_log.md#2026-10-03---model_pair_registry---checkpoint-acquisition-attempt-001).
That job used Torch 2.4.1+cu121 and Torchvision 0.19.1+cu121; strict backbone
loading passed for DINO and SwAV. Supervised loading passed with the documented
53 zero-valued legacy BatchNorm counters supplied by backward compatibility.
Those checks did not run feature inference or establish image exposure.

| Registry ID | Fixed existing artifact | Bytes | Representation |
| --- | --- | ---: | --- |
| DINO_R50 | `outputs/cache/model_checkpoints/dino_resnet50_pretrain.pth` | 94,345,885 | Official original DINO ResNet-50 backbone |
| SWAV_R50_800 | `outputs/cache/model_checkpoints/swav_800ep_pretrain.pth.tar` | 113,703,565 | Official standard-width SwAV-800 ResNet-50 backbone |
| SUP_R50_V1 | `outputs/cache/model_checkpoints/resnet50-0676ba61.pth` | 102,530,333 | Torchvision `ResNet50_Weights.IMAGENET1K_V1` backbone |
| RANDOM_R50_S17 | Not yet instantiated; deterministic recipe below | No artifact yet | Same untrained architecture, one fixed initialization |

| Pair | Fixed signed advantage A minus B | Role |
| --- | --- | --- |
| P1 | DINO_R50 − SUP_R50_V1 | Lead nominated promotion pair |
| P2 | SWAV_R50_800 − SUP_R50_V1 | Second nominated promotion pair |
| P3 | DINO_R50 − SWAV_R50_800 | Third nominated promotion pair |
| F1 | SUP_R50_V1 − RANDOM_R50_S17 | Mandatory trained-versus-random falsifier; never promotion-eligible |

No optional fourth promotion pair is proposed. P1/P2/P3 share checkpoints and
are dependent contrasts. Their qualification remains conditional on the
requirements below; listing them does not count three successful comparisons.
Do not substitute a newer DINO family, wider SwAV, supervised V2 or moving
`DEFAULT`, add a seed search, or replace a failed pair after model scores.

## What the official sources establish

The original [DINO release](https://github.com/facebookresearch/dino) links the
selected ResNet-50 backbone separately from its full student/teacher training
checkpoint. Its [ResNet-50 training arguments](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt)
specify 800 epochs, LARS, six local crops, global/local crop scales
`[0.14,1]`/`[0.05,0.14]`, and teacher momentum 0.996. Its published example
100-epoch command is not the selected checkpoint's recipe. The documented
lineage is ImageNet training with DINO self-distillation; the released
[backbone loader](https://raw.githubusercontent.com/facebookresearch/dino/main/hubconf.py)
uses a Torchvision ResNet-50 with its category head removed. The repository
declares Apache-2.0 licensing.

The [SwAV model table](https://github.com/facebookresearch/swav) identifies the
selected standard-width release as 800 epochs, batch size 4096 and
`2×224 + 6×96` multi-crop training. The associated
[training script](https://raw.githubusercontent.com/facebookresearch/swav/main/scripts/swav_800ep_pretrain.sh)
names ImageNet training, 128-dimensional projection and 3,000 prototypes.
These training heads are excluded from EP17. SwAV learns consistency of
cluster assignments across views. Its repository's
[license](https://raw.githubusercontent.com/facebookresearch/swav/main/LICENSE)
is **CC BY-NC 4.0**, not MIT. Record that license and its source when retaining
or sharing an artifact; no redistribution is proposed here.

[Torchvision 0.19's named V1 record](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html)
identifies ImageNet-1K category supervision and 25,557,032 parameters including
the classifier. It describes a simple training recipe; the exact historic
sample order and all training-run details are not established by the reviewed
documentation. Its implementation is the stride-in-3×3 ResNet variant commonly
called V1.5. Record [BSD-3-Clause](https://raw.githubusercontent.com/pytorch/vision/v0.19.1/LICENSE)
as the Torchvision **code** license. The official
[model documentation](https://docs.pytorch.org/vision/0.19/models.html#general-information-on-pre-trained-weights)
separately notes that pretrained weights may carry training-data-derived
terms. Do not relabel the ImageNet data or every checkpoint right as BSD.
The registry records this distinction; it does not certify unrestricted
redistribution or replace the existing research-use terms.

All four proposed feature backbones use the same standard-width architecture.
The official [SwAV hub loader](https://raw.githubusercontent.com/facebookresearch/swav/main/hubconf.py)
also instantiates Torchvision's ResNet-50 for this release; compatibility is
not inferred only from matching tensor names. Its randomly initialized
category head is excluded along with all training projection/prototype heads.
The expected retained parameter count is **23,508,032**, derived by removing
the `2048×1000 + 1000 = 2,049,000` classifier parameters from the documented
full model. This is an architectural expectation, not a new tensor inventory.
BatchNorm running statistics are buffers, not trainable parameters. SwAV's
training heads and DINO's full teacher/student training systems are not part
of this feature count. Reuse the completed loading checks; a future extractor
must actually match the expected retained structure, rather than treating
file size or an approximate “23M” label as the count.

Nominal ImageNet-1K lineage is useful matching, not proof that training images,
sample order, augmentations, teacher/prototype mechanisms, optimizer states,
precision or recipe were identical. All such differences stay fixed across
each neural measurement edge and remain disclosed confounds for training
attribution.

## Proposed common preprocessing

Use one fixed preprocessing path for every trained and random backbone:

1. The authorized builder decodes the exact event-matched stimulus into a PIL
   RGB image. Preserve stored pixel orientation; do not apply an automatic
   EXIF rotation, recoloring, content-dependent crop or replacement image.
   The permitted image adapter must resolve any exceptional format or missing
   mapping before a batch is eligible; this proposal does not extract pixels.
2. Resize the shorter edge to 256 with aspect ratio preserved, PIL bilinear
   interpolation and `antialias=True`; centrally crop 224×224 using
   Torchvision 0.19.1's `CenterCrop(224)` convention.
3. Convert to float32 channel-first RGB in `[0,1]`, then normalize by means
   `(0.485,0.456,0.406)` and standard deviations `(0.229,0.224,0.225)`.
   The input shape is `N×3×224×224`. No image augmentation, flip ensemble,
   test-time crop averaging or per-image standardization is proposed.

This follows the supervised V1 spatial/normalization recipe documented in
[Torchvision 0.19](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html);
[Resize](https://docs.pytorch.org/vision/0.19/generated/torchvision.transforms.Resize.html)
documents the explicit antialias option and PIL behavior. Before any feature
cache, root observed and selected **Pillow12.2.0** in the existing
`$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python` environment
(Python3.11.15,Torch2.4.1+cu121,Torchvision0.19.1+cu121) on October4.
Use `env -u PYTHONPATH PYTHONNOUSERSITE=1` to preserve that environment binding.

This is a **prospective common input choice**, not a claim that the providers'
original evaluation recipes were identical. Official
[DINO linear evaluation](https://raw.githubusercontent.com/facebookresearch/dino/main/eval_linear.py)
uses bicubic resizing. Official
[SwAV linear evaluation](https://raw.githubusercontent.com/facebookresearch/swav/main/eval_linear.py)
uses red-channel standard deviation **0.228**. The proposed common recipe uses
bilinear and **0.229** for all models. It therefore defines the named
representations under a common input transformation and may differ from
provider benchmark performance. Do not silently run both recipes and choose
the favorable one; a provider-specific sensitivity would require a separate
prospective specification and count under the original search policy.

## Proposed eligible layers and feature shapes

Run the existing backbones in evaluation/inference mode with fixed learned
weights and BatchNorm state. Do not adapt BatchNorm to EP17 images, fine-tune,
train a projection, retain classification logits or normalize pooled feature
vectors to unit length. The implementation must make numerical precision
explicit before its pilot; the proposed representation uses float32 inference
and stored features, with no mixed-precision or quantized substitute. On CUDA,
disable both matrix-multiplication and cuDNN TF32
(`torch.backends.cuda.matmul.allow_tf32=False` and
`torch.backends.cudnn.allow_tf32=False`), use the highest float32 matmul
precision setting, and leave autocast off. Float32 tensor storage alone is
not the precision policy. Freeze these settings before caching any features;
there is no hardware/precision search for a favorable representation.

Take the output after the final residual block/ReLU of each named stage,
then take an unweighted arithmetic mean over its two spatial axes:

| Layer ID; shallower-first tie order | Spatial output for 224×224 input | Stored feature vector |
| --- | --- | --- |
| L1: `layer1` | 256×56×56 | 256 |
| L2: `layer2` | 512×28×28 | 512 |
| L3: `layer3` | 1024×14×14 | 1024 |
| L4: `layer4` | 2048×7×7 | 2048 |

These shapes follow the
[fixed Torchvision 0.19.1 ResNet structure](https://raw.githubusercontent.com/pytorch/vision/v0.19.1/torchvision/models/resnet.py).
Every pair receives exactly these four feature opportunities. Do not also
count `avgpool` as a fifth layer: it duplicates L4 global averaging. Do not
concatenate layers, flatten spatial maps, choose an intermediate block from
neural scores or choose an ROI-specific layer menu. Pooling deliberately
discards within-map position; any result applies to these pooled
representations, not every representation the backbone could supply.

Feature centering and PCA remain learned inside development-training
concepts only, as required by policy. This proposal **does not choose the
still-unfrozen PCA-rank/ridge grids, effective-rank threshold, CKA rank or
inner-fold construction**. Those belong in the common linking specification
before candidate-discriminating scores. All models must receive the same
rank grid, penalty grid and selection budget; rank feasibility must use one
prospectively fixed rule. The existing tie order remains shallower layer,
smaller PCA rank, larger ridge penalty. Beta edges retain one setting per
model shared across B/C/D; support and ceiling edges reuse fixed predictions.

## Proposed deterministic random falsifier

F1 compares SUP_R50_V1 with **one** untrained standard-width ResNet-50.
Propose CPU initialization using Torch 2.4.1/Torchvision 0.19.1, a fresh
CPU RNG scope set to **seed 17**, and `resnet50(weights=None,
zero_init_residual=False)` with otherwise default arguments. Construct the
normal 1000-class model once, then remove `fc`; changing construction order
would change RNG consumption and is not the same initialization. Persist
that initialized backbone once in the episode's ignored model cache before
extracting features; there is no seed selection or reroll on low neural fit.

The registered library initialization uses Kaiming-normal convolutions and
unit/zero BatchNorm affine parameters; retain fresh running mean zero,
variance one and counters zero, then use evaluation mode without updates.
This follows the
[pinned constructor](https://raw.githubusercontent.com/pytorch/vision/v0.19.1/torchvision/models/resnet.py).
The same four layers, input transform, folds and linking-model selection
opportunities apply. A trained advantage is a diagnostic, not a new causal
estimate of objective, corpus or architecture. A random advantage or tie is
reported and investigated under the original policy; it cannot be promoted,
be hidden by picking another seed, or be presumed impossible. No expected
numerical margin is invented here.

RANDOM_R50_S17 has no learned pretraining-image exposure by construction.
That fact does not make the paired supervised model exposure-clean, and the
architecture/initialization design itself is not independent of prior vision
research.

## Exposure status and claim-specific qualification

| Model | Documented training lineage | Exact THINGS/CNeuroMod-image exposure now | Permitted registry statement |
| --- | --- | --- | --- |
| DINO_R50 | Official DINO/ImageNet training release | **Unknown**; no exhaustive image or near-duplicate comparison performed | Fixed pretrained representation; no clean-exposure claim |
| SWAV_R50_800 | Official SwAV/ImageNet training release | **Unknown**; no exhaustive image or near-duplicate comparison performed | Fixed pretrained representation; no clean-exposure claim |
| SUP_R50_V1 | Named ImageNet-1K supervised weights | **Unknown**; no exhaustive image or near-duplicate comparison performed | Fixed pretrained representation; no clean-exposure claim |
| RANDOM_R50_S17 | No pretraining; proposed deterministic constructor | No training-image exposure by construction, not yet instantiated | Falsifier only |

Official training lineage, release date, semantic class overlap, filenames and
the absence of a THINGS warning cannot establish pixel-level nonoverlap.
No ImageNet inventory, target archive or near-duplicate detector was run in
this work. Do not convert “unknown” to “no overlap found” or invent an exact
zero-exposure count.

The contract makes two different rules. **Known exact or near-duplicate
exposure makes the affected checkpoint control-only. Unknown exposure cannot
support a clean-exposure claim.** Unknown is not itself the known-overlap
exclusion. With explicit unknown labels and all other prerequisites met,
root may qualify the nominated pairs for the amended measurement-conditional
question; this document does not assert that qualification has occurred.
The interpretation would concern new neural concept roles in the same four
people, not necessarily images or concepts unseen during model pretraining.

If affirmative overlap evidence arises before scoring, retain its actual
scope and source and mark every pair involving that checkpoint control-only.
Do not drop inconvenient target images, repair the 720-concept universe or
swap a new checkpoint to preserve three eligible pairs without explicit
design resolution. If fewer than three promotion pairs qualify, the required
panel remains incomplete. If such evidence arises after scoring, preserve
the scores and exposure chronology and restrict the claim; do not relabel
past exposure as clean. Any future empirical near-duplicate procedure needs
its own fixed matching rule and authorized image scope before results.

## What this proposal enables and leaves open

After root adoption, the next useful work is a small implementation of these
named backbones, layer outputs and one deterministic falsifier using the
already acquired assets, followed by source-free shape/loading checks and a
separately selected feature resource pilot. Reuse successful acquisition
checks; there is no reason to download the checkpoints again or repeat a
generic provenance chain. No pilot resource request or image access is
authorized by this document.

The exact event-matched authorized pixel adapter and physical reader boundary
still govern feature execution. Only development features can enter the
search worker. Calibration concepts cannot fit or score a candidate model;
their candidate features are unnecessary for the primary linking analysis.
Audit embeddings remain evaluator-only until the final write-once lock and
cannot be inspected to choose layers, preprocessing or a model pair.

The common preprocessing, four-layer pooling and single-seed falsifier were
adopted October 4 at 22:51 UTC. The subsequent linking specification fixes the
common grids and numerical rules. The coordinator adopted the reviewed practical-effect,
forecast-error and audit-adequacy margins in
[prediction qualification](prediction_qualification_specification.md) on
October 4, 23:24 UTC, before any candidate score. Structural ROI/support
qualification and the exact EP17/EP18 metadata-overlap decision subsequently
completed; actual calibration, isolated feature delivery and the one-shot
evaluator remain unfinished. This registry does not waive those dependencies
or activate neural scoring.
