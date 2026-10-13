# Finding B/C reconstruction source binding

Prepared October 11, 2026 from source text and retained qualification records only.
This note is not a runtime implementation, asset authentication or execution grant.
Finding B and C remain `not_evaluable_locked` in
[the existing lock](../protocol_lock.yaml). Its reconstruction reason is the
unqualified generator checkpoint/configuration/latent/DIST runtime binding.
[PROTOCOL_TABLE.yaml](../../PROTOCOL_TABLE.yaml), `freeze_rule`, requires an
explicit scientist amendment and regenerated lock before a post-lock,
pre-score change. Neither is produced here. Existing converter measurements
and their input grants are unchanged.

## Source identities and what is already verified

The fixed Wang source is V1.0.0, commit
`edbff02edd6a08d3c6c829af45a957fd14bdc460`. Reuse the authenticated
23,110,908-byte archive, MD5 `bb80dcc1a038737d6d8fc2c95b3e5586`, recorded in
[neutral conformance attempt005](../pipeline_conformance/attempt-005/neutral_conformance_report.json).
Its existing location is
`$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/source_qualification/attempt-001/KamitaniLab/InterSiteNeuralCodeConversion-V1.0.0.zip`.
Only these plaintext members were read for the reconstruction binding, under
prefix `KamitaniLab-InterSiteNeuralCodeConversion-edbff02/`:

- `reconstruction/recon_icnn_image_gd.py` (called **R** below; retained SHA256 `2f37541c98b1b9fd6562f21f788794f6c4b0062fe3c3c7fd4fcc40bd13d95444`).
- `reconstruction/config/recon_icnn_vgg19_relu7generator_gd_200iter_decoded_ImageNet.yaml` (SHA256 `0085210754b7b0384b02319d160e0f161b5aa8ebb65634bae8c477089a8e7c93`).
- `reconstruction/config/generator/relu7generator.yaml`, `reconstruction/config/encoder/vgg19.yaml`, and `reconstruction/data/files.json`.
- `evaluation/identification_accuracy/recon_image_eval_dnn.py` for the existing identification interface.

The selected dependency is BdPy 0.25, not a current package replacement.
Reuse its [source receipt](../source_pipeline_binding/bdpy_0_25/source_receipt.json)
and retained `bdpy-0.25-py3-none-any.whl` beside that receipt, 127,418 bytes,
SHA256 `dc027f301abce102fc63b47d55cd5b5db2be389c5dbcd3d8750bf73ea2835a48`.
`bdpy/recon/torch/icnn.py` (**I**) and `bdpy/recon/utils.py` (**U**) were read
directly as plaintext members of that wheel; they are not in the earlier
partial plaintext export. The existing plaintext
[`models.py`](../source_pipeline_binding/bdpy_0_25/plaintext/bdpy/dl/torch/models.py),
[`torch.py`](../source_pipeline_binding/bdpy_0_25/plaintext/bdpy/dl/torch/torch.py)
and [`feature.py`](../source_pipeline_binding/bdpy_0_25/plaintext/bdpy/feature/feature.py)
provide model dispatch, differentiable pre-ReLU hooks and feature normalization.
No module was imported, installed or executed in preparing this note.

## Required assets: documentary names versus authenticated payloads

| Asset | Exact documentary binding | Retained evidence and remaining limit |
| --- | --- | --- |
| Generator archive | `bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip`; Figshare file **52399790**, dedicated URL `https://ndownloader.figshare.com/files/52399790`; **832,692,892 bytes**, MD5 **8be66997045859a49b1d2bcef6640490** | [Attempt017](../source_qualification/attempt-017/qualification_report.json), lines 1354–1365, authenticates supplier metadata against the tagged manifest. It does not authenticate this archive's payload or member inventory. No filesystem-wide absence claim is made. |
| Generator checkpoint | Config expects `bvlc_reference_caffenet_generator_ILSVRC2012_Training/generator_relu7.pt` beneath `data/models/` | Expected archive member derived from the tagged extraction destination and config; actual member identity, serialized structure, tensor shapes and strict load are unverified. |
| Latent upper bound | Same directory, `relu7_act_range_3x.txt`, space-delimited, reshaped to `[4096]` | Expected member; actual values, finiteness, shape and valid bounds remain unverified. The filename does not establish how the source estimated these bounds. |
| Encoder | `VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt`, `ilsvrc_2012_mean.npy`, `estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat` | Reuse attempt017's authenticated VGG archive/member identities and [public conformance007](../pipeline_conformance/attempt-007/public_runtime_conformance_report.json): VGG strict load and mean-array shape/finite check passed. That check did not qualify reconstruction or consume the feature-SD statistic. |
| CaffeNet statistics named by generator config | `bvlc_reference_caffenet/ilsvrc_2012_mean.npy` and `estimated_cnn_feat_std_bvlc_reference_caffenet_ImgSize_227x227_chwise_dof1.mat` | Their archive/member identities were authenticated in attempt017. **R does not read these two generator-config fields**; it reads the encoder statistics instead. The CaffeNet classifier is not a substitute for the generator or identification AlexNet. |
| Custom DIST | Literal 19-entry `dists_alpha` and `dists_beta` dictionaries in **R:119–160**, also retained verbatim in attempt017 `runtime_and_reconstruction.dists_like_custom_loss` | Source coefficients are known. There is no separately referenced DISTS checkpoint to obtain. Their differentiable reconstruction execution is unverified. |

The tagged demo `decoded_features.zip` and Deeprecon decoder directory are
not LAION inputs. The ImageNet-training feature-mean ZIP listed in the demo
manifest is not read by this decoded-feature configuration: **R:292–300**
reads each selected decoder's own `model/y_mean.mat` instead. Reusing a demo
mean would change the selected method.

## Exact selected reconstruction behavior

The September 30 [decision record](../protocol_decisions_2026-09-30.yaml)
already selects all 19 layers, 200 SGD iterations, seed 42, no restarts and
final-iteration output. The facts below resolve source expressions; they do
not activate them or claim numerical conformance.

**Models and features.** Encoder name `vgg19` dispatches to BdPy `VGG19`;
generator name `relu7generator` dispatches to `AlexNetGenerator`, input width
4096, three 4096-wide linear stages, then transposed convolutions. Its hidden
LeakyReLU slope is **0.3**, independent of the converter's 0.2. Source topology
implies a 3×256×256 generator output, center-cropped to 3×224×224 by **I:345–359**;
this shape is a source-derived expectation, not an observed generator forward.
Both models enter eval mode. Only the latent tensor is in the SGD optimizer.

The encoder order is `conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2,
conv3_3, conv3_4, conv4_1, conv4_2, conv4_3, conv4_4, conv5_1, conv5_2,
conv5_3, conv5_4, fc6, fc7, fc8`. All units/channels enter every reconstruction
iteration; the converter's sampled-channel training rule does not apply.
Differentiable pre-ReLU activations use BdPy `FeatureExtractor(detach=False)`.

**Decoded-feature scaling, R:307–350.** The fixed config selects
`feature_std_train_mean_center`. For each layer, subtract that decoder's
training `y_mean`, normalize with `normalize_feature(...,
channel_wise_mean=False, channel_wise_std=False, channel_axis=0,
shift="self", scale=mean(public_feature_std[layer]), std_ddof=1)`, then restore
`y_mean`. In the pinned helper, for a convolutional `[C,H,W]` feature array,
the centering mean is global but its scalar SD denominator is the **mean of
the per-channel spatial sample SDs**; it is not the SD of the flattened array.
For a one-dimensional FC array the ordinary sample SD is used. Within-source
reconstructions require the source decoder mean; content and brain conversions
decoded by a target decoder require that target decoder's mean. No true test
features or test-dependent scale may substitute. The adapter still needs to
bind the existing frozen decoder means and restore exact feature axes.

**Loss, R:30–57 and I:379–396.** After scaling, form float32 layer norms and
`w_l = norm_l**(-2) / sum(norm**(-2))`. The actual loss is
`sum_l [w_l * sum((activation_l - target_l)**2) + DIST_l]`.
`DIST_l` is **minus 343639** times the sum of the literal alpha-weighted mean
similarity and beta-weighted variance/covariance similarity, with `c1=c2=1e-6`.
For convolutional layers the moments use axes `[2,3]` and similarities are
averaged; FC moments use all elements. The custom term is **not multiplied by
`w_l`**. Layers accumulate in reverse declared order. No new norm epsilon,
feature deletion or substitute modern DISTS implementation is specified.

**Latent initialization and updates.** **R:209** draws one unseeded upstream
`numpy.random.normal(0,1,[4096])` outside its subject/image loops and passes
that same initial array into every reconstruction. **I:280** copies it to
float32 per call. EP21's already selected seed-42 adaptation must bind this
same initialization behavior; no image-selected restart is allowed. The first
forward uses the unclipped normal draw. Each iteration `it=0..199` sets
`lr = 2 + it*(1e-10-2)/200`, momentum 0.9 and decay 0.01. Thus the final
learning rate is approximately **0.0100000000995**, not 1e-10.
Normalize the latent gradient by its mean absolute value only when positive;
perform SGD; multiply the updated latent by 0.99; clamp first to lower bound
0 and then to the supplied coordinatewise upper bound; recast float32
(**I:398–441**). Decay is an explicit post-step operation, not optimizer weight
decay. Jitter and blurring are disabled. No generator/encoder weights are fit.

**Color and returned output.** With the generator present, **I** does not
apply the caller's `preproc` to its generated tensor; the generated, cropped
tensor feeds VGG directly. **R** loads the encoder mean file at lines 107–108,
but does not bind that local value into the preprocessing callbacks. Its
`image_deprocess` therefore uses literal BGR means `[104,117,123]`, adds them,
reverses BGR to RGB and returns HWC. Do not silently replace these literals
with the VGG feature-export mean-file rule.

**I:410–416** saves the image generated **before** that iteration's latent
update and ultimately returns the last such image. It does not generate an
extra image after the 200th update. **R:376–385** writes that raw RGB result
and loss history to `recon_image-<ID>.mat`, then writes
`normalize_image(clip_extreme(result,pct=4))` as the final TIFF. In pinned
**U:8–24**, `pct=4` means global **2nd/98th percentile** clipping; the caller's
comment claiming 0.02% per tail is inconsistent with the executed expression.
Normalization maps the global minimum to 0, maximum to 255 when nonconstant,
then casts uint8. Every-ten-iteration JPEG snapshots use normalization without
this final clipping; they are not alternative identification/display outputs.
The caller's existing-directory skip is not evidence that a cell completed.
EP21 must retain every requested success/failure and display failure placeholders.

## Finding C: reuse the qualified public AlexNet amendment

The [October 3 amendment](../protocol_public_alexnet_amendment_2026-10-03.yaml)
replaces the private author checkpoint requirement in the next applicable
identification lock with the existing
`outputs/cache/model_checkpoints/public_bvlc_alexnet/bvlc_alexnet_public_converted.pt`.
[Conversion attempt001](../source_qualification/public_alexnet_conversion/attempt001.json)
records all 16 mapped tensors and all 32 fixture/layer comparisons passed in
job 46426204. It does not claim author-private checkpoint identity.
No new conversion or private checkpoint request is needed.

Preserve direct Lanczos resize to 227×227, RGB→BGR BCHW float32 in 0–255 units,
subtract `[104,117,123]`, eval mode, and pre-ReLU `conv1..conv5,fc6,fc7,fc8`.
Flatten each image in C order without added normalization. Identification
uses Pearson similarity against every other eligible image in the same frozen
test universe; finite exact ties are incorrect, and a nonfinite required
candidate makes the image unavailable rather than shrinking its denominator.
Pixel identification stays separate from the equal-eight-layer summary.
These rules and the passed asset conversion do not activate Finding C.

## Minimal remaining work and bounded asset handling

1. Authenticate the exact generator archive and its required checkpoint/bounds
   members, then establish strict compatible loading and those statistics'
   finite support. The supplier metadata above is reusable; do not repeat a
   broad source inventory or reacquire VGG/AlexNet. Read the pinned reconstruction
   helper already retained in the BdPy wheel, not a newly installed revision.
2. Prepare the original reconstruction adapter using the expressions above,
   existing decoded-feature collection, correct frozen decoder means and
   qualified public VGG feature-SD statistic. Runtime compatibility, actual
   memory/time, gradient execution and publication remain unmeasured. Required
   unavailable feature/model cells stay unavailable; no reduced favorable panel.
3. Before empirical activation, resolve the remaining runtime/asset fields,
   preserve the explicit amendment/regenerated-lock requirement, and freeze the
   actual display IDs from structural metadata under the existing first-four-ID,
   first-directed-pair rule. No stimulus, neural, decoded or scoring payload was
   inspected to prepare this note.

Existing [`audit_public_model_payloads.py`](../source_qualification/tools/audit_public_model_payloads.py)
contains reusable `download_or_reuse` (line 324), `inspect_zip` (line 169),
safe-member checks and 4-MiB streaming. Its existing CLI/`ASSETS` allowlist is
**VGG and CaffeNet only**; invoking it unchanged does not authenticate a
generator. Any selected generator asset step needs an explicit narrow binding
to the exact generator pin rather than broadening the old historical attempt.
The established downloader uses the dedicated `ndownloader.figshare.com` host
and verifies HTTP status, declared size and MD5 before publication.
[`convert_public_bvlc_alexnet.py`](../source_qualification/tools/convert_public_bvlc_alexnet.py)
and its completed conversion report remain reusable evidence; they are not a
generator conversion tool. The public-runtime VGG loader supplies a previously
used strict `weights_only=True` loading pattern, but its fixed VGG admission
and report are not generator authorization or generator runtime evidence.

The only established new transfer size is **832,692,892 bytes**. Required
uncompressed generator member sizes and peak runtime memory are unknown until
the archive is inspected. The existing archive inspector has a 2,000,000,000-byte
total and 1,500,000,000-byte per-member refusal bound. If deliberately reused
unchanged, one compressed archive plus at most that bounded extracted payload
would require at most **2,832,692,892 bytes**, excluding reused assets and small
reports; this is a conditional engineering ceiling, not an observed package
size or selected resource grant. No whole-archive extraction, asset download,
runtime import/test, job submission or lock mutation occurred here.
