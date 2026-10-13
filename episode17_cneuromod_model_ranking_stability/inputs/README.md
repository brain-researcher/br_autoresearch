# EP17 inputs

Large neural, structural, stimulus, and feature payloads stay outside Git.
Their authorized locations and scientific roles are described in
[../DATASETS.md](../DATASETS.md).

The available CNeuroMod source still mixes development, calibration, and
audit concepts. It is not a search-worker input. The trusted builder must
create role-filtered metadata, calibration, development, and sealed-audit
handoffs before neural model comparison.

Do not place neural arrays, THINGS image pixels, credentials, model caches, or
audit results in this directory.

The three nominated model checkpoints were acquired on 2026-10-03 under
[`outputs/cache/model_checkpoints/`](../outputs/cache/model_checkpoints/),
outside Git. See the [acquisition and CPU-load checks](../outputs/experiment_log.md#2026-10-03---model_pair_registry---checkpoint-acquisition-attempt-001)
for exact files and official download sources. Their presence does not open
neural access or complete the model/exposure registry.
