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
