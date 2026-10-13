# Episode 05 inputs

This read-only directory is reserved for the Dryad sessions, publication
assets, and reference analysis code described in `../DATASETS.md`. Those
materials are accessed in place at the identified read-only provider source:
`$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source`.
The October 3 source inspection identified 24 session MAT files plus README.
Source/support records are under `../outputs/runs/source_support/attempt001/`;
that historical attempt checked the Main-example window, with the paper-window
check separately selected as attempt 002. No source payload is copied here.

Before scoring, record which release is being used, which sessions are
eligible, which animal and implant each session belongs to, and which trials
share behavior, LFP, and spikes. The active episode must not write into this
directory or replace a source after outcome access. All generated results
belong under `../outputs/`.
