# EP18 inputs

Large EEG, image, metadata, and feature payloads stay outside Git. Their
authorized locations and roles are listed in [../DATASETS.md](../DATASETS.md).

This directory is a documentation boundary only. Do not copy audit EEG,
restricted THINGS images, credentials, or large derived features here.

The deterministic 30/16 role assignment and event join can be frozen from
outcome-blind metadata without opening EEG. The selected preparation writes
disjoint, durable event metadata tables and separate deployment manifests.
Those manifests are file lists, not an implemented access boundary.

Before EEG-driven candidate comparison, deploy the development worker under
an evaluator-controlled identity or an externally controlled runtime/mount.
It receives only the 30 development participants under `/inputs/development_eeg`,
its development event table, approved feature inputs and its own output area.
The mixed preservation tree, audit table/EEG, sibling caches, evaluator outputs,
host credentials and host execution/socket interfaces must be absent from that
worker's accessible filesystem. A different identity alone is insufficient if
the public or group-readable audit source remains reachable.

After staging dependencies and authorized development inputs, the worker also
needs externally enforced network restrictions that prevent retrieving audit
EEG from OpenNeuro or mirrors. A runtime that the development process can
reconfigure to mount host paths or restore network access does not qualify.
Same-account permissions, ordinary symlinks and a self-controlled container
do not establish the declared seal. The evaluator keeps the 16 audit inputs
separate and runs the frozen procedure only after the registered opening gate.

The coordinator must verify the boundary from inside the actual development
runtime using denied-path and denied-download checks without reading audit
payloads, and record the tested identity/mount/network scope. No account,
firewall or shared infrastructure change is performed by this preparation.
This is the concrete deployment still needed; it does not delay authorized
outcome-blind source/feature work or request another approval for deterministic
metadata assignment.

The trusted coordinator may act as evaluator and control an isolated child
runtime; a second Unix account or new human approval is not itself required.
The child must lack any route back to the coordinator's host context. The
distinction is control over the actual filesystem/process/network boundary,
not merely an instruction to avoid audit files. A supported existing runtime
can be assessed on compute with harmless sentinel files before any data mount.
