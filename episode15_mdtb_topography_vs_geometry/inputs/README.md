# Episode 15 inputs

Inputs are read-only. The acquired MDTB participant ZIPs mix Task A and Task B
and must not be mounted into a candidate run.

`../DATASETS.md` requires three operator-created, content-addressed handoffs:
development A+B, audit anatomy+A, and evaluator-only audit B. None is present
here, and the Task-B-blind 12/12 role manifest has not been instantiated. The
frozen 20-row coverage manifest is also absent. Do not add a symlink to the
mixed shared source, copy participant maps into Git, import a published
MDTB-fitted atlas, or infer launch permission from source availability.
