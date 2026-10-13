#!/usr/bin/env python3
"""Explicit fresh pilot002 after pilot001's pre-access inherited-setgid refusal.

The original pilot implementation and report are preserved. Invoke with the
same --mode pilot and --capacity-observation arguments, in the selected Slurm
allocation under the unchanged 2 GiB / 120 s internal / 180 s external bounds.
Importing this module changes no paths and performs no work.
"""
from __future__ import annotations

import materialize_neural_roles as pilot
from neural_role_private_paths import private_directory


def main():
    pilot.PRIVATE = pilot.CACHE_PARENT / "private/role_materialization/pilot002"
    pilot.PUBLIC = pilot.EPISODE / "outputs/runs/role_materialization/pilot002"
    pilot.private_directory = private_directory
    return pilot.main()


if __name__ == "__main__":
    raise SystemExit(main())
