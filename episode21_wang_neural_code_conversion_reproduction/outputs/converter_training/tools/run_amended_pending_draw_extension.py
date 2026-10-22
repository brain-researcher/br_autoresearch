"""Extend only held, unstarted original draw children 40--61 to 48 hours.

The private caller keeps the original stage, array identities, scientific
operations and outputs. Live callers and their captured profile stay unchanged.
"""
from pathlib import Path
import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "_ep21_pending_draw_extension_caller", Path(__file__).with_name("run_amended_full_readout.py"))
caller = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = caller
_spec.loader.exec_module(caller)
original_profile = caller._profile
ROOT = caller._stage("draw", 0).root
SELECTION = ROOT / "pending_draw_extension001.json"
ARRAY = "47586235"
INDICES = tuple(range(40, 62))
RESOURCES = dict(cpus=3, memory_gib=16, gpus=0, minutes=2880,
    deadline_seconds=171000, numeric_threads=1)


def selection():
    chosen = json.loads(caller._text(SELECTION))
    caller._require(chosen.get("schema_version") == "ep21.pending_draw_clock_extension.v1"
        and chosen.get("selection_id") == "pending_draw_extension001"
        and chosen.get("authorized") is True
        and chosen.get("original_array_job_id") == ARRAY
        and caller._same(chosen.get("batch_indices"), list(INDICES))
        and caller._same(chosen.get("resources"), RESOURCES)
        and isinstance(chosen.get("approval_source"), str)
        and bool(chosen["approval_source"].strip()),
        "activated_exact_pending_draw_clock_extension_required")
    held = chosen.get("held_pending_children", {})
    caller._require(isinstance(held, dict) and set(held) == {str(i) for i in INDICES}
        and all(isinstance(held[str(i)], dict)
            and held[str(i)].get("array_job_id") == ARRAY
            and type(held[str(i)].get("batch_index")) is int
            and held[str(i)]["batch_index"] == i
            and held[str(i)].get("state") == "PENDING"
            and held[str(i)].get("reason") == "JobHeldUser"
            and type(held[str(i)].get("elapsed_seconds")) is int
            and held[str(i)]["elapsed_seconds"] == 0 for i in INDICES),
        "exact_unstarted_held_children_at_selection_required")
    return chosen


def profile(stage):
    caller._require(stage.phase == "draw" and type(stage.batch_index) is int
        and stage.batch_index in INDICES
        and os.environ.get("SLURM_ARRAY_JOB_ID") == ARRAY
        and os.environ.get("SLURM_ARRAY_TASK_ID") == str(stage.batch_index),
        "selected_original_pending_draw_child_required")
    chosen = selection()
    original = original_profile(stage)
    draw = original["resources"]["draw"]
    caller._require(caller._same({key: draw[key] for key in
        ("cpus", "memory_gib", "gpus", "minutes", "deadline_seconds")},
        dict(cpus=3, memory_gib=16, gpus=0, minutes=1440, deadline_seconds=84600)),
        "unchanged_original_draw_resource_baseline_required")
    resources = {phase: dict(values) for phase, values in original["resources"].items()}
    resources["draw"].update(minutes=2880, deadline_seconds=171000)
    return {**original, "resources": caller._resource_plan(resources),
        "pending_draw_extension_selection": chosen}


# No stage/binding/scientific implementation or live module is changed.
caller._profile = profile

if __name__ == "__main__":
    raise SystemExit(caller.main())
