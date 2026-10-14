"""Optional, separately selected L40S admission for seventeen existing jobs.

The four-update measurement is evidence, never authority by itself. Original
per-pair grants and all allocation/training checks remain in the two callers.
The running pair000 and completed pair005 do not read this amendment.
"""
import json
import os

from run_content_cuda_pair import _read_text, _require, _same


AMENDMENT_RELATIVE = "outputs/content_cuda_l40s_production_hardware_amendment.json"
PENDING_STAGE_JOBS = {
    "roster-006": "47382243", "roster-007": "47382244",
    "remaining-001": "47382245", "remaining-002": "47382246", "remaining-003": "47382247",
    "remaining-008": "47382248", "remaining-009": "47382249", "remaining-010": "47382250",
    "remaining-011": "47382251", "remaining-012": "47382252", "remaining-013": "47382253",
    "remaining-014": "47382254", "remaining-015": "47382255", "remaining-016": "47382256",
    "remaining-017": "47382257", "remaining-018": "47382258", "remaining-019": "47382259",
}
AMENDMENT_FIXED = {
    "schema_version": "ep21.content_cuda_l40s_production_hardware.v1",
    "authorized": True,
    "operation": "extend_seventeen_pending_content_jobs_to_measured_L40S",
    "pending_stage_jobs": PENDING_STAGE_JOBS,
    "allowed_device_names": ["Tesla V100-SXM2-32GB", "NVIDIA H100 80GB HBM3", "NVIDIA L40S"],
    "scheduler_constraint": "(GPU_SKU:V100_SXM2&GPU_MEM:32GB)|(GPU_SKU:H100_SXM5&GPU_MEM:80GB)|(GPU_SKU:L40S&GPU_MEM:48GB)",
    "measurement_job": "47592902",
    "measurement_report": "outputs/runs/content_resource_measurement/cuda_l40s001/report.json",
    "original_grants_resources_science_unchanged": True,
    "selected_before_first_execution": True,
    "automatic_retry": False,
}


def load_amendment(base, stage_name):
    if stage_name not in PENDING_STAGE_JOBS:
        return None
    path = base / AMENDMENT_RELATIVE
    if not os.path.lexists(path):
        return None
    value = json.loads(_read_text(path))
    _require(isinstance(value, dict)
             and value.keys() == AMENDMENT_FIXED.keys() | {"approval_source", "selected_at"},
             "complete_L40S_production_hardware_amendment_required")
    for key, expected in AMENDMENT_FIXED.items():
        _require(_same(value[key], expected), "L40S_production_hardware_amendment_scope_changed")
    for key in ("approval_source", "selected_at"):
        _require(isinstance(value[key], str) and bool(value[key].strip()),
                 "separate_L40S_production_hardware_selection_required")
    return value
