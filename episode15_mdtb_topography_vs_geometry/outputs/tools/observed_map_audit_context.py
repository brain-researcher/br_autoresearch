"""Inactive observed-map audit contracts; no file access or access authority.

The currently absent scientist amendment is required before examining any
supplied audit bundle. This module never inserts or adopts that amendment.
The runner must authenticate the 188 voxel identities against the existing
development selection before publishing predictions or opening audit Task B.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from calibration_falsifiers import VARIANTS
from development_lopo import DEVELOPMENT_IDS, _array
from local_functional_source import TASK_A_CONDITIONS
from observed_beta_summary import PARTITION_RUN_IDS
from registered_endpoint_metrics import CONDITIONS


AUDIT_IDS = ("sub-02", "sub-03", "sub-04", "sub-06", "sub-09", "sub-10",
             "sub-19", "sub-22", "sub-26", "sub-28", "sub-29", "sub-30")
ROUTE_ID = "observed_map_audit"
ADOPTION_KEY = "scientist_amendment_20261016_observed_map_audit"
SPECIFICATION = "outputs/proposed_observed_map_audit.md"
SUPPORT_SELECTION = "outputs/runs/observed_beta_development/finite_support001/selection.npz"


def require_adoption(policy):
    """Fail closed before any supplied audit value is converted or inspected."""
    if not isinstance(policy, Mapping):
        raise ValueError("an explicitly supplied episode policy is required")
    amendment = policy.get(ADOPTION_KEY)
    if (not isinstance(amendment, Mapping) or amendment.get("status") != "adopted"
            or amendment.get("specification") != SPECIFICATION
            or any(not isinstance(amendment.get(key), str) or not amendment[key].strip()
                   for key in ("authority", "adopted_at_utc"))):
        raise PermissionError("observed-map audit proposal is not explicitly adopted")
    return deepcopy(dict(amendment))


@dataclass(frozen=True)
class AuditContext:
    target_id: str
    training_ids: tuple[str, ...] = DEVELOPMENT_IDS
    deleted_id: None = None
    donor_id: str | None = None


def validate_audit_context(context):
    if (type(context) is not AuditContext or context.target_id not in AUDIT_IDS
            or context.training_ids != DEVELOPMENT_IDS or context.deleted_id is not None
            or (context.donor_id is not None
                and (context.donor_id not in AUDIT_IDS or context.donor_id == context.target_id))
            or set(context.training_ids).intersection(AUDIT_IDS)):
        raise ValueError("explicit audit target and all twelve development-only sources required")


def audit_identity_pairings():
    """Existing fixed cycle algorithm/seed applied to the canonical audit IDs."""
    indices = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([20261007, 1502]))).permutation(len(AUDIT_IDS))
    cycle = tuple(AUDIT_IDS[int(index)] for index in indices)
    successors = {person: cycle[(index + 1) % len(cycle)] for index, person in enumerate(cycle)}
    return tuple((person, successors[person]) for person in AUDIT_IDS)


def validate_prediction_bundle(bundle, *, policy, require_sampling_geometry=False):
    """Validate a complete frozen A-only panel, without reading audit B."""
    require_adoption(policy)
    if not isinstance(bundle, Mapping):
        raise ValueError("an explicit complete prediction bundle is required")
    expected_axes = {"development_ids": DEVELOPMENT_IDS, "audit_ids": AUDIT_IDS,
        "source_training_ids": DEVELOPMENT_IDS, "condition_ids": CONDITIONS,
        "variant_ids": VARIANTS, "partition_run_ids": PARTITION_RUN_IDS}
    if (bundle.get("status") != "complete_observed_map_audit_predictions"
            or bundle.get("route_id") != ROUTE_ID
            or bundle.get("numerical_certificates_complete") is not True
            or bundle.get("audit_Task_B_accessed") is not False):
        raise ValueError("complete certified audit-A-only prediction bundle required")
    checked = dict(bundle)
    for key, expected in expected_axes.items():
        supplied = bundle.get(key, ())
        actual = (tuple(tuple(row) for row in supplied) if key == "partition_run_ids" else tuple(supplied))
        if actual != expected:
            raise ValueError("the complete frozen " + key + " axis is required")
        checked[key] = expected
    voxel_ids = tuple(bundle.get("voxel_ids", ()))
    if (len(voxel_ids) != 188 or len(set(voxel_ids)) != 188
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                   for value in voxel_ids)):
        raise ValueError("all 188 ordered frozen development voxel identities required")
    checked["voxel_ids"] = tuple(map(int, voxel_ids))
    donors = tuple(bundle.get("donor_ids", ()))
    if donors != tuple(donor for _, donor in audit_identity_pairings()):
        raise ValueError("the unchanged fixed audit donor cycle is required")
    checked["donor_ids"] = donors
    mapping = tuple(bundle.get("condition_mapping", ()))
    rest = TASK_A_CONDITIONS.index("rest")
    if (len(mapping) != 29 or any(type(value) is not int for value in mapping)
            or set(mapping) != set(range(29)) or mapping[rest] != rest
            or any(mapping[index] == index for index in range(29) if index != rest)):
        raise ValueError("a fixed rest and complete non-rest condition derangement required")
    checked["condition_mapping"] = mapping
    # All identity checks above precede conversion of any prediction value.
    for key, shape in (("M0a", (18, 188)), ("M0b", (18, 188)),
                       ("predictions", (12, 16, 18, 188)),
                       ("condition_control_predictions", (12, 16, 18, 188)),
                       ("donor_control_predictions", (12, 16, 18, 188)),
                       ("random_basis_predictions", (12, 8, 18, 188)),
                       ("matched_warp_predictions", (12, 4, 8, 18, 188))):
        checked[key] = _array(bundle[key], shape, key)
    if require_sampling_geometry or "sampling_native_world" in bundle or "sampling_mask_headers" in bundle:
        headers = bundle.get("sampling_mask_headers", ())
        if len(headers) != 12 or any(not isinstance(header, Mapping) for header in headers):
            raise ValueError("all twelve anatomy-only sampling headers required")
        checked["sampling_native_world"] = _array(bundle.get("sampling_native_world"),
            (12, 188, 3), "frozen anatomy-only sampling coordinates")
        checked["sampling_mask_headers"] = tuple(deepcopy(dict(header)) for header in headers)
    return checked
