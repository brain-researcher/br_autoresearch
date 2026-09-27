"""Semantic resource authorization for the expensive EP12 full-search null."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .policy import EpisodePolicy
from .yaml_subset import load_yaml_subset


AUTHORIZATION_SOURCE = "explicit_user_instruction_in_active_EP12_Codex_task"
AUTHORIZATION_SCOPE = (
    "minimum_full_development_search_including_observed_and_required_99_null_searches"
)


class ResourceAuthorizationError(RuntimeError):
    """The requested full-null program is not explicitly resource-authorized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceAuthorizationError(message)


def _read(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = load_yaml_subset(path)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ResourceAuthorizationError(
            f"cannot read resource authorization {path}: {error}"
        ) from error
    _require(isinstance(value, Mapping), "resource authorization must be a mapping")
    return dict(value)


def verify_resource_authorization(
    *,
    authorization_path: Path,
    policy: EpisodePolicy,
    procedure_lock: Mapping[str, Any] | None = None,
    runtime_bundle: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Verify authorization by scope, finite limits, and data-role boundaries."""

    _require(
        not authorization_path.is_symlink(),
        "authorization may not be a symbolic link",
    )
    path = authorization_path.resolve(strict=True)
    _require(path.is_file(), "authorization is not a safe file")
    authorization = _read(path)
    scope = authorization.get("scope")
    source = authorization.get("authorization_source")
    _require(
        scope == AUTHORIZATION_SCOPE,
        "resource authorization does not cover the full 99-replicate program",
    )
    _require(
        source == AUTHORIZATION_SOURCE,
        "resource authorization does not come from the explicit user instruction",
    )
    _require(
        authorization.get("cpu_budget_stop_disabled_until_minimum_full_search_complete")
        is True
        or authorization.get("allow_full_search_null_dispatch") is True,
        "full-search-null dispatch is not authorized",
    )
    _require(
        authorization.get("final_connectivity_access_authorized") is False,
        "resource authorization may not authorize final connectivity",
    )
    if procedure_lock is not None:
        _require(
            procedure_lock.get("procedure_locked") is True
            and procedure_lock.get("null_eligible") is True
            and procedure_lock.get("final_connectivity_accessed") is False
            and procedure_lock.get("final_connectivity_access_authorized") is False,
            "resource authorization requires an eligible development-only procedure lock",
        )

    authorized = authorization.get("measured_planning_core_hours")
    if authorized is None:
        authorized = authorization.get("authorized_core_hours")
    _require(
        isinstance(authorized, (int, float))
        and not isinstance(authorized, bool)
        and math.isfinite(float(authorized))
        and float(authorized) > 0,
        "authorized core hours must be positive and finite",
    )
    if runtime_bundle is not None:
        plan_path = runtime_bundle.resolve(strict=True) / "resource_plan.json"
        plan = _read(plan_path)
        projected = plan.get("projected_maximum_program_allocated_core_hours")
        _require(
            isinstance(projected, (int, float))
            and not isinstance(projected, bool)
            and math.isfinite(float(projected)),
            "runtime resource plan lacks a finite maximum projection",
        )
        _require(
            float(projected) <= float(authorized),
            "runtime resource projection exceeds the explicit authorization",
        )
    limits = authorization.get("unchanged_limits", {})
    if isinstance(limits, Mapping):
        _require(
            int(limits.get("minimum_valid_trials", policy.minimum_valid_trials))
            == policy.minimum_valid_trials
            and int(limits.get("maximum_valid_trials", policy.maximum_valid_trials))
            == policy.maximum_valid_trials,
            "resource authorization changes the bounded-search trial limits",
        )
    return {
        "authorization_path": str(path),
        "authorization_source": source,
        "scope": scope,
        "authorized_core_hours": float(authorized),
        "required_null_replicates": policy.null_replicates,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


__all__ = [
    "AUTHORIZATION_SCOPE",
    "AUTHORIZATION_SOURCE",
    "ResourceAuthorizationError",
    "verify_resource_authorization",
]
