"""Semantic contract epochs for the observed prefix and post-36 tail.

The historical transition proof remains in the frozen prelaunch archive. Live
code needs only the scientifically consequential epoch and prevalence-rank
definition; it does not rebuild an attestation or content-address chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .yaml_subset import load_yaml_subset


PREFIX_EPOCH = "observed_prefix_v1"
ACTIVE_EPOCH = "post36_active_v2"
RANK_RULE = "representation_latent_effective_rank"


class ContractRoleError(ValueError):
    """The semantic contract role is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractRoleError(message)


def validate_contract_provenance(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize old provenance records to the two scientific epoch labels."""

    raw = dict(value or {})
    rank_rule = raw.get("prevalence_parameter_rank_definition", RANK_RULE)
    _require(rank_rule == RANK_RULE, "post-36 prevalence-rank definition changed")
    return {
        "prefix_epoch": PREFIX_EPOCH,
        "active_epoch": ACTIVE_EPOCH,
        "prevalence_parameter_rank_definition": RANK_RULE,
        "operator_development_outcome_values_displayed": bool(
            raw.get("operator_development_outcome_values_displayed", True)
        ),
        "values_used_for_transition_decisions": bool(
            raw.get("values_used_for_transition_decisions", False)
        ),
        "final_connectivity_accessed": bool(
            raw.get("final_connectivity_accessed", False)
        ),
    }


def load_contract_provenance(
    *,
    prefix_contract_path: Path,
    active_contract_path: Path,
) -> dict[str, Any]:
    """Check the two contract documents' semantic role without receipt chaining."""

    prefix = load_yaml_subset(prefix_contract_path)
    active = load_yaml_subset(active_contract_path)
    _require(
        prefix.get("contract_id") == "ep12_outgoing_tum_v1",
        "unexpected observed-prefix scientific contract",
    )
    _require(
        active.get("contract_id") == "ep12_outgoing_tum_post36_v2",
        "unexpected post-36 scientific contract",
    )
    rank_rule = active.get("mode_support", {}).get(
        "prevalence_parameter_rank_definition"
    )
    _require(rank_rule == RANK_RULE, "post-36 prevalence-rank definition changed")
    return validate_contract_provenance(
        {"prevalence_parameter_rank_definition": rank_rule}
    )


def prefix_record_fields(provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_contract_provenance(provenance)
    return {"contract_epoch": PREFIX_EPOCH}


def active_record_fields(provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_contract_provenance(provenance)
    return {
        "contract_epoch": ACTIVE_EPOCH,
        "prevalence_parameter_rank_definition": RANK_RULE,
    }


def require_record_fields(
    record: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
    *,
    active: bool,
    label: str,
) -> None:
    expected = active_record_fields(provenance) if active else prefix_record_fields(provenance)
    for key, expected_value in expected.items():
        observed = record.get(key)
        if key == "prevalence_parameter_rank_definition" and observed is None:
            # Historical active records predate this explicit convenience field;
            # their epoch already selects the post-36 definition.
            continue
        _require(observed == expected_value, f"{label} contract role mismatch: {key}")


__all__ = [
    "ACTIVE_EPOCH",
    "PREFIX_EPOCH",
    "RANK_RULE",
    "ContractRoleError",
    "active_record_fields",
    "load_contract_provenance",
    "prefix_record_fields",
    "require_record_fields",
    "validate_contract_provenance",
]
