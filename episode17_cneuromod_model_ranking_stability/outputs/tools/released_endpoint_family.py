"""Bind the existing derived calibration operators to EP17's fixed family.

Call only after the coordinator releases these exact derived artifacts. No
raw calibration, development or audit response, feature or score is read.
No calibration is repeated and no support or applicability is reselected.
The returned existing EndpointFamily snapshots all supplied arrays; recording
its ordered endpoints/edges, scales and inapplicability remains the caller's
pre-score obligation. Loading it alone grants no fitting or audit access.
"""
import json
import zipfile

import numpy as np

import endpoint_evaluator as endpoint
import released_development_inputs as existing

operators = existing.operators
FIELDS = ("A_all", "A_all_spatial_bins", "A_N_weights", "R_N_weights",
          "V_NC", "V_NC_ceiling_denominator")
EDGE_COMPONENTS = (("B_to_C", "H_BC"), ("C_to_D", "H_CD"))


class EndpointInputError(ValueError):
    """Fixed binding code; private reader exceptions are never serialized."""


def require(condition, code):
    if not condition:
        raise EndpointInputError(code)


def _group_decisions(applicability):
    names = {f"{person}_{roi}" for person, roi in endpoint.GROUPS}
    require(applicability.get("source_private_calibration") == str(operators.SOURCE_PRIVATE)
            and applicability.get("development_forecast_fitted") is False
            and applicability.get("candidate_scores_consumed") is False
            and applicability.get("worker_release_or_final_endpoint_eligibility_established") is False
            and set(applicability.get("groups", {})) == names,
            "completed_calibration_applicability_binding_required")
    dispositions = {}
    for name in sorted(names):
        group = applicability["groups"][name]
        require(group.get("full_endpoint_eligibility_requires_separate_development_checks") is True
                and set(group.get("beta_explanations", {})) == {edge for edge, _ in EDGE_COMPONENTS},
                "complete_group_applicability_required")
        dispositions[name] = {}
        for edge, component in EDGE_COMPONENTS:
            item = group["beta_explanations"][edge]
            require(type(item.get("applicable")) is bool
                    and item.get("axis_key") == name + "_A_all"
                    and item.get("gain_key") == name + "_" + edge + "_gain"
                    and item.get("beta_primary_endpoints_removed") is False
                    and item.get("frozen_columns_removed") is False,
                    "calibration_gain_binding_changed")
            require(item.get("reason") == "qualified_under_frozen_gain_rule" if item["applicable"]
                    else item.get("reason") in endpoint.GAIN_INAPPLICABILITY,
                    "calibration_gain_disposition_unresolved")
            dispositions[name][component] = item
    expected = {component: all(dispositions[name][component]["applicable"] for name in names)
                for _, component in EDGE_COMPONENTS}
    require(applicability.get("H_component_calibration_applicability") == expected,
            "global_gain_applicability_changed")
    return dispositions


def _required_keys(dispositions):
    keys = []
    for person, roi in endpoint.GROUPS:
        name = f"{person}_{roi}"
        keys.extend(name + "_" + field for field in FIELDS)
        for edge, component in EDGE_COMPONENTS:
            if dispositions[name][component]["applicable"]:
                keys.extend(name + "_" + edge + "_" + field for field in ("low_columns", "high_columns"))
    return tuple(keys)


def bind_released_endpoint_family(arrays, applicability):
    """Map the producer's exact key/decision schema to existing support checks."""
    dispositions = _group_decisions(applicability)
    require(set(_required_keys(dispositions)).issubset(arrays), "derived_support_vectors_missing")
    supports = {}
    for group in endpoint.GROUPS:
        name = "_".join(group)
        get = lambda field: arrays[name + "_" + field]
        strata = {}
        for edge, component in EDGE_COMPONENTS:
            item = dispositions[name][component]
            strata[component] = (endpoint.GainStrata(get(edge + "_low_columns"), get(edge + "_high_columns"))
                                 if item["applicable"] else endpoint.InapplicableGain(item["reason"]))
        supports[group] = endpoint.FrozenSupport(*(get(field) for field in FIELDS), strata)
        wanted = {key: True for key in endpoint.PRIMARY[:5]}
        wanted.update(D_VNC_raw=bool(len(get("V_NC"))), D_VNC_normalized=bool(len(get("V_NC"))))
        require(applicability["groups"][name].get("endpoint_calibration_prerequisites") == wanted,
                "calibration_endpoint_applicability_changed")
    return endpoint.EndpointFamily(supports)


def load_released_endpoint_family(*, metadata_byte_cap, array_byte_cap, resource_check):
    """Read only the fixed completed operator attempt; no automatic release."""
    try:
        require(callable(resource_check) and all(type(cap) is int and cap > 0
                for cap in (metadata_byte_cap, array_byte_cap)), "explicit_resource_bounds_required")
        check = resource_check
        check(existing.InputEvent("read_endpoint_operator_report"))
        report = existing._json(operators.PUBLIC / "operator_report.json", metadata_byte_cap, check)
        require(report.get("analysis_id") == "calibration_support" and report.get("status") == "completed"
                and report.get("errors") == [] and report.get("operator_artifacts_complete") is True
                and report.get("participant_ROI_groups_processed") == 8 and report.get("beta_group_edges_processed") == 16
                and report.get("private_attempt") == str(operators.PRIVATE)
                and report.get("source_private_calibration") == str(operators.SOURCE_PRIVATE),
                "completed_derived_operator_attempt_required")
        path = operators.PRIVATE / "applicability.json"
        check(existing.InputEvent("read_calibration_endpoint_applicability"))
        require(existing._ordinary_input_file(path, operators.PRIVATE) <= metadata_byte_cap,
                "endpoint_metadata_byte_cap")
        applicability = json.loads(existing.feature_io.archive.bounded_bytes(path, metadata_byte_cap))
        keys = _required_keys(_group_decisions(applicability))
        path = operators.PRIVATE / "operators.npz"
        check(existing.InputEvent("read_calibration_support_vectors"))
        require(existing._ordinary_input_file(path, operators.PRIVATE) <= array_byte_cap, "endpoint_array_byte_cap")
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require(len(entries) == len({item.filename for item in entries}) and len(entries) <= 512
                    and all(item.compress_type == zipfile.ZIP_STORED and not item.is_dir() for item in entries)
                    and sum(item.file_size for item in entries) <= array_byte_cap,
                    "derived_operator_container_changed")
        arrays, total = {}, 0
        with np.load(path, allow_pickle=False) as archive:
            require(set(keys).issubset(archive.files), "derived_support_vectors_missing")
            for key in keys:
                check(existing.InputEvent("read_one_calibration_support_vector"))
                value = archive[key]
                total += value.nbytes
                require(value.dtype.kind in "iuf" and value.ndim == 1 and total <= array_byte_cap,
                        "derived_support_vector_representation_changed")
                arrays[key] = value
        check(existing.InputEvent("bind_calibration_endpoint_family", byte_count=total))
        family = bind_released_endpoint_family(arrays, applicability)
        check(existing.InputEvent("bind_calibration_endpoint_family", when="after", byte_count=total))
        return family
    except EndpointInputError:
        raise
    except Exception:
        raise EndpointInputError("released_endpoint_family_binding_failed") from None
