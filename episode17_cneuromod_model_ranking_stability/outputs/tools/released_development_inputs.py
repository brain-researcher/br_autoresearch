"""Load the exact existing EP17 artifacts for the fixed prediction-cost probe.

Call only after the coordinator records the procedural builder handoff and
selects resources. No access grant, reader release, scoring, persistence, path
discovery or retry occurs here. Metadata and feature rows may cover development;
neural selections contain only fold03..fold06 repeats and the sixteen smallest
original A_all columns per group. No mixed source, audit or raw calibration
response file is opened.

The existing probe expects full ndarray headers. Unobserved response cells are
NaN placeholders, explicitly NOT complete ResponsePanels for scientific fitting.
The existing full fitter refuses them through its finite-value checks. HDF5
may internally cache physical chunks spanning unselected rows/columns; only
the exact admitted logical selections are returned to Python.
"""
from dataclasses import dataclass
import json
import re
import stat
from types import MappingProxyType
import zipfile

import h5py
import numpy as np

import calibrate_private_roles as calibration
import development_prediction_cost as cost
import development_responses as responses
import derive_private_operators as operators
import endpoint_evaluator as endpoint
import materialize_neural_roles as original
import materialize_neural_roles_columns as columns
import materialize_neural_roles_full as copied
import nested_prediction as nested
import private_development_feature_io as feature_io

COPY_PRIVATE, COPY_PUBLIC = calibration.COPY_PRIVATE, calibration.COPY_PUBLIC
FEATURE_PRIVATE = original.CACHE_PARENT / "private/stimulus_features"
FEATURE_PUBLIC = original.EPISODE / "outputs/runs/stimulus_features"
HDF_CACHE_BYTES = 1024 * 1024  # One open response handle at a time.


class DevelopmentInputError(ValueError):
    """Fixed operational code only; never serialize a private reader error."""


def require(condition, code):
    if not condition:
        raise DevelopmentInputError(code)


@dataclass(frozen=True)
class InputEvent:
    operation: str
    when: str = "before"
    byte_count: int = 0
    participant: str | None = None
    stage: str | None = None
    original_column: int | None = None


@dataclass(frozen=True)
class PredictionCostInputs:
    design: nested.DevelopmentDesign
    features: object
    responses: object
    frozen_columns: object
    operational_dimensions: object
    complete_development_response_release: bool = False


def _json(path, cap, check):
    check(InputEvent("read_operational_metadata"))
    return json.loads(feature_io.archive.bounded_bytes(path, cap))


def _ordinary_input_file(path, parent):
    """Inspect admitted inputs without changing inherited Oak setgid modes.

    Builder-time helpers require exact 0700 directories. Completed inputs may
    retain 0700 or 2700; both deny group/other access. Files retain the existing
    ordinary 0600, single-link requirement. This grants no reader access.
    """
    require(path.is_relative_to(parent), "private_file_outside_attempt")
    directories = [parent]
    for part in path.parent.relative_to(parent).parts:
        directories.append(directories[-1] / part)
    for current in directories:
        info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) in (0o700, 0o2700),
                "private_directory_permissions_invalid")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
            "private_file_representation_invalid")
    return info.st_size


def _copy_entries(report):
    require(report.get("status") == "completed" and report.get("mode") == "full"
            and report.get("full_copy_complete") is True and report.get("all_copy_bytes_equal") is True
            and report.get("neural_read_completed") is True and report.get("errors") == []
            and report.get("copy_layout") == columns.LAYOUT
            and report.get("private_attempt") == str(COPY_PRIVATE)
            and report.get("public_attempt") == str(COPY_PUBLIC)
            and report.get("role_assignment_attempt") == str(original.ASSIGNMENT),
            "completed_role_copy_attempt002_required")
    products = report.get("products", [])
    expected = {(person, stage) for person in endpoint.SUBJECTS for stage in endpoint.STAGES}
    require(len(products) == 12 and {(p["participant"], p["stage"]) for p in products} == expected,
            "twelve_copied_products_required")
    result = {}
    for item in products:
        person, stage = item["participant"], item["stage"]
        width = endpoint.support.PARTICIPANT_COLUMNS[person]
        require(item.get("complete") is True and item.get("verified_columns") == width
                and item.get("dtype_str") == "<f4", "copied_product_incomplete")
        entries = item.get("outputs", [])
        require(len(entries) == 3 and {entry["role"] for entry in entries} == set(original.ROLES),
                "copied_role_roster_changed")
        entry = next(entry for entry in entries if entry["role"] == "development")
        path = COPY_PRIVATE / "development" / person / f"{stage}.h5"
        require(entry.get("copy_bytes_equal") is True and entry.get("empty_role") is False
                and entry.get("verified_columns") == width and entry.get("array_file") == str(path)
                and type(entry.get("rows")) is int and 0 < entry["rows"] <= original.SHAPES[person][0]
                and entry.get("shape") == [entry["rows"], 1, 1, width]
                and entry.get("dtype_str") == "<f4"
                and type(entry.get("array_file_bytes")) is int and entry["array_file_bytes"] > 0,
                "development_copy_binding_changed")
        result[person, stage] = entry
    return result


def _row_maps(entries, cap, check):
    check(InputEvent("bind_frozen_development_row_map"))
    # Existing helper reads qualified projected metadata only; the returned
    # mixed-source path is discarded without opening or inspecting it.
    _, routed = copied.load_full_plan()
    selected = {person: [row for row in routed[person] if row["role"] == "development"]
                for person in endpoint.SUBJECTS}
    del routed
    path = COPY_PRIVATE / "development/source_index_map.csv"
    require(_ordinary_input_file(path, COPY_PRIVATE) <= cap, "development_map_byte_cap")
    fields, actual = original.stage.table_rows(feature_io.archive.bounded_bytes(path, cap), original.MAP_FIELDS)
    require(fields == list(original.MAP_FIELDS), "development_map_fields_changed")
    expected = [{field: str(local) if field == "local_row_index_zero_based" else row[field]
                 for field in original.MAP_FIELDS}
                for person in sorted(selected) for local, row in enumerate(selected[person])]
    require(actual == expected, "development_copy_map_not_frozen_projection")
    result = {person: tuple(row for row in actual if row["participant"] == person)
              for person in endpoint.SUBJECTS}
    require(all(entries[person, stage]["rows"] == len(result[person])
                for person in endpoint.SUBJECTS for stage in endpoint.STAGES), "development_copy_row_count_changed")
    return result


def _frozen_columns(cap, check):
    report = _json(operators.PUBLIC / "operator_report.json", cap, check)
    require(report.get("analysis_id") == "calibration_support" and report.get("status") == "completed"
            and report.get("errors") == [] and report.get("operator_artifacts_complete") is True
            and report.get("participant_ROI_groups_processed") == 8 and report.get("beta_group_edges_processed") == 16
            and report.get("private_attempt") == str(operators.PRIVATE)
            and report.get("source_private_calibration") == str(operators.SOURCE_PRIVATE),
            "completed_calibration_operators_required")
    path = operators.PRIVATE / "operators.npz"
    check(InputEvent("read_calibration_derived_axes"))
    require(_ordinary_input_file(path, operators.PRIVATE) <= cap, "operator_file_byte_cap")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len({entry.filename for entry in entries}) and len(entries) <= 512
                and all(entry.compress_type == zipfile.ZIP_STORED and not entry.is_dir() for entry in entries)
                and sum(entry.file_size for entry in entries) <= cap, "operator_container_changed")
    result = {}
    with np.load(path, allow_pickle=False) as arrays:
        for group in endpoint.GROUPS:
            check(InputEvent("read_one_calibration_derived_axis", participant=group[0]))
            axis = endpoint.column_ids(arrays[f"{group[0]}_{group[1]}_A_all"], "frozen_A_all")
            require(len(axis) >= cost.TARGET_COLUMNS
                    and bool((axis < endpoint.support.PARTICIPANT_COLUMNS[group[0]]).all()),
                    "calibrated_original_columns_changed")
            result[group] = axis
    return MappingProxyType(result)


def _training_plan(design, repeats, frozen_columns):
    require(set(design.fold_ids) == set(cost.FOLDS), "fixed_cost_fold_labels_required")
    train = np.asarray([i for i, fold in enumerate(design.fold_ids) if fold in cost.TRAINING_FOLDS], dtype=np.int64)
    require(len({design.concept_ids[i] for i in train}) == 320, "fixed_cost_training_concepts_required")
    positions = {group: np.argsort(frozen_columns[group])[:cost.TARGET_COLUMNS] for group in endpoint.GROUPS}
    plans = {}
    for person in endpoint.SUBJECTS:
        physical_rows = np.asarray(sorted({row for i in train for row in repeats[person][i]}), dtype=np.int64)
        lookup = {int(row): index for index, row in enumerate(physical_rows)}
        image_takes = tuple(np.asarray([lookup[row] for row in repeats[person][i]], dtype=np.int64) for i in train)
        groups = tuple(group for group in endpoint.GROUPS if group[0] == person)
        ids = np.asarray(sorted({int(column) for group in groups for column in frozen_columns[group][positions[group]]}), dtype=np.int64)
        locations = {int(column): index for index, column in enumerate(ids)}
        joins = {group: np.asarray([locations[int(column)] for column in frozen_columns[group][positions[group]]], dtype=np.int64)
                 for group in groups}
        plans[person] = (physical_rows, image_takes, ids, joins)
    return train, positions, plans


def _read_training_responses(design, repeats, frozen_columns, entries, *, response_byte_cap, check):
    train, positions, plans = _training_plan(design, repeats, frozen_columns)
    response_bytes = 3 * nested.DEVELOPMENT_IMAGES * sum(len(axis) for axis in frozen_columns.values()) * 8
    require(type(response_byte_cap) is int and response_byte_cap > 0, "explicit_cost_response_allocation_required")
    check(InputEvent("plan_cost_response_headers", byte_count=response_bytes))
    require(0 < response_bytes <= response_byte_cap,
            "explicit_cost_response_allocation_required")
    check(InputEvent("allocate_cost_only_response_headers", byte_count=response_bytes))
    values = {group: {stage: np.full((nested.DEVELOPMENT_IMAGES, len(frozen_columns[group])), np.nan, dtype=np.float64)
                      for stage in endpoint.STAGES} for group in endpoint.GROUPS}
    logical_values, selections = 0, 0
    for person in endpoint.SUBJECTS:
        rows, image_takes, ids, joins = plans[person]
        for stage in endpoint.STAGES:
            entry = entries[person, stage]
            path = COPY_PRIVATE / "development" / person / f"{stage}.h5"
            check(InputEvent("open_development_training_response", participant=person, stage=stage))
            with h5py.File(path, "r", rdcc_nbytes=HDF_CACHE_BYTES) as handle:
                dataset = columns.output_dataset(handle, entry["shape"], entry["dtype_str"])
                check(InputEvent("allocate_training_repeat_buffer", byte_count=len(rows) * len(ids) * 4,
                                 participant=person, stage=stage))
                retained = np.empty((len(rows), len(ids)), dtype=np.float32)
                for offset, column in enumerate(ids):
                    check(InputEvent("read_training_column", participant=person, stage=stage, original_column=int(column)))
                    selected = dataset[rows, 0, 0, int(column)]
                    require(selected.shape == (len(rows),) and selected.dtype == np.dtype("<f4")
                            and bool(np.isfinite(selected).all()), "training_response_values_invalid")
                    retained[:, offset] = selected
                    logical_values += len(rows)
                    selections += 1
                    del selected
            for image_index, take in zip(train, image_takes):
                check(InputEvent("average_training_repeats", participant=person, stage=stage))
                mean, centered = calibration.numerical.center(retained[take])
                require(bool(np.isfinite(mean).all()), "training_image_mean_invalid")
                for group, source_positions in joins.items():
                    values[group][stage][image_index, positions[group]] = mean[source_positions]
                del mean, centered
            del retained
    for stages in values.values():
        for value in stages.values():
            value.flags.writeable = False
    panels = MappingProxyType({group: nested.ResponsePanel(design, frozen_columns[group], MappingProxyType(values[group]))
                               for group in endpoint.GROUPS})
    return panels, {"response_header_bytes": response_bytes, "logical_training_response_values": logical_values,
                    "logical_training_column_reads": selections, "training_images": len(train),
                    "training_concepts": 320, "target_columns_per_group": cost.TARGET_COLUMNS,
                    "validation_or_fold01_response_values_returned": False,
                    "HDF_physical_cache_may_span_unselected_cells": True}


def load_development_prediction_cost_inputs(*, feature_attempt, feature_byte_cap, metadata_byte_cap,
                                          response_byte_cap, resource_check):
    """Load selected builder inputs; all caps/access/compute admission are external.

    This callable neither runs the probe nor releases a complete development
    response view. The coordinator must separately select the existing
    measure_development_prediction_cost callable. Preserve its report even if
    later full fitting is infeasible; no smaller fit is selected here.
    """
    try:
        require(callable(resource_check), "resource_callback_required")
        check = resource_check
        check(InputEvent("begin_cost_input_loading"))
        require(isinstance(feature_attempt, str) and re.fullmatch(r"full[0-9]{3,6}", feature_attempt) is not None
                and type(metadata_byte_cap) is int and metadata_byte_cap > 0, "explicit_input_scope_required")
        feature_report = _json(FEATURE_PUBLIC / feature_attempt / "feature_report.json", metadata_byte_cap, check)
        require(feature_report.get("analysis_id") == "stimulus_features" and feature_report.get("attempt") == feature_attempt
                and feature_report.get("status") == "completed_private_staging"
                and feature_report.get("full_development_features_complete") is True
                and feature_report.get("development_images") == 2423 and feature_report.get("development_concepts") == 480
                and feature_report.get("model_ids") == list(nested.MODEL_IDS)
                and feature_report.get("layers") == list(nested.LAYERS)
                and feature_report.get("neural_values_read") is False
                and feature_report.get("audit_pixels_or_features_opened") is False,
                "completed_full_development_feature_attempt_required")
        design, manifest = feature_io.load_development_metadata()
        entries = _copy_entries(_json(COPY_PUBLIC / "materialization_report.json", metadata_byte_cap, check))
        row_maps = _row_maps(entries, metadata_byte_cap, check)
        frozen_columns = _frozen_columns(metadata_byte_cap, check)
        bound_columns, repeats, _, _ = responses._bind_metadata(design, manifest, row_maps, frozen_columns, check)
        frozen_columns = MappingProxyType(bound_columns)
        # Complete all twelve ordinary-file identities/headers before any
        # response values; only development paths are checked or opened.
        for (person, stage), entry in entries.items():
            path = COPY_PRIVATE / "development" / person / f"{stage}.h5"
            check(InputEvent("validate_development_header", participant=person, stage=stage))
            require(_ordinary_input_file(path, COPY_PRIVATE) == entry["array_file_bytes"],
                    "development_copy_file_size_changed")
            with h5py.File(path, "r", rdcc_nbytes=HDF_CACHE_BYTES) as handle:
                columns.output_dataset(handle, entry["shape"], entry["dtype_str"])
        feature_path = FEATURE_PRIVATE / feature_attempt / "development/features.h5"
        require(feature_io.pilot._ordinary_private_file(feature_path).st_size == feature_report.get("feature_file_bytes"),
                "feature_report_file_size_changed")
        features = feature_io.load_complete_features(feature_path, design, manifest,
                                                     byte_cap=feature_byte_cap, check=check)
        panels, dimensions = _read_training_responses(design, repeats, frozen_columns, entries,
            response_byte_cap=response_byte_cap, check=check)
        check(InputEvent("complete_cost_input_loading", when="after"))
        return PredictionCostInputs(design, features, panels, frozen_columns, MappingProxyType(dimensions))
    except DevelopmentInputError:
        raise
    except Exception:
        raise DevelopmentInputError("development_cost_input_loading_failed") from None
