"""Deliver complete development image means from caller-admitted role copies.

DATASETS.md's response construction and SEARCH_POLICY.yaml:response require
equal retained-repeat means, unchanged provider scaling and original columns.
The caller supplies the completed development-only row maps/image manifest,
already-open private role files and calibration-frozen A_all axes. Matching
metadata does not authenticate their origin or establish reader separation.
There is no path discovery, file opening, persistence, access grant or fitting.

All metadata and twelve copied HDF5 headers are checked before values. Each
needed physical512-column stripe is read once per participant/stage for both
ROIs. Every eligible repeat is used in acquisition order; nonfinite frozen
columns refuse the whole delivery. Ineligible images are excluded only by the
supplied frozen metadata. No response-driven column or image selection occurs.
The existing nested fitter separately checks partition/block variance and SST.

Resource admission precedes allocations, reads and mean reductions. The caller
owns open-file cache sizes, allocation/time limits and source immutability.
Returned complete panels own immutable byte-backed float64 means; no partial
panel bank is returned after a failure. This module reads no audit role.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

import calibration_core as calibration
import endpoint_evaluator as endpoint
import materialize_neural_roles as roles
import materialize_neural_roles_columns as copied
import nested_prediction as nested


class ResponseDeliveryError(ValueError):
    """A complete development handoff was refused; no selective rescue."""


def require(condition, code):
    if not condition:
        raise ResponseDeliveryError(code)


@dataclass(frozen=True)
class ResourceEvent:
    operation: str
    when: str = "before"
    participant: str | None = None
    stage: str | None = None
    group: tuple | None = None
    column_start: int | None = None
    column_stop: int | None = None
    shape: tuple = ()
    byte_count: int = 0


def _event(check, operation, **context):
    check(ResourceEvent(operation, **context))


def _integer(value, name, minimum=0):
    # CSV numeric fields have a declared integer meaning; image/concept/fold/
    # block identities never pass through this parser or through str().
    if isinstance(value, str):
        require(value.isascii() and value.isdecimal(), name + "_invalid")
        value = int(value)
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and value >= minimum, name + "_invalid")
    return int(value)


def _keys(value, expected, code):
    require(isinstance(value, Mapping) and set(value) == set(expected), code)


def _identity(row, image, concept, fold, block):
    require(row.get("concept_name") == concept
            and row.get("role") == "development"
            and row.get("outer_fold") == fold and type(row.get("outer_fold")) is type(fold)
            and row.get("development_block") == block and type(row.get("development_block")) is type(block)
            and row.get("audit_block") == "", "development_image_identity_changed")
    require(isinstance(image, str) and image, "provider_image_identity_invalid")


def _bind_metadata(design, image_manifest, row_maps, frozen_columns, check):
    require(isinstance(design, nested.DevelopmentDesign), "complete_development_design_required")
    require(isinstance(image_manifest, (tuple, list)) and len(image_manifest) == nested.DEVELOPMENT_IMAGES,
            "complete_development_image_manifest_required")
    _keys(row_maps, endpoint.SUBJECTS, "all_four_development_row_maps_required")
    _keys(frozen_columns, endpoint.GROUPS, "all_eight_frozen_A_all_axes_required")
    _event(check, "bind_metadata")
    manifest, concepts = {}, {}
    for index, row in enumerate(image_manifest):
        require(isinstance(row, Mapping), "image_manifest_row_invalid")
        image, concept = design.image_ids[index], design.concept_ids[index]
        fold, block = design.fold_ids[index], design.block_ids[index]
        require(row.get("provider_image_path") == image and image not in manifest
                and row.get("eligible_image") == "True" and row.get("common_to_four") == "True",
                "frozen_eligible_image_manifest_changed")
        _identity(row, image, concept, fold, block)
        counts = {person: _integer(row.get("retained_count_" + person), "retained_image_count", 2)
                  for person in endpoint.SUBJECTS}
        manifest[image] = (concept, fold, block, counts)
        if concept in concepts:
            require(concepts[concept] == (fold, block), "concept_partition_changed")
        concepts[concept] = (fold, block)

    columns, repeat_rows, row_counts, stripes = {}, {}, {}, {}
    for group in endpoint.GROUPS:
        _event(check, "bind_columns", group=group)
        axis = endpoint.column_ids(frozen_columns[group], "frozen_A_all")
        require(bool((axis < endpoint.support.PARTICIPANT_COLUMNS[group[0]]).all()),
                "original_column_outside_participant")
        columns[group] = axis
    for person in endpoint.SUBJECTS:
        _event(check, "bind_participant_rows", participant=person)
        rows = row_maps[person]
        require(isinstance(rows, (tuple, list)) and 0 < len(rows) <= roles.SHAPES[person][0],
                "complete_role_local_rows_required")
        row_counts[person] = len(rows)
        selected = {image: [] for image in design.image_ids}
        original_rows, acquisition_ranks = set(), set()
        for local, row in enumerate(rows):
            require(isinstance(row, Mapping) and row.get("participant") == person
                    and _integer(row.get("local_row_index_zero_based"), "local_row_index") == local,
                    "role_local_row_axis_changed")
            original = _integer(row.get("glmsingle_trial_index_zero_based"), "original_trial_index")
            require(original < roles.SHAPES[person][0] and original not in original_rows,
                    "original_trial_row_duplicate_or_outside_source")
            original_rows.add(original)
            image, concept = row.get("expected_provider_image_path"), row.get("concept_name")
            require(concept in concepts, "role_row_concept_outside_development")
            fold, block = concepts[concept]
            _identity(row, image, concept, fold, block)
            require(row.get("eligible_image") in ("True", "False"), "row_eligibility_invalid")
            if row["eligible_image"] == "False":
                require(image not in manifest, "frozen_image_eligibility_changed")
                continue
            require(image in manifest and manifest[image][:3] == (concept, fold, block),
                    "eligible_row_outside_frozen_image_roster")
            require(row.get("ordinal_status") == "valid", "retained_repeat_chronology_changed")
            acquisition = _integer(row.get("retained_acquisition_rank"), "retained_acquisition_rank")
            require(acquisition < roles.SHAPES[person][0] and acquisition not in acquisition_ranks,
                    "retained_acquisition_rank_duplicate_or_outside_source")
            acquisition_ranks.add(acquisition)
            selected[image].append((acquisition, local,
                _integer(row.get("retained_repeat_rank"), "retained_repeat_rank"),
                _integer(row.get("repeat_half"), "repeat_half", 1)))
        image_rows = []
        for image in design.image_ids:
            ordered = sorted(selected[image])
            require(len(ordered) == manifest[image][3][person], "frozen_retained_repeat_count_changed")
            require(all(item[2] == index and item[3] == 1 + index % 2
                        for index, item in enumerate(ordered)), "retained_repeat_chronology_changed")
            image_rows.append(tuple(item[1] for item in ordered))
        repeat_rows[person] = tuple(image_rows)
        # Original IDs, not positional group columns, determine the physical
        # stripe join. Reuse each stripe even when both ROIs touch it.
        groups = tuple((person, roi) for roi in endpoint.ROIS)
        starts = sorted({int(column) // copied.COLUMNS * copied.COLUMNS
                         for group in groups for column in columns[group]})
        plans = []
        for start in starts:
            stop = min(start + copied.COLUMNS, endpoint.support.PARTICIPANT_COLUMNS[person])
            ids = sorted({int(column) for group in groups for column in columns[group]
                          if start <= column < stop})
            locations = {column: index for index, column in enumerate(ids)}
            joins = {}
            for group in groups:
                positions = [index for index, column in enumerate(columns[group]) if start <= column < stop]
                if positions:
                    joins[group] = (np.asarray(positions, dtype=np.int64),
                                   np.asarray([locations[int(columns[group][index])] for index in positions], dtype=np.int64))
            plans.append((start, stop, np.asarray([column - start for column in ids], dtype=np.int64), joins))
        stripes[person] = tuple(plans)
    return columns, repeat_rows, row_counts, stripes


def build_development_responses(design, image_manifest, row_maps, stage_files, frozen_columns, *, resource_check):
    """Return eight complete ResponsePanels, without opening or releasing files.

    image_manifest is the ordered2423-row eligible development projection of
    image_assignments.csv. It includes the four retained_count_<participant>
    fields. row_maps contains complete per-participant development rows from
    the copied source_index_map.csv, including metadata-ineligible images.
    stage_files[(participant, stage)] is an already-open caller-owned HDF5
    handle for that completed ordinary development copy. The caller binds its
    authentic completed-copy identity; this routine checks format and alignment.
    """
    require(callable(resource_check), "external_resource_callback_required")
    _event(resource_check, "begin")
    _keys(stage_files, {(person, stage) for person in endpoint.SUBJECTS for stage in endpoint.STAGES},
          "all_twelve_development_stage_files_required")
    columns, repeats, row_counts, stripes = _bind_metadata(
        design, image_manifest, row_maps, frozen_columns, resource_check)
    datasets = {}
    for person in endpoint.SUBJECTS:
        for stage in endpoint.STAGES:
            shape = (row_counts[person], 1, 1, endpoint.support.PARTICIPANT_COLUMNS[person])
            _event(resource_check, "validate_header", participant=person, stage=stage, shape=shape)
            datasets[person, stage] = copied.output_dataset(stage_files[person, stage], shape, "<f4")
    # All metadata and all12 headers have passed before any data selection.
    result = {group: {} for group in endpoint.GROUPS}
    for person in endpoint.SUBJECTS:
        groups = tuple((person, roi) for roi in endpoint.ROIS)
        for stage in endpoint.STAGES:
            means = {}
            for group in groups:
                shape = (nested.DEVELOPMENT_IMAGES, len(columns[group]))
                _event(resource_check, "allocate_image_means", participant=person, stage=stage,
                       group=group, shape=shape, byte_count=8 * shape[0] * shape[1])
                means[group] = np.empty(shape, dtype=np.float64)
            for start, stop, offsets, joins in stripes[person]:
                context = dict(participant=person, stage=stage, column_start=start, column_stop=stop)
                shape = (row_counts[person], stop - start)
                _event(resource_check, "read_stripe", **context, shape=shape, byte_count=4 * shape[0] * shape[1])
                block = datasets[person, stage][:, 0, 0, start:stop]
                require(isinstance(block, np.ndarray) and block.shape == shape and block.dtype == np.dtype("<f4"),
                        "development_stripe_representation_changed")
                for image_index, rows in enumerate(repeats[person]):
                    shape = (len(rows), len(offsets))
                    _event(resource_check, "slice_repeats", **context, shape=shape, byte_count=4 * shape[0] * shape[1])
                    values = block[np.ix_(rows, offsets)]
                    require(bool(np.isfinite(values).all()), "nonfinite_frozen_response_repeat")
                    _event(resource_check, "average_repeats", **context, shape=shape,
                           byte_count=8 * (shape[0] + 1) * shape[1])
                    mean, centered = calibration.center(values)
                    require(bool(np.isfinite(mean).all()), "nonfinite_response_image_mean")
                    for group, (positions, take) in joins.items():
                        means[group][image_index, positions] = mean[take]
                    del values, mean, centered
                del block
            for group in groups:
                values = means.pop(group)
                _event(resource_check, "freeze_panel", participant=person, stage=stage, group=group,
                       shape=values.shape, byte_count=values.nbytes)
                result[group][stage] = endpoint.frozen_array(values)
                del values
    return MappingProxyType({group: nested.ResponsePanel(design, columns[group], MappingProxyType(result[group]))
                             for group in endpoint.GROUPS})
