"""Inactive, complete EP18 core64/cosine12 reference fitting connector.

The coordinator must explicitly release inputs, select the numeric recipe and
admit resources before any invocation. initialize binds all30 development
people, the complete72-sequence EEG cache, all22448 original UUIDs and every
completed shard of the five core image families. It streams the frozen role
join once and copies only structural event fields and the two binary nuisance
flags. No labels, RBFs, false banks, DINOv2 values, captions, RT, accuracy or audit
view are opened. Source imports of previously qualified helpers grant no data
access. Existing model-runtime/feature provenance remains required before EEG.

participant fits one entire original36-split reference: training-only image
scaling/PCA,36 nuisance fits,30x13 inner image fits,6 outer image/reduced fits.
Only three models are constructed. Image PCA/whitener/sensor/coefficients and
solver diagnostics for every outer fold, and all inner paired loss curves, are
published privately without pickle. Cost-only is the first frozen person,
using identical code/recipe and retaining reusable complete results. A stopped
person remains incomplete: there is no partial reference or automatic retry.
finalize requires all30 complete people, reports the registered G/T/margins,
and grants neither search credit nor candidate comparison/scientific release.

Eight-field release: development_manifest, eeg_cache, image_extraction_config,
private_routing, event_paths (exact30 mapping), feature_registry and
model_runtime_qualification, plus explicit image_source (original_shards or
completed_assembly). The latter uses the original run's completed assembly,
including its recorded task replacement lineage, never the failed original
shard. Every path is explicit; no discovery or available-case fallback.
Recipe numeric fields have no defaults. The CLI requires actual compute,
explicit thread/deadline/RSS/output caps; native time/RSS enforcement and the
episode ledger remain coordinator-owned. All command output is operational.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import re
import signal
import time
from types import MappingProxyType, SimpleNamespace

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
from continuous_eeg_cache import load_prepared_participant
from continuous_sequence_model import SequenceDesign, _PreparedTerm
from feature_extractors import DIMENSIONS
from prepare_continuous_eeg import SequenceEventInputs, SequencePreparationRecipe
import prepare_false_partition_inputs as events_io
import production_baseline_controller as controller
from production_baseline_controller import (CORE_MODELS, RunBudget, require, digest,
    _path, _json, _publish_json, _publish_arrays, _split_record)
from production_continuous_designs import (NUISANCE_RECIPE, BASELINE_TIE, EVENT_FIELDS,
    TemporalBasis, _kernels, _expanded, _compose, _event_arrays, _npz, readonly)
from run_development_eeg_preprocessing import COEFFICIENT_SHA, SPAN
from run_image_feature_shard import load_config, validate_manifest
from training_transforms import (CORE_IMAGE, CommonSensorSpace, FeaturePanelTransform,
    fit_feature_transform, fit_residual_whitener)

SCHEMA = "ep18_complete_core_reference_v1"
RELEASE_FIELDS = {"development_manifest", "eeg_cache", "image_extraction_config",
    "private_routing", "event_paths", "feature_registry", "model_runtime_qualification", "image_source"}
RECIPE_FIELDS = {"recipe_id", "image_panel", "image_rank", "temporal_basis", "nuisance_recipe",
    "intercept_penalty", "nuisance_penalty", "fit_weight_scale", "fit_weight_multiplier",
    "normalization", "covariance_centered", "covariance_denominator", "covariance_target",
    "shrinkage_strength", "baseline_tie_break", "solver", "metadata_byte_cap", "npz_byte_cap",
    "working_array_byte_cap"}


def validate_recipe(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    require(set(value) == RECIPE_FIELDS and isinstance(value["recipe_id"], str)
            and 0 < len(value["recipe_id"]) <= 128, "explicit_core_recipe_required")
    expected = {"image_panel": "core_image", "image_rank": 64, "temporal_basis": "raised_cosine_12",
        "nuisance_recipe": NUISANCE_RECIPE, "fit_weight_scale": "mean_sequences",
        "fit_weight_multiplier": 1, "normalization": "ratio_of_equal_sequence_means",
        "covariance_centered": True, "covariance_denominator": "sum_weights",
        "covariance_target": "trace_over_61_identity", "baseline_tie_break": BASELINE_TIE}
    require(all(value[key] == item for key, item in expected.items())
            and value["covariance_centered"] is True and type(value["image_rank"]) is int
            and type(value["fit_weight_multiplier"]) in (int, float), "fixed_core_conventions_required")
    require(all(type(value[key]) in (int, float) and np.isfinite(value[key]) and value[key] >= 0
                for key in ("intercept_penalty", "nuisance_penalty", "shrinkage_strength"))
            and value["shrinkage_strength"] <= 1, "explicit_nonnegative_penalties_and_shrinkage_required")
    require(set(value["solver"]) == {"atol", "btol", "conlim", "maxiter"}
            and all(type(value["solver"][key]) in (int, float) and np.isfinite(value["solver"][key])
                    and value["solver"][key] > 0 for key in ("atol", "btol", "conlim"))
            and value["solver"]["conlim"] > 1 and type(value["solver"]["maxiter"]) is int
            and value["solver"]["maxiter"] > 0, "finite_explicit_solver_limits_required")
    require(all(type(value[key]) is int and value[key] > 0 for key in
                ("metadata_byte_cap", "npz_byte_cap", "working_array_byte_cap")), "explicit_array_caps_required")
    return value


def _release(value, cap):
    require(set(value) == RELEASE_FIELDS, "exact_core_only_release_required")
    require(value["image_source"] in ("original_shards", "completed_assembly"), "explicit_image_source_required")
    for key in RELEASE_FIELDS - {"event_paths", "image_source"}:
        _path(value[key])
    development = _json(_path(value["development_manifest"]), cap)
    people = tuple(development["participants"])
    require(development["role"] == "development" and len(people) == len(set(people)) == 30
            and people == tuple(sorted(people)) and development["event_count"] == 739440
            and development["frozen_event_join"] == "development_event_image_join.tsv.gz"
            and isinstance(value["event_paths"], dict) and set(value["event_paths"]) == set(people),
            "complete_frozen_development_role_required")
    for person in people:
        require(isinstance(person, str) and len(person) == 6 and person.startswith("sub-")
                and person[4:].isdigit(), "development_identity_differs")
        path = _path(value["event_paths"][person])
        require(path.name == person + "_task-rsvp_events.tsv" and path.parent.name == "eeg"
                and path.parent.parent.name == person
                and f"{person}/eeg/{person}_task-rsvp_eeg.vhdr" in
                    development["EEG_relative_paths_for_controlled_mount"], "released_event_path_differs")
    return development, people


def _eeg_binding(release, people, cap):
    root = _path(release["eeg_cache"])
    eeg = _json(root / "complete.json", cap)
    original = _json(root / "settings.json", cap)
    require(Path(original["release"]["development_manifest"]) == _path(release["development_manifest"]),
            "EEG_cache_must_use_the_same_frozen_development_role")
    require(eeg["status"] == "complete_development_EEG_preprocessing" and eeg["participant_count"] == 30
            and set(eeg["participants"]) == set(people) and eeg["physical_sequences"] == 2160
            and eeg["main_events"] == 667440 and eeg["source_role"] == "development_only"
            and eeg["sample_unit"] == "V" and eeg["projection_or_training_transform_applied"] is False
            and eeg["model_scores_computed"] is False and eeg["audit_access"] is False
            and eeg["span_recipe"] == asdict(SPAN), "complete_original_EEG_cache_required")
    raw = dict(eeg["preparation_recipe"])
    require(raw.pop("coefficient_sha256") == COEFFICIENT_SHA, "frozen_FIR_coefficient_record_required")
    recipe = SequencePreparationRecipe(**raw)
    require(len(recipe.fir_coefficients) == 129 and recipe.fir_origin_sample == 0
            and recipe.decimation_phase_mod4 == 0 and recipe.detrend == "none"
            and recipe.source_sample_rate_hz == 1000 and recipe.target_sample_rate_hz == 250
            and recipe.input_sample_unit == "V" and recipe.operation_order == "detrend_then_fir_then_decimate"
            and hashlib.sha256(np.asarray(recipe.fir_coefficients, dtype="<f8").tobytes()).hexdigest() == COEFFICIENT_SHA,
            "original_EEG_preparation_conventions_required")
    channels = tuple(eeg["channel_order"])
    require(len(channels) == len(set(channels)) == 62 and all(
        Path(eeg["participants"][person]["directory"]) == root / person for person in people),
        "frozen_channels_and_person_paths_required")
    return eeg, recipe, channels


def _image_binding(release, cap, check):
    """Complete five-family producer records and headers; no details/pixels."""
    config_path = _path(release["image_extraction_config"])
    require(config_path.stat().st_size <= cap, "bounded_extraction_config_required")
    config = load_config(config_path)
    manifest = _json(Path(config["pixel_cache_dir"]) / "manifest.json", cap)
    ids = validate_manifest(manifest, config)
    run = Path(config["run_dir"])
    require(not (run / "pixel_delivery_failure.json").exists(), "completed_pixel_delivery_required")
    registry = _json(_path(release["feature_registry"]), cap)
    qualified = _json(_path(release["model_runtime_qualification"]), cap)
    require(registry["record_id"] == "ep18_feature_recipe_20261003"
            and tuple(registry["core_image_panel"]) == CORE_IMAGE
            and qualified["status"] == "asset_runtime_qualification_completed"
            and qualified["asset_runtime_qualified"] is True
            and qualified["image_or_EEG_or_concept_data_read"] is False
            and Path(qualified["registry"]) == _path(release["feature_registry"])
            and set(CORE_IMAGE[1:]).issubset(qualified["load_results"]), "existing_core_feature_provenance_required")
    asset_map = _json(_path(qualified["asset_map"]), cap)
    asset_root = Path(asset_map["asset_root"])
    if not asset_root.is_absolute():
        asset_root = Path(__file__).resolve().parents[2] / asset_root
    for key in ("supervised_resnet50_v1", "dino_resnet50", "clip", "blip"):
        require(Path(manifest["assets"][key]).resolve() ==
                (asset_root / asset_map["assets"][key]["local_path"]).resolve(), "qualified_core_assets_differ")
    families = {name: [] for name in CORE_IMAGE}
    assembly_id, source_record_id = None, None
    if release["image_source"] == "completed_assembly":
        assembled = run / "assembled"
        assembly = _json(assembled / "report.json", cap)
        assembled_ids = _json(assembled / "opaque_ids.json", cap)
        require(not (assembled / "failure.json").exists()
                and assembly["status"] == "all_image_features_completed"
                and assembly["images"] == 22448 and assembly["all_images_each_family_exactly_once"] is True
                and assembly["EEG_read"] is False and assembly["batch_size"] == 1
                and assembly["complete_tasks"] == len(manifest["tasks"])
                and set(assembly["families"]) == set(DIMENSIONS) and assembled_ids == list(ids),
                "complete_original_assembly_and_all_UUIDs_required")
        if "recovery" in assembly:
            recovery = assembly["recovery"]
            require(recovery["schema"] == "ep18_clip_task005_recovery001"
                    and recovery["replaced_task"] == manifest["tasks"][5]
                    and recovery["replaced_task"]["family"] == "clip_vit_b32_image"
                    and recovery["partial_rows_reused"] == 0 and recovery["automatic_retry"] is False
                    and recovery["original_config_sha256"] == digest(config)
                    and re.fullmatch(r"[0-9a-f]{64}", recovery["grant_sha256"])
                    and Path(recovery["original_config"]) == config_path
                    and Path(recovery["original_assembly_target"]) == assembled
                    and Path(recovery["preserved_failed_task"]) == run/"tasks"/"task_00005"
                    and _path(recovery["replacement_task"]) != run/"tasks"/"task_00005"
                    and assembly["source_record"] == "recovery_sources.json",
                    "explicit_complete_task5_replacement_lineage_required")
            sources = _json(assembled/"recovery_sources.json", cap)
            source_record_id = digest(sources)
            require(sources["binding"] == recovery and len(sources["tasks"]) == len(manifest["tasks"])
                    and [row["task_index"] for row in sources["tasks"]] == list(range(len(manifest["tasks"]))),
                    "complete_recovery_task_source_record_required")
            for row in sources["tasks"]:
                expected = (_path(recovery["replacement_task"]) if row["task_index"] == 5 else
                            run/"tasks"/("task_%05d" % row["task_index"]))
                require(Path(row["directory"]) == expected and isinstance(row["report_sha256"], str)
                        and re.fullmatch(r"[0-9a-f]{64}", row["report_sha256"]), "only_task5_source_may_change")
        else:
            require("source_record" not in assembly, "unrecognized_replacement_lineage_refused")
        # All six completion records, but only core feature files/values. The
        # assembled producer owns original/replacement task validation.
        for family in DIMENSIONS:
            prior = assembly["families"][family]
            require(prior["images"] == 22448 and prior["dimension"] == DIMENSIONS[family]
                    and prior["tasks"] == sum(task["family"] == family for task in manifest["tasks"]),
                    "assembly_family_coverage_differs")
            if family not in CORE_IMAGE:
                continue
            path = assembled / (family + ".npy")
            require(path.stat().st_size == prior["feature_bytes"]
                    and path.stat().st_size <= 22448*DIMENSIONS[family]*4+65536,
                    "bounded_complete_assembled_core_file_required")
            families[family].append({"path": str(path), "start": 0, "stop": 22448,
                "bytes": prior["feature_bytes"], "report_id": digest(assembly)})
        assembly_id = digest(assembly)
    for task in (manifest["tasks"] if release["image_source"] == "original_shards" else ()):
        if task["family"] not in families:
            continue
        check()
        directory = run / "tasks" / ("task_%05d" % task["task_index"])
        report = _json(directory / "report.json", cap)
        expected = ids[task["start"]:task["stop"]]
        require(not (directory / "failure.json").exists()
                and report["status"] == "complete_image_shard"
                and all(report[key] == value for key, value in task.items())
                and report["opaque_ids"] == expected and report["images"] == report["inferences"] == len(expected)
                and report["dimension"] == DIMENSIONS[task["family"]] and report["dtype"] == "float32"
                and report["batch_size"] == 1 and report["EEG_or_source_metadata_read"] is False,
                "every_core_shard_must_be_complete_and_identical")
        path = directory / "features.npy"
        require(path.stat().st_size == report["feature_bytes"]
                and path.stat().st_size <= len(expected) * DIMENSIONS[task["family"]] * 4 + 65536,
                "bounded_complete_core_feature_file_required")
        families[task["family"]].append({"path": str(path), "start": task["start"], "stop": task["stop"],
            "bytes": report["feature_bytes"], "report_id": digest(report)})
    # No feature body is accessed until every family's completion is known.
    for family in CORE_IMAGE:
        cursor = 0
        for shard in families[family]:
            require(shard["start"] == cursor and shard["stop"] > cursor, "core_shard_coverage_differs")
            values = np.load(shard["path"], mmap_mode="r", allow_pickle=False)
            require(values.dtype == np.float32 and values.shape == (shard["stop"] - cursor, DIMENSIONS[family]),
                    "core_feature_header_differs")
            del values
            cursor = shard["stop"]
        require(cursor == 22448, "all22448_images_in_each_core_family_required")
    evidence = {"config_id": digest(config), "pixel_manifest_id": digest(manifest),
        "registry_id": digest(registry), "runtime_qualification_id": digest(qualified), "asset_map_id": digest(asset_map),
        "image_source": release["image_source"], "assembly_id": assembly_id, "source_record_id": source_record_id}
    return tuple(ids), families, evidence


def initialize(release_path, recipe_path, output_dir, *, budget):
    budget.require_compute_threads()
    recipe = validate_recipe(_json(_path(recipe_path), budget.record))
    cap = recipe["metadata_byte_cap"]
    release = _json(_path(release_path), cap)
    development, people = _release(release, cap)
    eeg, _, _ = _eeg_binding(release, people, cap)
    ids, families, evidence = _image_binding(release, cap, budget.check)
    routing, main = events_io.load_routing(release["private_routing"], ids)
    index = {key: i for i, key in enumerate(ids)}
    root = _path(output_dir)
    root.mkdir(mode=0o700, exist_ok=False)
    _publish_json(root / "settings.json", {"schema": SCHEMA, "release": release, "recipe": recipe,
        "release_id": digest(release), "recipe_id": digest(recipe), "development_id": digest(development),
        "eeg_id": digest(eeg), "participants": people, "evidence": evidence, "families": families,
        "main_ids": main, "all_ids": ids, "main_to_feature": [index[key] for key in main],
        "audit_access": False, "scientific_release_authorized": False}, budget)
    event_dir = root / "events"
    event_dir.mkdir(mode=0o700)
    join = _path(release["development_manifest"]).parent / development["frozen_event_join"]
    require(join.stat().st_size <= events_io.JOIN_CAP, "bounded_frozen_development_join_required")
    records, consumed = {}, []
    with gzip.open(join, "rt", encoding="utf-8", newline="") as source:
        for person, frozen in itertools.groupby(csv.DictReader(source, delimiter="\t"), key=lambda row: row["participant"]):
            require(len(consumed) < 30 and person == people[len(consumed)], "original_role_join_order_required")
            original = events_io.source_main_rows(_path(release["event_paths"][person]), budget.check)
            arrays = events_io.participant_arrays(person, frozen, original, routing, main, budget.check)
            _event_arrays(arrays)
            size = _publish_arrays(event_dir / (person + ".npz"), arrays, budget)
            records[person] = {"bytes": size, "main_images": 22248, "physical_sequences": 72}
            consumed.append(person)
            del original, arrays
    require(tuple(consumed) == people, "all30_development_event_bindings_required")
    _publish_json(root / "inputs_complete.json", {"status": "complete_core_reference_inputs",
        "settings_id": digest(_json(root / "settings.json", budget.record)), "participants": records,
        "image_families": CORE_IMAGE, "images": 22448, "EEG_values_read": False,
        "audit_access": False, "labels_or_false_banks_read": False}, budget)
    return {"status": "core_reference_initialized", "participant_count": 30, "image_families": 5}


class CoreFeatureRows:
    """Header-qualified shards; at most one family's selected rows in RAM."""
    def __init__(self, families, check):
        self.families, self.check = families, check

    def rows(self, family, indices):
        require(family in CORE_IMAGE, "core_family_required")
        indices = np.asarray(indices)
        require(indices.ndim == 1 and indices.dtype.kind in "iu" and np.all((indices >= 0) & (indices < 22448)),
                "original_feature_row_indices_required")
        result = np.empty((len(indices), DIMENSIONS[family]), dtype=np.float32)
        covered = np.zeros(len(indices), dtype=bool)
        for shard in self.families[family]:
            self.check()
            positions = np.flatnonzero((indices >= shard["start"]) & (indices < shard["stop"]))
            if not len(positions):
                continue
            values = np.load(shard["path"], mmap_mode="r", allow_pickle=False)
            result[positions] = values[indices[positions] - shard["start"]]
            covered[positions] = True
            del values
        require(covered.all() and np.isfinite(result).all(), "finite_complete_core_features_required")
        return result


@dataclass
class CoreInputs:
    settings: dict
    recipe: dict
    people: tuple
    main_ids: tuple
    main_to_feature: np.ndarray
    routing: dict
    images: CoreFeatureRows
    eeg_recipe: object
    channel_order: tuple
    check: object
    root: Path
    records: dict

    def load_participant(self, person):
        require(person in self.people, "fixed_development_person_required")
        path = self.root / "events" / (person + ".npz")
        record = self.records[person]
        require(path.stat().st_size == record["bytes"] and record["main_images"] == 22248
                and record["physical_sequences"] == 72, "event_record_binding_differs")
        arrays = _npz(path, (*EVENT_FIELDS, "concept_by_block_image_row"), self.recipe["npz_byte_cap"], self.check)
        _event_arrays(arrays)
        sequences = load_prepared_participant(Path(self.settings["release"]["eeg_cache"]) / person,
            participant_id=person, channel_order=self.channel_order, recipe=self.eeg_recipe, budget_check=self.check)
        result = CoreParticipant(self, person, sequences, MappingProxyType(arrays))
        result.verify_sequences()
        return result


def bind_initialized(root, budget):
    root = _path(root)
    settings = _json(root / "settings.json", budget.record)
    recipe = validate_recipe(settings["recipe"])
    release, cap = settings["release"], recipe["metadata_byte_cap"]
    development, people = _release(release, cap)
    eeg, eeg_recipe, channels = _eeg_binding(release, people, cap)
    ids, families, evidence = _image_binding(release, cap, budget.check)
    routing, main = events_io.load_routing(release["private_routing"], ids)
    complete = _json(root / "inputs_complete.json", cap)
    index = {key: i for i, key in enumerate(ids)}
    require(settings["schema"] == SCHEMA and settings["recipe_id"] == digest(recipe)
            and settings["release_id"] == digest(release) and settings["development_id"] == digest(development)
            and settings["eeg_id"] == digest(eeg) and settings["participants"] == list(people)
            and settings["evidence"] == evidence and settings["families"] == families
            and settings["all_ids"] == list(ids) and settings["main_ids"] == main
            and settings["main_to_feature"] == [index[key] for key in main]
            and complete["status"] == "complete_core_reference_inputs"
            and complete["settings_id"] == digest(settings) and set(complete["participants"]) == set(people),
            "initialized_core_inputs_changed_or_incomplete")
    return CoreInputs(settings, recipe, people, tuple(main), readonly(np.asarray(settings["main_to_feature"], dtype=np.int64)),
        routing, CoreFeatureRows(families, budget.check), eeg_recipe, channels, budget.check, root, complete["participants"])


@dataclass
class CoreParticipant:
    inputs: CoreInputs
    participant_id: str
    sequences: tuple
    events: object

    def verify_sequences(self):
        require(len(self.sequences) == 72, "all72_physical_sequences_required")
        rows_by_sequence, seen = {}, set()
        lookup = {int(value): i for i, value in enumerate(self.events["sample_zero_based"])}
        for sequence in self.sequences:
            self.inputs.check()
            prepared = sequence.preparation_input
            require(isinstance(prepared, SequenceEventInputs) and prepared.participant_id == self.participant_id
                    and len(prepared.events) == 309 and sequence.sequence_id not in rows_by_sequence
                    and sequence.block == prepared.span.block and sequence.within_block == prepared.span.within_block,
                    "physical_sequence_event_binding_differs")
            rows = []
            for presentation, event in enumerate(prepared.events):
                sample, frozen = event.sample_zero_based, event.frozen_row
                require(sample in lookup, "event_sample_not_in_released_flags")
                row = lookup[sample]
                require(row not in seen and frozen["participant"] == self.participant_id
                        and self.inputs.routing[frozen["canonical_image"]] == self.inputs.main_ids[row]
                        and self.events["sample_one_based"][row] == int(frozen["sample"])
                        and sample == int(frozen["sample_zero_based"])
                        and self.events["concept"][row] == int(frozen["concept_number"])
                        and self.events["exemplar"][row] == int(frozen["exemplar_number"])
                        and self.events["block"][row] == sequence.block
                        and self.events["physical_sequence"][row] == prepared.span.physical_sequence
                        and self.events["within_block_sequence"][row] == sequence.within_block
                        and self.events["presentation"][row] == presentation == int(frozen["presentation"]),
                        "original_UUID_clock_concept_block_flag_join_differs")
                rows.append(row)
                seen.add(row)
            require(np.array_equal(prepared.source_sample_indices, self.events["sample_zero_based"][rows])
                    and np.array_equal(prepared.onsets_seconds,
                        (self.events["sample_zero_based"][rows] - prepared.span.start_sample) / 1000.),
                    "exact_unrounded_event_clock_required")
            rows_by_sequence[sequence.sequence_id] = readonly(np.asarray(rows, dtype=np.int64))
        require(len(seen) == 22248, "complete_main_event_coverage_required")
        return rows_by_sequence

    def prepare_training(self, training, *, split, budget_check):
        require(split == folds._split(split.outer_index, split.inner_index), "original_nested_split_required")
        expected = tuple(item.sequence_id for item in self.sequences if item.block in split.training_blocks)
        require(tuple(item.sequence_id for item in training) == expected
                and len(training) == len(split.training_blocks) * 6, "complete_original_training_roster_required")
        rows = self.verify_sequences()
        selected = np.sort(np.concatenate([rows[item.sequence_id] for item in training]))
        require(len(np.unique(selected)) == len(selected) and all(np.array_equal(
            np.sort(self.events["concept"][selected[self.events["block"][selected] == block]]), np.arange(1854))
            for block in split.training_blocks), "all1854_unique_training_concepts_per_block_required")
        total_rows = sum(len(item.sample_times_seconds) for item in self.sequences)
        # Conservative explicit arrays only; includes retained EEG, targets,
        # shared nuisance/kernels, family-at-a-time PCA and transforms. Native
        # LAPACK workspace and Python metadata remain under the process RSS cap.
        bound = 8 * (total_rows * (8*62 + 3*88 + 2*64 + 16)
            + 22248 * (3*320 + 201*12) + 12*len(selected)*2048 + 12*2048**2)
        require(bound <= self.inputs.recipe["working_array_byte_cap"], "core_working_array_admission")
        ids = tuple(self.inputs.main_ids[row] for row in selected)
        transforms, transformed = [], []
        for family in CORE_IMAGE:
            budget_check()
            values = self.inputs.images.rows(family, self.inputs.main_to_feature[selected])
            fitted = fit_feature_transform(values, training_unit_ids=ids, requested_rank=64)
            del values
            budget_check()
            values = self.inputs.images.rows(family, self.inputs.main_to_feature)
            transformed.append(fitted.apply(values))
            transforms.append(fitted)
            del values
        panel = FeaturePanelTransform("core_image", CORE_IMAGE, tuple(transforms))
        features = readonly(np.concatenate(transformed, axis=1))
        metadata = MappingProxyType({item.sequence_id: (item.preparation_input.span, item.sample_times_seconds)
                                     for item in self.sequences})
        builder = CoreDesignBuilder(self.participant_id, self.events, metadata, panel, features, rows, budget_check)
        return tuning.TuningPreparation(expected, builder)


@dataclass
class CoreDesignBuilder:
    participant_id: str
    events: object
    predictor_metadata: object
    image_transform: object
    image_features: np.ndarray
    rows_by_sequence: dict
    check: object

    def __call__(self, sequence):
        require(isinstance(sequence, folds.SequencePredictors) and sequence.sequence_id in self.rows_by_sequence,
                "predictor_only_sequence_required")
        self.check()
        rows = self.rows_by_sequence[sequence.sequence_id]
        prepared = sequence.preparation_input
        span, times = self.predictor_metadata[sequence.sequence_id]
        require(isinstance(prepared, SequenceEventInputs) and prepared.participant_id == self.participant_id
                and prepared.span == span and sequence.block == int(self.events["block"][rows[0]])
                and sequence.within_block == int(self.events["within_block_sequence"][rows[0]])
                and np.array_equal(sequence.sample_times_seconds, times)
                and np.array_equal(prepared.source_sample_indices, self.events["sample_zero_based"][rows]),
                "frozen_core_predictor_binding_differs")
        basis = TemporalBasis("raised_cosine_12")
        ticks = np.rint(times * 1000).astype(np.int64)
        require(np.all(np.abs(ticks / 1000. - times) < 1e-10), "acquisition_grid_required")
        onsets = self.events["sample_zero_based"][rows] - span.start_sample
        kernels = _kernels(ticks, onsets, basis)
        boundaries = _kernels(ticks, (0, int(onsets[-1]) + 50), basis)
        x = 2 * times / ((span.stop_sample - span.start_sample) / 1000.) - 1
        drift = np.column_stack((np.ones(len(x)), x, .5*(3*x*x-1), .5*(5*x*x*x-3*x)))
        position = 2*np.arange(309)/308. - 1
        positions = np.column_stack((np.ones(309), position, .5*(3*position*position-1)))
        flags = np.column_stack((self.events["target_flag"][rows], self.events["response_flag"][rows]))
        reduced = readonly(np.column_stack((drift, _expanded(kernels, positions, len(x), 12),
                                            _expanded(boundaries, np.eye(2), len(x), 12))))
        nuisance = readonly(np.column_stack((reduced, _expanded(kernels, flags, len(x), 12))))
        reference = SequenceDesign(sequence.sequence_id, times, nuisance, {})
        term = _PreparedTerm(readonly(self.image_features[rows]), kernels, 12, slice(0, 0))
        return {"nuisance": reference, "reduced_task_nuisance": _compose(reference, reduced, {}, basis),
                "image": _compose(reference, nuisance, {"image": term}, basis)}


def _prepare_core(sequences, split, sensor, recipe, prepare_training, limits, check, stage):
    """Original fold algebra with only the three registered reference designs."""
    training = tuple(item for item in sequences if item.block in split.training_blocks)
    heldout = tuple(item for item in sequences if item.block in split.heldout_blocks)
    stage("training_image_PCA")
    preparation = prepare_training(training, split=split, budget_check=check)
    require(isinstance(preparation, tuning.TuningPreparation)
            and tuple(preparation.training_sequence_ids) == tuple(item.sequence_id for item in training),
            "split_bound_training_preparation_required")
    stage("core_design_construction")
    train = folds._designs(training, preparation, CORE_MODELS, check)
    test = folds._designs(heldout, preparation, CORE_MODELS, check)
    require(all(train[name].sequences[0].schema == test[name].sequences[0].schema for name in CORE_MODELS)
            and train["nuisance"].shape[1] == 88 and train["reduced_task_nuisance"].shape[1] == 64,
            "registered_core_columns_required")
    penalties = {name: tuning._penalty((recipe["intercept_penalty"],)
        + (recipe["nuisance_penalty"],)*(train[name].shape[1]-1), train[name].shape[1])
        for name in CORE_MODELS[:2]}
    weight = recipe["fit_weight_multiplier"] / len(training)
    weights = readonly(np.concatenate([item.retained_mask*(weight/np.count_nonzero(item.retained_mask))
                                       for item in training]))
    targets = np.concatenate([sensor.apply(item.sensor_samples, channel_order=item.channel_order) for item in training])
    heldout_targets = tuple(sensor.apply(item.sensor_samples, channel_order=item.channel_order) for item in heldout)
    stage("training_nuisance_fit")
    fit = folds._fit(train["nuisance"], targets, weights, penalties["nuisance"], limits, check)
    stage("training_residual_whitening")
    residual = (targets - train["nuisance"] @ fit.coefficients)[weights > 0]
    covariance_weights = weights[weights > 0]
    centered = residual - covariance_weights @ residual / covariance_weights.sum()
    trace = float(np.sum(covariance_weights[:, None]*centered*centered)/covariance_weights.sum())
    del centered
    whitener = fit_residual_whitener(residual, sample_weights=covariance_weights,
        center_covariance=True, covariance_denominator=float(covariance_weights.sum()),
        shrinkage_target=np.eye(61)*(trace/61.), shrinkage_strength=recipe["shrinkage_strength"])
    cache = tuning._SplitCache(split, training, heldout, preparation, train, test, targets, heldout_targets,
        weights, MappingProxyType(penalties), fit, np.empty(0), whitener)
    stage("nuisance_heldout_scoring")
    cache.nuisance_losses = tuning._score(cache, "nuisance", fit, check)
    tuning.normalize_losses(cache.nuisance_losses, cache.nuisance_losses, normalization=recipe["normalization"])
    return cache


def fit_person(person, recipe, *, check, retain_inner, retain_outer, progress):
    recipe = validate_recipe(recipe)
    sensor, limits = CommonSensorSpace(person.inputs.channel_order), folds.SolverLimits(**recipe["solver"])
    sequences = folds._roster(person.sequences, sensor, check)
    convention = SimpleNamespace(normalization=recipe["normalization"])
    outer = []
    for outer_index in range(6):
        inner = []
        progress.update(outer_index=outer_index)
        for inner_index in range(6):
            if inner_index == outer_index:
                continue
            progress.update(inner_index=inner_index)
            cache = _prepare_core(sequences, folds._split(outer_index, inner_index), sensor, recipe,
                person.prepare_training, limits, check, lambda value: progress.update(stage=value))
            progress["stage"] = "complete13_image_inner_grid"
            result = controller._core_inner(cache, convention, limits, check)
            retain_inner(result)
            inner.append(result)
            progress["inner_splits_completed"] += 1
            del cache
        mean = np.mean([item.normalized_image_losses for item in inner], axis=0)
        choice = min(range(13), key=lambda index: (mean[index], -tuning.RIDGE_GRID[index], index))
        progress["inner_index"] = None
        cache = _prepare_core(sequences, folds._split(outer_index), sensor, recipe, person.prepare_training,
            limits, check, lambda value: progress.update(stage=value))
        progress["stage"] = "selected_outer_image_and_reduced_fits"
        fitted = controller._core_outer(cache, choice, convention, sensor, limits, check)
        retain_outer(fitted)
        outer.append({"outer_index": outer_index, "image_index": choice,
                      "normalized_losses": dict(fitted.result.normalized_losses)})
        progress["outer_splits_completed"] += 1
        del cache, fitted, inner, result
    return {"participant_id": person.participant_id, "outer": outer,
        "normalized_participant_losses": {name: float(np.mean([row["normalized_losses"][name] for row in outer]))
                                          for name in CORE_MODELS}}


def _save_outer(directory, fitted, budget):
    builder = fitted.preparation.build_designs
    require(isinstance(builder, CoreDesignBuilder), "actual_core_training_builder_required")
    parameters = {"image_transform": builder.image_transform, "whitener": fitted.whitener,
        "sensor_basis": fitted.sensor_space.basis, "channel_order": fitted.sensor_space.channel_order,
        "training_sequence_ids": fitted.preparation.training_sequence_ids,
        "penalties": fitted.penalties, "fits": fitted.fits, "result": fitted.result}
    _split_record(directory / ("outer_%02d.json" % fitted.result.split.outer_index), parameters, budget)


def _operational_finish(directory, report, budget):
    """Reserved small termination record, even after the soft deadline/RSS stop.

    This performs only bounded serialization; it never resumes scientific work.
    Native SIGKILL cannot guarantee this record; missing completion still refuses.
    """
    class Reserve:
        record = 65536
        written = 0
        def admit_write(self, amount):
            require(amount <= 65536, "operational_reserve_exceeded")
    _publish_json(directory / "operational.json", report, Reserve())


def run_participant(root, person, *, cost_only, budget):
    budget.require_compute_threads()
    inputs = bind_initialized(root, budget)
    require(person in inputs.people and (not cost_only or person == inputs.people[0]),
            "cost_person_must_be_first_frozen_development_identity")
    directory = inputs.root / person
    directory.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    progress = {"stage": "loading_released_person", "inner_splits_completed": 0,
        "outer_splits_completed": 0, "budget_checks": 0, "outer_index": None, "inner_index": None}
    report = {"status": "incomplete_core_reference_person", "participant_id": person, "cost_only": bool(cost_only),
        "settings_id": digest(inputs.settings), "scientific_release_authorized": False, "audit_access": False}
    require(budget.durable - budget.written >= 65536, "operational_failure_reserve_required")
    budget.durable -= 65536
    def check():
        progress["budget_checks"] += 1
        budget.check()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise TimeoutError("allocation_signal_received")
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        inputs.check = check
        inputs.images.check = check
        _publish_json(directory / "settings.json", report, budget)
        bound = inputs.load_participant(person)
        result = fit_person(bound, inputs.recipe, check=check,
            retain_inner=lambda row: _split_record(directory /
                ("inner_%02d_%02d.json" % (row.split.outer_index, row.split.inner_index)), row, budget),
            retain_outer=lambda row: _save_outer(directory, row, budget), progress=progress)
        require(progress["inner_splits_completed"] == 30 and progress["outer_splits_completed"] == 6,
                "all36_original_splits_required")
        _publish_json(directory / "complete.json", {**report, **result, "status": "complete_core_reference_person",
            "inner_splits": 30, "outer_splits": 6, "models": CORE_MODELS,
            "concept_or_false_scores_computed": False, "search_credit": 0}, budget)
        report["status"] = "complete_core_reference_person"
    except Exception as error:
        report["error_class"] = type(error).__name__
        frames, frame = [], error.__traceback__
        while frame is not None and len(frames) < 24:
            frames.append({"file": Path(frame.tb_frame.f_code.co_filename).name,
                           "function": frame.tb_frame.f_code.co_name, "line": frame.tb_lineno})
            frame = frame.tb_next
        report["frames"] = frames
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report.update(progress=progress, elapsed_seconds=time.monotonic()-started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _operational_finish(directory, report, budget)
    return report


def finalize(root, *, budget):
    budget.require_compute_threads()
    root = _path(root)
    settings = _json(root / "settings.json", budget.record)
    complete = _json(root / "inputs_complete.json", budget.record)
    require(settings["schema"] == SCHEMA and complete["settings_id"] == digest(settings)
            and complete["status"] == "complete_core_reference_inputs", "original_complete_inputs_required")
    people = tuple(settings["participants"])
    require(len(people) == len(set(people)) == 30 and people == tuple(sorted(people)), "complete_fixed_cohort_required")
    rows = []
    for person in people:
        budget.check()
        row = _json(root / person / "complete.json", budget.record)
        operation = _json(root / person / "operational.json", budget.record)
        require(row["status"] == operation["status"] == "complete_core_reference_person"
                and row["settings_id"] == operation["settings_id"] == digest(settings)
                and row["participant_id"] == person and row["inner_splits"] == 30 and row["outer_splits"] == 6
                and row["models"] == list(CORE_MODELS) and row["scientific_release_authorized"] is False
                and row["concept_or_false_scores_computed"] is False and row["search_credit"] == 0,
                "all30_complete_identical_reference_people_required")
        require([item["outer_index"] for item in row["outer"]] == list(range(6)), "all_six_outer_pairs_required")
        losses = row["normalized_participant_losses"]
        require(set(losses) == set(CORE_MODELS) and all(type(value) in (int, float)
            and np.isfinite(value) and value >= 0 for value in losses.values()), "finite_complete_reference_losses_required")
        for name in CORE_MODELS:
            require(losses[name] == float(np.mean([item["normalized_losses"][name] for item in row["outer"]])),
                    "retained_outer_and_person_mean_differ")
        rows.append(losses)
    gain = float(np.mean([row["nuisance"]-row["image"] for row in rows]))
    task = float(np.mean([row["reduced_task_nuisance"]-row["nuisance"] for row in rows]))
    record = {"status": "complete_30_person_core_calibration", "schema": SCHEMA,
        "settings_id": digest(settings), "recipe_id": settings["recipe_id"], "participants": people,
        "participant_count": 30, "inner_splits": 900, "outer_splits": 180,
        "participant_normalized_losses": dict(zip(people, rows)), "G": gain, "T": task,
        "delta_C": max(.001, .05*gain), "delta_P": max(.0005, .025*gain),
        "delta_Q": max(.002, .25*gain), "delta_T": max(.001, .25*task),
        "image_gain_positive": gain > 0, "reference_underidentified": gain <= 0,
        "search_credit": 0, "concept_or_false_scores_computed": False,
        "scientific_release_authorized": False, "audit_access": False}
    _publish_json(root / "core_reference.complete.json", record, budget)
    return {"status": record["status"], "participant_count": 30, "inner_splits": 900, "outer_splits": 180}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("initialize", "participant", "finalize"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--release")
    parser.add_argument("--recipe")
    parser.add_argument("--participant-id")
    parser.add_argument("--cost-only", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--rss-limit-bytes", type=int, required=True)
    parser.add_argument("--durable-byte-limit", type=int, required=True)
    parser.add_argument("--record-byte-limit", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        budget = RunBudget(deadline_epoch=args.deadline_epoch, rss_limit_bytes=args.rss_limit_bytes,
            durable_byte_limit=args.durable_byte_limit, record_byte_limit=args.record_byte_limit, threads=args.threads)
        if args.phase == "initialize":
            require(args.release and args.recipe and args.participant_id is None and not args.cost_only,
                    "explicit_initialization_inputs_required")
            report = initialize(args.release, args.recipe, args.run_dir, budget=budget)
        elif args.phase == "participant":
            require(args.participant_id and args.release is None and args.recipe is None, "frozen_person_invocation_required")
            report = run_participant(args.run_dir, args.participant_id, cost_only=args.cost_only, budget=budget)
        else:
            require(args.participant_id is None and args.release is None and args.recipe is None and not args.cost_only,
                    "cohort_finalization_has_no_new_inputs")
            report = finalize(args.run_dir, budget=budget)
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        return 0 if not report["status"].startswith("incomplete") else 1
    except Exception as error:
        print(json.dumps({"status": "core_reference_refused", "error_class": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
