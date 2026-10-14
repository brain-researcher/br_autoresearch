"""Unused first-generation continuation of original EP18 core split checkpoints.

Only complete original _split_record JSON/NPZ pairs are reused. An orphan NPZ
or partial file has no completion authority; a final JSON without its exact
archive is inconsistent and refused. Saved outer models stay in the explicit
predecessor. Only their checked scalar results are reused; callbacks receive
new fits only. This module never writes a canonical completion or rewrites a
predecessor. Its caller owns distinct output paths, resources and the original
deadline. There is no chained-resume format, discovery, retry or clock reset.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import zipfile

import numpy as np

import continuous_fold_runner as folds
import continuous_tuning as tuning
import core_reference_paired_inner as paired
import production_baseline_controller as controller
import production_core_reference as core
from training_transforms import CommonSensorSpace, CORE_IMAGE


def _positive(value):
    core.require(type(value) is int and value > 0, "explicit_positive_checkpoint_cap_required")
    return value


def _record(path, metadata_cap, array_cap, check):
    """Decode only the original serializer's numeric descriptors, without pickle."""
    check()
    meta = core._json(path, metadata_cap)
    archive = path.with_suffix(".npz")
    core.require(set(meta) == {"record", "array_file", "array_bytes"}
        and meta["array_file"] == archive.name and type(meta["array_bytes"]) is int
        and 0 < meta["array_bytes"] <= array_cap and archive.is_file()
        and archive.stat().st_size == meta["array_bytes"], "complete_exact_split_archive_required")
    descriptors = {}
    def inspect(value):
        if isinstance(value, dict):
            if "array" in value:
                core.require(set(value) == {"array","dtype","shape","tuple"}
                    and isinstance(value["array"], str) and value["array"] not in descriptors
                    and isinstance(value["shape"], list) and all(type(n) is int and n >= 0 for n in value["shape"])
                    and type(value["tuple"]) is bool, "original_array_descriptor_required")
                dtype = np.dtype(value["dtype"])
                core.require(dtype.kind in "biuf" and not dtype.hasobject, "numeric_checkpoint_dtype_required")
                descriptors[value["array"]] = value
            else:
                for child in value.values():
                    inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        else:
            core.require(value is None or type(value) in (str, int, bool)
                         or (type(value) is float and math.isfinite(value)), "finite_original_scalar_required")
    inspect(meta["record"])
    core.require(sum(math.prod(value["shape"])*np.dtype(value["dtype"]).itemsize
                     for value in descriptors.values()) <= array_cap, "checkpoint_array_cap_exceeded")
    # Original np.savez is uncompressed; inspect its finite member sizes before
    # loading, so a small compressed replacement cannot bypass the array cap.
    with zipfile.ZipFile(archive) as packed:
        members = packed.infolist()
        core.require(len(members) == len(descriptors)
            and {item.filename for item in members} == {name+".npy" for name in descriptors}
            and all(item.compress_type == zipfile.ZIP_STORED for item in members)
            and sum(item.file_size for item in members) <= array_cap, "original_bounded_npz_members_required")
    with np.load(archive, allow_pickle=False) as arrays:
        core.require(len(arrays.files) == len(descriptors) and set(arrays.files) == set(descriptors),
                     "checkpoint_array_names_differ")
        def decode(value):
            if isinstance(value, dict) and "array" in value:
                check()
                array = arrays[value["array"]]
                core.require(array.dtype.str == value["dtype"] and list(array.shape) == value["shape"]
                             and array.dtype.kind in "biuf" and np.isfinite(array).all(),
                             "checkpoint_array_dtype_shape_or_finiteness_differs")
                if value["tuple"]:
                    core.require(array.ndim == 1 and len(array) >= 13, "original_numeric_tuple_required")
                    return tuple(array.tolist())
                array.setflags(write=False)
                return array
            if isinstance(value, dict):
                return {key:decode(child) for key,child in value.items()}
            if isinstance(value, list):
                return [decode(child) for child in value]
            return value
        record = decode(meta["record"])
    return record, {"json_path": str(path), "array_path": str(archive), "array_bytes": meta["array_bytes"]}


def _split(value, expected):
    core.require(value == {"outer_index":expected.outer_index, "inner_index":expected.inner_index,
        "training_blocks":list(expected.training_blocks), "heldout_blocks":list(expected.heldout_blocks)},
        "saved_original_split_differs")


def _ids(sequences, split):
    return (tuple(row.sequence_id for row in sequences if row.block in split.training_blocks),
            tuple(row.sequence_id for row in sequences if row.block in split.heldout_blocks))


def _inner(record, split, sequences, normalization):
    core.require(set(record) == {"split","training_sequence_ids","heldout_sequence_ids",
        "image_sequence_losses","nuisance_sequence_losses","normalized_image_losses"},
        "complete_original_inner_record_required")
    _split(record["split"], split)
    training, heldout = _ids(sequences, split)
    core.require(tuple(record["training_sequence_ids"]) == training
        and tuple(record["heldout_sequence_ids"]) == heldout, "saved_sequence_order_differs")
    image, nuisance, normalized = (record[key] for key in
        ("image_sequence_losses","nuisance_sequence_losses","normalized_image_losses"))
    core.require(isinstance(image,np.ndarray) and image.shape == (13,12)
        and isinstance(nuisance,np.ndarray) and nuisance.shape == (12,)
        and isinstance(normalized,np.ndarray) and normalized.shape == (13,), "complete_original_inner_grid_required")
    core.require(np.array_equal(normalized, tuning.normalize_losses(image,nuisance,normalization=normalization)),
                 "saved_inner_normalization_differs")
    return controller.CoreInnerResult(split, training, heldout, image, nuisance, normalized)


def _choice(inner, outer_index):
    rows = [inner[(outer_index,j)].normalized_image_losses for j in range(6) if j != outer_index]
    mean = np.mean(rows,axis=0)
    return min(range(13),key=lambda index:(mean[index],-tuning.RIDGE_GRID[index],index))


def _outer(record, outer_index, inner, sequences, sensor, recipe):
    fields = {"image_transform","whitener","sensor_basis","channel_order","training_sequence_ids",
              "penalties","fits","result"}
    core.require(set(record) == fields, "complete_original_outer_parameters_required")
    split = folds._split(outer_index)
    training, heldout = _ids(sequences,split)
    result = record["result"]
    core.require(set(result) == {"split","image_index","errors","normalized_losses"},
                 "complete_original_outer_result_required")
    _split(result["split"],split)
    index = result["image_index"]
    core.require(type(index) is int and index == _choice(inner,outer_index), "saved_outer_choice_differs_from_original_inner_selection")
    core.require(tuple(record["training_sequence_ids"]) == training
        and tuple(record["channel_order"]) == sensor.channel_order
        and np.array_equal(record["sensor_basis"], sensor.basis), "saved_training_sensor_binding_differs")
    panel = record["image_transform"]
    core.require(panel["panel"] == "core_image" and tuple(panel["family_order"]) == CORE_IMAGE
                 and len(panel["families"]) == 5, "complete_original_five_family_PCA_required")
    ranks = []
    for transform in panel["families"]:
        loading = transform["loadings"]
        core.require(isinstance(loading,np.ndarray) and loading.ndim == 2
            and transform["requested_rank"] == 64 and transform["effective_rank"] == loading.shape[0]
            and 0 <= loading.shape[0] <= 64 and np.shape(transform["mean"]) == (loading.shape[1],)
            and np.shape(transform["population_sd"]) == (loading.shape[1],), "saved_core_PCA_dimensions_differ")
        ranks.append(loading.shape[0])
    widths = {"nuisance":88,"reduced_task_nuisance":64,"image":88+12*sum(ranks)}
    core.require(set(record["fits"]) == set(record["penalties"]) == set(core.CORE_MODELS),
                 "all_three_original_models_required")
    for name,width in widths.items():
        fit, penalty = record["fits"][name], record["penalties"][name]
        expected = np.full(width, recipe["nuisance_penalty"])
        expected[0] = recipe["intercept_penalty"]
        if name == "image":
            expected[88:] = tuning.RIDGE_GRID[index]
        core.require(set(fit) == {"coefficients","stop_codes","iterations"}
            and np.shape(fit["coefficients"]) == (width,61)
            and np.shape(penalty) == (width,) and np.array_equal(penalty,expected)
            and len(fit["stop_codes"]) == len(fit["iterations"]) == 61
            and all(type(value) is int and value in (0,1,2,4,5) for value in fit["stop_codes"])
            and all(type(value) is int and value >= 0 for value in fit["iterations"]),
            "saved_coefficient_penalty_or_solver_dimensions_differ")
    whitener = record["whitener"]
    core.require(np.shape(whitener["residual_mean_for_covariance"]) == (61,)
        and all(np.shape(whitener[key]) == (61,61) for key in
            ("empirical_covariance","shrinkage_target","regularized_covariance","inverse_root"))
        and whitener["covariance_centered"] is True
        and whitener["shrinkage_strength"] == recipe["shrinkage_strength"], "saved_whitener_dimensions_differ")
    errors = result["errors"]
    _split(errors["split"],split)
    expected_counts = tuple(int(row.retained_mask.sum()) for row in sequences if row.block in split.heldout_blocks)
    core.require(tuple(errors["training_sequence_ids"]) == training
        and tuple(errors["heldout_sequence_ids"]) == heldout
        and tuple(errors["retained_sample_counts"]) == expected_counts
        and set(errors["whitened_mse"]) == set(result["normalized_losses"]) == set(core.CORE_MODELS),
        "saved_outer_error_roster_differs")
    losses = {}
    for name in core.CORE_MODELS:
        raw = errors["whitened_mse"][name]
        core.require(np.shape(raw) == (12,), "all_twelve_outer_sequence_losses_required")
        value = float(tuning.normalize_losses(raw,errors["whitened_mse"]["nuisance"],
                                             normalization=recipe["normalization"]))
        core.require(type(result["normalized_losses"][name]) in (int,float)
                     and result["normalized_losses"][name] == value, "saved_outer_normalization_differs")
        losses[name] = value
    return {"outer_index":outer_index,"image_index":index,"normalized_losses":losses}


def bind_predecessor(predecessor_directory, *, original_root, participant_id,
                     initialized_settings, recipe, initialized_settings_id, metadata_byte_cap):
    """Bind only original initialization/person metadata, before any EEG loading."""
    metadata_cap = _positive(metadata_byte_cap)
    predecessor = core._path(predecessor_directory)
    core.require(predecessor == core._path(original_root)/participant_id,
                 "explicit_original_root_person_predecessor_required")
    settings = core._json(predecessor.parent/"settings.json",metadata_cap)
    complete = core._json(predecessor.parent/"inputs_complete.json",metadata_cap)
    person_record = core._json(predecessor/"settings.json",metadata_cap)
    core.require(settings["schema"] == core.SCHEMA
        and core.digest(settings) == initialized_settings_id == core.digest(initialized_settings)
        and settings["recipe_id"] == core.digest(recipe) and settings["recipe"] == recipe
        and complete["status"] == "complete_core_reference_inputs"
        and complete["settings_id"] == initialized_settings_id
        and participant_id in settings["participants"] and participant_id in complete["participants"]
        and person_record["settings_id"] == initialized_settings_id
        and person_record["participant_id"] == participant_id
        and person_record["scientific_release_authorized"] is False and person_record["audit_access"] is False,
        "original_initialized_settings_person_recipe_binding_differs")
    return predecessor


def load_completed_splits(person, recipe, *, predecessor_directory, initialized_settings_id,
                          metadata_byte_cap, array_byte_cap, sequences, sensor, check):
    """Read fixed split names from one explicitly selected original participant."""
    metadata_cap, array_cap = _positive(metadata_byte_cap), _positive(array_byte_cap)
    check()
    predecessor = bind_predecessor(predecessor_directory, original_root=person.inputs.root,
        participant_id=person.participant_id, initialized_settings=person.inputs.settings, recipe=recipe,
        initialized_settings_id=initialized_settings_id, metadata_byte_cap=metadata_cap)
    inner, outer, inner_refs, outer_refs = {}, {}, [], []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            check()
            path = predecessor/("inner_%02d_%02d.json" % (i,j))
            if not path.exists():
                continue
            record, ref = _record(path,metadata_cap,array_cap,check)
            inner[(i,j)] = _inner(record,folds._split(i,j),sequences,recipe["normalization"])
            inner_refs.append(dict(ref,outer_index=i,inner_index=j))
            del record
    for i in range(6):
        check()
        path = predecessor/("outer_%02d.json" % i)
        if not path.exists():
            continue
        core.require(all((i,j) in inner for j in range(6) if j != i),
                     "completed_outer_requires_all_five_predecessor_inner_records")
        record, ref = _record(path,metadata_cap,array_cap,check)
        outer[i] = _outer(record,i,inner,sequences,sensor,recipe)
        outer_refs.append(dict(ref,outer_index=i))
        del record
    return inner, outer, inner_refs, outer_refs


def fit_person(person, recipe, *, predecessor_directory, initialized_settings_id,
               metadata_byte_cap, array_byte_cap, check, retain_inner, retain_outer, progress):
    """Continue every missing original split; retain callbacks receive NEW work only."""
    recipe = core.validate_recipe(recipe)
    sensor = CommonSensorSpace(person.inputs.channel_order)
    limits = folds.SolverLimits(**recipe["solver"])
    sequences = folds._roster(person.sequences,sensor,check)
    convention = SimpleNamespace(normalization=recipe["normalization"])
    inner, outer, inner_refs, outer_refs = load_completed_splits(person,recipe,
        predecessor_directory=predecessor_directory,initialized_settings_id=initialized_settings_id,
        metadata_byte_cap=metadata_byte_cap,array_byte_cap=array_byte_cap,sequences=sequences,sensor=sensor,check=check)
    progress.update(reused_inner_records=inner_refs,reused_outer_records=outer_refs,
        reused_inner_splits_completed=len(inner),reused_outer_splits_completed=len(outer),
        new_inner_splits_completed=0,new_outer_splits_completed=0,
        inner_splits_completed=len(inner),outer_splits_completed=len(outer))
    actual_new = 0
    for i in range(6):
        for j in range(i+1,6):
            missing = [key for key in ((i,j),(j,i)) if key not in inner]
            if not missing:
                continue
            a,b = missing[0]
            progress.update(outer_index=a,inner_index=b)
            cache = core._prepare_core(sequences,folds._split(a,b),sensor,recipe,person.prepare_training,
                limits,check,lambda value:progress.update(stage=value))
            progress["stage"] = "complete13_image_inner_grid"
            if len(missing) == 2:
                second = paired._counterpart(cache,sequences,sensor,convention,check)
                rows = paired._inner_pair(cache,second,convention,limits,check)
                del second
            else:
                rows = (controller._core_inner(cache,convention,limits,check),)
            del cache
            actual_new += 14
            for row in rows:
                retain_inner(row)
                inner[(row.split.outer_index,row.split.inner_index)] = row
                progress["new_inner_splits_completed"] += 1
                progress["inner_splits_completed"] += 1
            del row, rows
    for i in range(6):
        if i in outer:
            continue
        progress.update(outer_index=i,inner_index=None)
        choice = _choice(inner,i)
        cache = core._prepare_core(sequences,folds._split(i),sensor,recipe,person.prepare_training,
            limits,check,lambda value:progress.update(stage=value))
        progress["stage"] = "selected_outer_image_and_reduced_fits"
        fitted = controller._core_outer(cache,choice,convention,sensor,limits,check)
        retain_outer(fitted)
        outer[i] = {"outer_index":i,"image_index":choice,"normalized_losses":dict(fitted.result.normalized_losses)}
        actual_new += 3
        progress["new_outer_splits_completed"] += 1
        progress["outer_splits_completed"] += 1
        del cache, fitted
    core.require(len(inner) == 30 and len(outer) == 6,"complete_original_logical_panel_required")
    ordered = [outer[i] for i in range(6)]
    progress.update(actual_new_multivariate_fits=actual_new,original_logical_multivariate_fits=438)
    return {"participant_id":person.participant_id,"outer":ordered,
        "normalized_participant_losses":{name:float(np.mean([row["normalized_losses"][name] for row in ordered]))
                                          for name in core.CORE_MODELS}}
