"""One private, complete EP17 NestedPredictionResult checkpoint.

Only a finished four-model development result is supported. Loading lets later
stages reuse it without refitting; it cannot resume an interrupted nested fit.
Independent frozen block order, calibration supports/operators/32 banks, real
response/feature sources and enforced reader separation remain caller inputs.
This container is not an audit lock, provenance authenticator or access grant.
Selection objectives and learned arrays are private outcome-bearing artifacts.

The caller supplies an admitted existing owned0700 directory and a positive
serialized-byte cap; no directory, storage default or resource grant is made.
One0600 ZIP_STORED container holds compact JSON and numeric NPY1.0 members.
Writing uses one exclusive sibling .partial, preserves it on failure, and
publishes by an atomic no-overwrite hard link only after close/flush/fsync.
Only that writer's partial is removed after publication. Partial names are not
loadable. There is no callback after publication that can falsely report a
successfully published checkpoint as a resource failure.

Arrays are written without a full-result copy; only strided PCA loading views
are copied into the container's required C order. Reading validates each numeric
header against this format's expected shape/dtype before allocating its byte
payload, then retains that immutable bytes buffer directly. No pickle loader,
compression, hash chain, payload reader, fitting or new qualification runs.
Caller-owned result arrays must remain unchanged for the duration of saving.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
import zipfile

import numpy as np

import endpoint_evaluator as endpoint
import nested_prediction as nested
import prediction_core as prediction

FORMAT = "ep17-complete-nested-prediction"
VERSION = 1


class CheckpointError(ValueError):
    """The complete checkpoint is refused; no partial result is returned."""


def require(condition, code):
    if not condition:
        raise CheckpointError(code)


@dataclass(frozen=True)
class CheckpointEvent:
    operation: str
    when: str
    member: str | None = None
    byte_count: int = 0
    shape: tuple = ()


@dataclass(frozen=True)
class CheckpointWriteResult:
    """Publication succeeded; cleanup status never invalidates published_path.

    A false cleanup flag means only the writer's extra partial link remains.
    Use the committed checkpoint; do not retry expensive fitting or publication.
    """
    published_path: Path
    partial_cleanup_complete: bool


def _event(check, operation, when="before", *, member=None, byte_count=0, shape=()):
    check(CheckpointEvent(operation, when, member, int(byte_count), tuple(shape)))


def _integer(value, name, minimum=0):
    require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
            and value >= minimum, name + "_invalid")
    return int(value)


def _real(value, name, positive=False):
    require(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)) and (not positive or value > 0), name + "_invalid")
    return float(value)


def _keys(value, expected, name):
    require(hasattr(value, "keys") and set(value) == set(expected), name + "_incomplete")


def _private_path(value):
    path = Path(value)
    require(not path.name.endswith(".partial"), "partial_checkpoint_not_loadable_or_publishable")
    require(not path.parent.is_symlink(), "owned_0700_checkpoint_parent_required")
    parent = path.parent.stat()
    require(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.geteuid()
            and stat.S_IMODE(parent.st_mode) == 0o700, "owned_0700_checkpoint_parent_required")
    return path


def _selection(value):
    require(isinstance(value, nested.Selection), "complete_selection_required")
    _keys(value.objectives, nested.SETTINGS, "all_84_selection_objectives")
    objectives = [_real(value.objectives[setting], "selection_objective") for setting in nested.SETTINGS]
    winner = min(range(len(nested.SETTINGS)), key=lambda index: (
        -objectives[index], prediction.hyperparameter_tie_key(nested.SETTINGS[index].layer,
                    nested.SETTINGS[index].rank, nested.SETTINGS[index].penalty)))
    require(value.setting == nested.SETTINGS[winner], "selected_setting_not_exact_registered_winner")
    return {"selected": winner, "objectives": objectives}


def _setting(value):
    require(value in nested.SETTINGS, "registered_setting_required")
    return [value.layer, int(value.rank), float(value.penalty)]


def _describe(result, check, maximum):
    require(isinstance(result, nested.NestedPredictionResult)
            and isinstance(result.design, nested.DevelopmentDesign), "complete_nested_prediction_result_required")
    design = result.design
    arrays = {}
    array_bytes = 0

    def array(name, value, shape, integer=False, *, pca_loadings=False):
        nonlocal array_bytes
        dtype = np.dtype(np.int64 if integer else np.float64)
        size = math.prod(shape) * dtype.itemsize
        require(array_bytes + size <= maximum, "serialized_byte_cap_exceeded")
        _event(check, "validate_checkpoint", member=name, byte_count=size, shape=shape)
        require(isinstance(value, np.ndarray) and value.dtype == dtype and value.shape == tuple(shape)
                and (value.flags.c_contiguous or pca_loadings),
                "checkpoint_array_shape_or_dtype_changed")
        require(np.isfinite(value).all(), "checkpoint_array_nonfinite")
        require(name not in arrays, "checkpoint_array_duplicate")
        # derive_rank_pca retains the maximum-rank fit's strided column view.
        # Canonicalize only these small matrices, after byte/resource admission;
        # prediction and response banks retain the no-copy C-order requirement.
        arrays[name] = np.ascontiguousarray(value) if pca_loadings else value
        array_bytes += size

    def pca(value, prefix, layer, rank, training_images):
        width = nested.FEATURE_WIDTHS[layer]
        require(isinstance(value, prediction.PCAFit) and value.rank == rank
                and _integer(value.training_images, "PCA_training_images", 1) == training_images,
                "PCA_training_or_rank_binding_changed")
        numerical_rank = _integer(value.numerical_rank, "PCA_numerical_rank", rank)
        require(numerical_rank <= min(width, training_images), "PCA_numerical_rank_invalid")
        array(prefix + "/feature_mean.npy", value.feature_mean, (width,))
        array(prefix + "/loadings.npy", value.loadings, (width, rank), pca_loadings=True)
        array(prefix + "/singular_values.npy", value.singular_values, (min(width, training_images),))
        return {"score_scale": _real(value.score_scale, "PCA_score_scale", positive=True),
                "numerical_rank": numerical_rank, "training_images": training_images}

    _keys(result.columns, endpoint.GROUPS, "column_groups")
    widths = []
    for group_index, group in enumerate(endpoint.GROUPS):
        columns = result.columns[group]
        require(isinstance(columns, np.ndarray) and columns.ndim == 1 and len(columns) > 0,
                "original_columns_required")
        array(f"columns/g{group_index}.npy", columns, (len(columns),), integer=True)
        require((columns >= 0).all() and (columns < endpoint.support.PARTICIPANT_COLUMNS[group[0]]).all()
                and len(np.unique(columns)) == len(columns), "original_columns_invalid")
        widths.append(len(columns))
    _keys(result.predictions, nested.MODEL_IDS, "all_four_prediction_models")
    for model_index, model in enumerate(nested.MODEL_IDS):
        _keys(result.predictions[model], endpoint.GROUPS, "prediction_groups")
        for group_index, group in enumerate(endpoint.GROUPS):
            _keys(result.predictions[model][group], endpoint.STAGES, "prediction_stages")
            for stage in endpoint.STAGES:
                array(f"predictions/m{model_index}/g{group_index}/{stage}.npy",
                      result.predictions[model][group][stage], (nested.DEVELOPMENT_IMAGES, widths[group_index]))
    _keys(result.baselines, endpoint.GROUPS, "shared_baseline_groups")
    for group_index, group in enumerate(endpoint.GROUPS):
        _keys(result.baselines[group], endpoint.STAGES, "shared_baseline_stages")
        for stage in endpoint.STAGES:
            array(f"baselines/g{group_index}/{stage}.npy", result.baselines[group][stage],
                  (nested.DEVELOPMENT_IMAGES, widths[group_index]))
    _keys(result.outer_selections, nested.MODEL_IDS, "outer_selection_models")
    _keys(result.outer_cka_transforms, nested.MODEL_IDS, "outer_CKA_models")
    outer = []
    for model_index, model in enumerate(nested.MODEL_IDS):
        _keys(result.outer_selections[model], design.fold_labels, "all_six_outer_selections")
        _keys(result.outer_cka_transforms[model], design.fold_labels, "all_six_outer_CKA_transforms")
        entries = []
        for fold_index, fold in enumerate(design.fold_labels):
            selected = result.outer_selections[model][fold]
            selection = _selection(selected)
            n_train = sum(label != fold for label in design.fold_ids)
            fitted = pca(result.outer_cka_transforms[model][fold], f"outer/m{model_index}/f{fold_index}",
                         selected.setting.layer, 64, n_train)
            entries.append({"selection": selection, "cka_pca": fitted})
        outer.append(entries)
    _keys(result.final_predictors, nested.MODEL_IDS, "all_four_final_predictors")
    final, target_means = [], {}
    for model_index, model in enumerate(nested.MODEL_IDS):
        fit = result.final_predictors[model]
        require(isinstance(fit, nested.FrozenPredictor), "frozen_final_predictor_required")
        require(tuple(fit.training_image_ids) == design.image_ids
                and tuple(fit.training_concept_ids) == design.concept_ids, "final_training_roster_changed")
        selection = _selection(fit.selection)
        require(fit.setting == fit.selection.setting, "final_selected_setting_changed")
        setting = _setting(fit.setting)
        prefix = f"final/m{model_index}"
        encoding = pca(fit.pca, prefix + "/pca", fit.setting.layer, fit.setting.rank, nested.DEVELOPMENT_IMAGES)
        cka = pca(fit.cka_pca, prefix + "/cka_pca", fit.setting.layer, 64, nested.DEVELOPMENT_IMAGES)
        shared_rank = min(fit.setting.rank, 64)
        require(np.array_equal(fit.pca.feature_mean, fit.cka_pca.feature_mean)
                and np.array_equal(fit.pca.loadings[:, :shared_rank], fit.cka_pca.loadings[:, :shared_rank]),
                "final_encoding_CKA_basis_changed")
        _keys(fit.ridges, endpoint.GROUPS, "final_ridge_groups")
        ridges = []
        for group_index, group in enumerate(endpoint.GROUPS):
            _keys(fit.ridges[group], endpoint.STAGES, "final_ridge_stages")
            entries = []
            for stage in endpoint.STAGES:
                ridge = fit.ridges[group][stage]
                require(isinstance(ridge, prediction.RidgeFit)
                        and _integer(ridge.training_images, "ridge_training_images", 1) == nested.DEVELOPMENT_IMAGES
                        and _real(ridge.penalty, "ridge_penalty", positive=True) == fit.setting.penalty,
                        "final_ridge_training_or_penalty_changed")
                where = prefix + f"/ridge/g{group_index}/{stage}"
                array(where + "/score_mean.npy", ridge.score_mean, (fit.setting.rank,))
                array(where + "/target_mean.npy", ridge.target_mean, (widths[group_index],))
                array(where + "/coefficients.npy", ridge.coefficients, (fit.setting.rank, widths[group_index]))
                key = (group, stage)
                if key in target_means:
                    require(np.array_equal(target_means[key], ridge.target_mean), "model_specific_final_target_mean_changed")
                else:
                    target_means[key] = ridge.target_mean
                entries.append({"penalty": float(ridge.penalty), "training_images": int(ridge.training_images)})
            ridges.append(entries)
        final.append({"setting": setting, "selection": selection, "pca": encoding, "cka_pca": cka,
                      "ridges": ridges, "training_image_ids": list(fit.training_image_ids),
                      "training_concept_ids": list(fit.training_concept_ids)})
    row_metadata = {name: list(getattr(design, name)) for name in
                    ("image_ids", "concept_ids", "roles", "fold_ids", "block_ids")}
    for name in ("fold_ids", "block_ids"):
        row_metadata[name] = [value if isinstance(value, str) else int(value) for value in row_metadata[name]]
    metadata = {"format": FORMAT, "version": VERSION, "design": row_metadata,
                "models": list(nested.MODEL_IDS), "groups": [list(group) for group in endpoint.GROUPS],
                "settings": [_setting(value) for value in nested.SETTINGS], "column_widths": widths,
                "outer": outer, "final": final}
    return metadata, arrays


class _BoundedWriter:
    def __init__(self, stream, maximum, check):
        self.stream, self.maximum, self.check = stream, maximum, check

    def write(self, value):
        require(self.stream.tell() + len(value) <= self.maximum, "serialized_byte_cap_exceeded")
        _event(self.check, "write_bytes", byte_count=len(value))
        return self.stream.write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def save_prediction_checkpoint(result, destination, *, max_bytes, resource_check):
    """Publish a new complete result; return its committed path/cleanup status.

    Existing destinations are never replaced. Once published_path is returned,
    it is the completed checkpoint regardless of partial_cleanup_complete.
    """
    require(callable(resource_check), "external_resource_callback_required")
    maximum = _integer(max_bytes, "max_bytes", 1)
    _event(resource_check, "preflight")
    path = _private_path(destination)
    require(not path.exists() and not path.is_symlink(), "checkpoint_destination_exists")
    metadata, arrays = _describe(result, resource_check, maximum)
    _event(resource_check, "encode_metadata")
    metadata_bytes = json.dumps(metadata, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # This is a lower bound, not an invented envelope; the writer enforces the
    # actual cap including NPY headers, ZIP records and metadata during I/O.
    require(sum(value.nbytes for value in arrays.values()) + len(metadata_bytes) <= maximum,
            "serialized_byte_cap_exceeded")
    _event(resource_check, "create_partial")
    descriptor, partial_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    partial = Path(partial_name)
    # Deliberately no failure cleanup: the single owned partial is retained
    # for explicit diagnosis, never promoted/read as a completed checkpoint.
    with os.fdopen(descriptor, "w+b") as stream:
        with zipfile.ZipFile(_BoundedWriter(stream, maximum, resource_check), "w",
                             compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, value in arrays.items():
                _event(resource_check, "write_member", member=name, byte_count=value.nbytes, shape=value.shape)
                with archive.open(name, "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, value, version=(1, 0), allow_pickle=False)
            _event(resource_check, "write_metadata", member="metadata.json", byte_count=len(metadata_bytes))
            archive.writestr("metadata.json", metadata_bytes)
        _event(resource_check, "flush_checkpoint")
        stream.flush()
        os.fsync(stream.fileno())
    _event(resource_check, "publish_checkpoint")
    os.link(partial, path)  # Atomic EEXIST refusal, including concurrent writers.
    try:
        partial.unlink()
    except OSError:
        return CheckpointWriteResult(path, False)
    return CheckpointWriteResult(path, True)


def _decode(metadata, take):
    require(isinstance(metadata, dict) and metadata.get("format") == FORMAT
            and metadata.get("version") == VERSION, "checkpoint_format_or_version_changed")
    require(metadata.get("models") == list(nested.MODEL_IDS)
            and metadata.get("groups") == [list(group) for group in endpoint.GROUPS]
            and metadata.get("settings") == [_setting(value) for value in nested.SETTINGS],
            "checkpoint_registered_roster_changed")
    design = nested.DevelopmentDesign(**metadata["design"])
    widths = metadata["column_widths"]
    require(isinstance(widths, list) and len(widths) == len(endpoint.GROUPS), "checkpoint_column_widths_incomplete")
    widths = [_integer(value, "column_width", 1) for value in widths]
    require(all(width <= endpoint.support.PARTICIPANT_COLUMNS[group[0]]
                for width, group in zip(widths, endpoint.GROUPS)), "checkpoint_column_width_invalid")

    def selection(value):
        require(isinstance(value, dict) and set(value) == {"selected", "objectives"}, "selection_metadata_invalid")
        chosen = _integer(value["selected"], "selected_index")
        objectives = value["objectives"]
        require(chosen < len(nested.SETTINGS) and isinstance(objectives, list)
                and len(objectives) == len(nested.SETTINGS), "all_84_selection_objectives_incomplete")
        return nested.Selection(nested.SETTINGS[chosen], MappingProxyType({setting: _real(score, "selection_objective")
            for setting, score in zip(nested.SETTINGS, objectives)}))

    def pca(value, prefix, layer, rank, n_train):
        width = nested.FEATURE_WIDTHS[layer]
        require(isinstance(value, dict) and set(value) == {"score_scale", "numerical_rank", "training_images"},
                "PCA_metadata_invalid")
        return prediction.PCAFit(take(prefix + "/feature_mean.npy", (width,)),
            take(prefix + "/loadings.npy", (width, rank)), _real(value["score_scale"], "PCA_score_scale", positive=True),
            take(prefix + "/singular_values.npy", (min(width, n_train),)),
            _integer(value["numerical_rank"], "PCA_numerical_rank", rank),
            _integer(value["training_images"], "PCA_training_images", 1))

    columns = {group: take(f"columns/g{index}.npy", (widths[index],), True)
               for index, group in enumerate(endpoint.GROUPS)}
    predictions = {model: {group: {stage: take(f"predictions/m{mi}/g{gi}/{stage}.npy",
        (nested.DEVELOPMENT_IMAGES, widths[gi])) for stage in endpoint.STAGES}
        for gi, group in enumerate(endpoint.GROUPS)} for mi, model in enumerate(nested.MODEL_IDS)}
    baselines = {group: {stage: take(f"baselines/g{gi}/{stage}.npy", (nested.DEVELOPMENT_IMAGES, widths[gi]))
                          for stage in endpoint.STAGES} for gi, group in enumerate(endpoint.GROUPS)}
    require(isinstance(metadata["outer"], list) and len(metadata["outer"]) == 4
            and isinstance(metadata["final"], list) and len(metadata["final"]) == 4,
            "checkpoint_all_model_fits_required")
    outer_selections, outer_cka, final = {}, {}, {}
    for mi, model in enumerate(nested.MODEL_IDS):
        entries = metadata["outer"][mi]
        require(isinstance(entries, list) and len(entries) == 6, "checkpoint_all_outer_folds_required")
        outer_selections[model], outer_cka[model] = {}, {}
        for fi, fold in enumerate(design.fold_labels):
            entry = entries[fi]
            require(isinstance(entry, dict) and set(entry) == {"selection", "cka_pca"}, "outer_fit_metadata_invalid")
            selected = selection(entry["selection"])
            outer_selections[model][fold] = selected
            outer_cka[model][fold] = pca(entry["cka_pca"], f"outer/m{mi}/f{fi}", selected.setting.layer,
                                        64, sum(label != fold for label in design.fold_ids))
        entry = metadata["final"][mi]
        selected = selection(entry["selection"])
        require(entry["setting"] == _setting(selected.setting), "final_selected_setting_changed")
        setting, prefix = selected.setting, f"final/m{mi}"
        encoding = pca(entry["pca"], prefix + "/pca", setting.layer, setting.rank, nested.DEVELOPMENT_IMAGES)
        cka = pca(entry["cka_pca"], prefix + "/cka_pca", setting.layer, 64, nested.DEVELOPMENT_IMAGES)
        require(isinstance(entry["ridges"], list) and len(entry["ridges"]) == 8, "final_ridge_groups_incomplete")
        ridges = {}
        for gi, group in enumerate(endpoint.GROUPS):
            entries = entry["ridges"][gi]
            require(isinstance(entries, list) and len(entries) == 3, "final_ridge_stages_incomplete")
            ridges[group] = {}
            for stage, meta in zip(endpoint.STAGES, entries):
                require(isinstance(meta, dict) and set(meta) == {"penalty", "training_images"}, "ridge_metadata_invalid")
                where = prefix + f"/ridge/g{gi}/{stage}"
                ridges[group][stage] = prediction.RidgeFit(take(where + "/score_mean.npy", (setting.rank,)),
                    take(where + "/target_mean.npy", (widths[gi],)),
                    take(where + "/coefficients.npy", (setting.rank, widths[gi])),
                    _real(meta["penalty"], "ridge_penalty", positive=True),
                    _integer(meta["training_images"], "ridge_training_images", 1))
        final[model] = nested.FrozenPredictor(setting, encoding, cka, nested._nested_map(ridges), selected,
            tuple(entry["training_image_ids"]), tuple(entry["training_concept_ids"]))
    return nested.NestedPredictionResult(design, MappingProxyType(columns), nested._nested_map(predictions),
        nested._nested_map(baselines), nested._nested_map(outer_selections), nested._nested_map(outer_cka),
        MappingProxyType(final))


def load_prediction_checkpoint(source, *, max_bytes, resource_check):
    """Load exactly one complete private container; never read a partial name."""
    require(callable(resource_check), "external_resource_callback_required")
    maximum = _integer(max_bytes, "max_bytes", 1)
    _event(resource_check, "preflight")
    path = _private_path(source)
    _event(resource_check, "open_checkpoint")
    require(not path.is_symlink(), "ordinary_checkpoint_file_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                and stat.S_IMODE(info.st_mode) == 0o600, "owned_0600_checkpoint_file_required")
        require(info.st_size <= maximum, "serialized_byte_cap_exceeded")
        _event(resource_check, "read_directory", byte_count=info.st_size)
        with zipfile.ZipFile(stream, "r") as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            require(len(names) == len(set(names)) and "metadata.json" in names, "checkpoint_members_missing_or_duplicate")
            require(all(entry.compress_type == zipfile.ZIP_STORED and entry.file_size <= maximum for entry in entries),
                    "checkpoint_member_storage_or_size_invalid")
            by_name = {entry.filename: entry for entry in entries}
            _event(resource_check, "read_metadata", member="metadata.json", byte_count=by_name["metadata.json"].file_size)
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            consumed = {"metadata.json"}

            def take(name, shape, integer=False):
                require(name in by_name and name not in consumed, "checkpoint_member_missing_or_reused")
                dtype = np.dtype(np.int64 if integer else np.float64)
                size = math.prod(shape) * dtype.itemsize
                require(size <= maximum, "serialized_byte_cap_exceeded")
                _event(resource_check, "read_header", member=name, byte_count=by_name[name].file_size, shape=shape)
                with archive.open(name, "r") as member:
                    require(np.lib.format.read_magic(member) == (1, 0), "checkpoint_NPY_version_changed")
                    actual_shape, fortran, actual_dtype = np.lib.format.read_array_header_1_0(member)
                    require(actual_shape == tuple(shape) and not fortran and actual_dtype == dtype
                            and not actual_dtype.hasobject, "checkpoint_array_header_changed")
                    require(member.tell() + size == by_name[name].file_size, "checkpoint_array_payload_size_changed")
                    _event(resource_check, "read_member", member=name, byte_count=size, shape=shape)
                    data = member.read(size)
                    require(len(data) == size and member.read(1) == b"", "checkpoint_array_payload_incomplete")
                consumed.add(name)
                return np.frombuffer(data, dtype=dtype).reshape(shape)

            result = _decode(metadata, take)
            require(consumed == set(names), "checkpoint_extra_members")
            canonical, arrays = _describe(result, resource_check, maximum)
            require(canonical == metadata and set(arrays) | {"metadata.json"} == set(names),
                    "checkpoint_metadata_inconsistent")
    return result
