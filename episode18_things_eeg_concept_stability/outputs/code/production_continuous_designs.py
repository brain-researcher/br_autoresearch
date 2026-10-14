"""Released development artifacts -> complete EP18 training-only model designs.

This is an explicit-config callable, not a scientific selection or executable
launch. The coordinator selects/persists the recipe and resource admission.
``bind_released_inputs(release, recipe, budget_check=...)`` reads existing
producer formats, never discovers paths or opens original EEG/audit files.
``bound.load_participant(id)`` loads one complete existing EEG cache and its
prepared event/false-group arrays. ``person.prepare_training`` is the callback
for continuous_tuning.run_tuned_participant; ``make_tuning_recipe(recipe)`` and
``solver_limits(recipe)`` supply its remaining explicit numeric choices.

All 30 development identities, 72 physical sequences, 22,248 main image UUIDs,
1,854 concepts and all eight candidate partitions are retained. A candidate
partition is only the image/context search output: this module neither grants
the selected-design recheck nor declares a qualified false bank or scientific
release. The cohort/search controller, exact EDF implementation/admission,
retained-fit persistence, original qualification/controls and exposure rules
remain caller responsibilities. No fitting of EEG, scores, writes or retries.

The label reader reuses the name/human producer formats without calling
prepare_label_geometry.load_labels, which reads a mixed-role structural record.
Every training set independently proves it contains all 1,854 unique concepts.
Identical label-only transforms may then be reused by exact unit/recipe key;
image transforms always fit the unique images in this split's training blocks.
Shared immutable _PreparedTerm kernels avoid duplicating identical temporal
expansions across the 15 existing SequenceDesign operators. Acquisition-tick
differences define closed support and bin edges; there is no 250-Hz event snap.

Byte admission covers retained input arrays and explicit design arrays. Native
SVD/eigendecomposition workspace and the caller's EEG/EDF caches still require
the coordinator's actual process/Slurm bound and budget_check callback.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import zipfile

import numpy as np

from continuous_eeg_cache import load_prepared_participant, read_json
from continuous_fold_runner import (FIXED_PANEL_MODELS, SequencePredictors,
    SolverLimits, WhiteningChoices, _split)
from continuous_sequence_model import SequenceDesign, _PreparedTerm
from continuous_tuning import (BasePenaltyPlan, BaselineChoice, MatchedIndices,
    TuningPreparation, TuningRecipe)
from feature_extractors import require_compute
from false_partition_search import RECIPE as FALSE_PARTITION_RECIPE
from label_rbf import fit_label_rbf_basis
from prepare_continuous_eeg import SequenceEventInputs, SequencePreparationRecipe
from prepare_false_partition_inputs import (feature_binding, load_release,
    load_routing)
from run_development_eeg_preprocessing import COEFFICIENT_SHA, SPAN
from slot_capacity import RIDGE_GRID
from training_transforms import (DIMENSIONS, LABEL, PANELS, fit_feature_panel)


OUTPUTS = Path(__file__).resolve().parents[1]
N_IMAGES, N_CONCEPTS = 22248, 1854
EVENT_FIELDS = ("concept", "exemplar", "block", "physical_sequence",
    "within_block_sequence", "presentation", "sample_one_based",
    "sample_zero_based", "target_flag", "response_flag")
BASIS_WIDTHS = {"binned_FIR_20ms": 40, "raised_cosine_8": 8,
                "raised_cosine_12": 12, "raised_cosine_16": 16}
NUISANCE_RECIPE = "local_legendre_0_to_3_position_0_to_2_E3_final_offset_target_coarse_response_v1"
CAPACITY_TIE = "nearest_term_edf_then_larger_lambda_then_index"
BASELINE_TIE = "minimum_mean_loss_then_larger_lambda_then_index"
RECIPE_FIELDS = {"recipe_id", "image_panel", "image_ranks", "label_ranks",
    "temporal_basis", "label_kernel_bandwidth", "nuisance_recipe",
    "intercept_penalty", "nuisance_penalty", "shared_feature_penalty",
    "fit_weight_scale", "fit_weight_multiplier", "normalization",
    "covariance_centered", "covariance_denominator", "covariance_target",
    "shrinkage_strength", "capacity_tie_break", "baseline_tie_break",
    "base_rank_relative_tolerance", "rbf_numerics", "solver",
    "metadata_byte_cap", "npz_byte_cap", "working_array_byte_cap",
    "dense_design_byte_limit"}


def require(value, code):
    if not value:
        raise ValueError(code)  # Categorical only: no source values or outcome text.


def _require_candidate_context_recipe(config):
    """Bind candidate context to the adopted physical span and FIR support."""
    require(config["recipe_id"] == FALSE_PARTITION_RECIPE
            and config["recorded_span_recipe"] == asdict(SPAN)
            and type(config["filter_left_support_samples"]) is int
            and config["filter_left_support_samples"] == 128
            and type(config["filter_right_support_samples"]) is int
            and config["filter_right_support_samples"] == 0,
            "candidate_context_differs_from_adopted_EEG_recipe")


def readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "explicit_absolute_path_required")
    return path


def episode_path(value):
    path = absolute(value)
    require(OUTPUTS.resolve() in path.resolve().parents, "episode_output_path_required")
    return path


def validate_recipe(value):
    """Require every choice; a recipe file is not release/adoption authority."""
    recipe = json.loads(json.dumps(value, allow_nan=False))
    require(set(recipe) == RECIPE_FIELDS and isinstance(recipe["recipe_id"], str)
            and bool(recipe["recipe_id"]), "complete_explicit_recipe_required")
    require(recipe["image_panel"] in ("core_image", "expanded_image")
            and recipe["temporal_basis"] in BASIS_WIDTHS
            and recipe["nuisance_recipe"] == NUISANCE_RECIPE, "registered_design_recipe_required")
    for key, families in (("image_ranks", PANELS[recipe["image_panel"]]), ("label_ranks", LABEL)):
        require(set(recipe[key]) == set(families)
                and all(type(rank) is int and rank in (32, 64, 128) for rank in recipe[key].values()),
                "complete_registered_family_ranks_required")
    require(type(recipe["label_kernel_bandwidth"]) in (int, float)
            and recipe["label_kernel_bandwidth"] in (.5, 1., 2.), "registered_bandwidth_required")
    require(recipe["fit_weight_scale"] == "mean_sequences" and recipe["fit_weight_multiplier"] == 1.
            and recipe["normalization"] == "ratio_of_equal_sequence_means"
            and recipe["covariance_centered"] is True
            and recipe["covariance_denominator"] == "sum_weights"
            and recipe["covariance_target"] == "trace_over_61_identity"
            and recipe["capacity_tie_break"] == CAPACITY_TIE
            and recipe["baseline_tie_break"] == BASELINE_TIE, "explicit_callback_conventions_required")
    for name in ("intercept_penalty", "nuisance_penalty", "shared_feature_penalty",
                 "shrinkage_strength", "base_rank_relative_tolerance"):
        number = recipe[name]
        require(type(number) in (int, float) and np.isfinite(number) and number >= 0,
                "finite_nonnegative_recipe_value_required")
    require(recipe["intercept_penalty"] == 0. and recipe["nuisance_penalty"] in RIDGE_GRID
            and recipe["shared_feature_penalty"] in RIDGE_GRID
            and 0 < recipe["shrinkage_strength"] <= 1
            and recipe["base_rank_relative_tolerance"] < 1, "recipe_numeric_range_differs")
    require(set(recipe["rbf_numerics"]) == {"eigenvalue_rtol", "eigenvalue_atol",
            "reconstruction_rtol", "reconstruction_max_atol"}, "explicit_RBF_numerics_required")
    for value in recipe["rbf_numerics"].values():
        require(type(value) in (int, float) and np.isfinite(value) and value >= 0,
                "finite_RBF_numerics_required")
    require(recipe["rbf_numerics"]["eigenvalue_rtol"] < 1, "RBF_rank_tolerance_differs")
    require(set(recipe["solver"]) == {"atol", "btol", "conlim", "maxiter"}, "explicit_solver_limits_required")
    for key in ("atol", "btol", "conlim"):
        require(type(recipe["solver"][key]) in (int, float)
                and np.isfinite(recipe["solver"][key]) and recipe["solver"][key] > 0,
                "positive_solver_limit_required")
    require(recipe["solver"]["conlim"] > 1 and type(recipe["solver"]["maxiter"]) is int
            and recipe["solver"]["maxiter"] > 0, "finite_solver_limits_required")
    for name in ("metadata_byte_cap", "npz_byte_cap", "working_array_byte_cap", "dense_design_byte_limit"):
        require(type(recipe[name]) is int and recipe[name] > 0, "explicit_byte_cap_required")
    return recipe


def solver_limits(recipe):
    return SolverLimits(**validate_recipe(recipe)["solver"])


def _npz(path, keys, cap, check):
    require(path.stat().st_size <= cap, "input_NPZ_byte_cap")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(keys) and {item.filename for item in entries} == {key + ".npy" for key in keys}
                and all(item.compress_type == zipfile.ZIP_STORED for item in entries)
                and sum(item.file_size for item in entries) <= cap, "input_NPZ_layout_differs")
    with np.load(path, allow_pickle=False) as source:
        result = {}
        for name in keys:
            check()
            result[name] = readonly(source[name])
    return result


def _table(path, cap):
    require(path.stat().st_size <= cap, "metadata_table_byte_cap")
    with path.open(encoding="utf-8", newline="") as source:
        table = csv.DictReader(source, delimiter="\t")
        require(table.fieldnames and len(set(table.fieldnames)) == len(table.fieldnames), "metadata_columns_differ")
        rows = list(table)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            "metadata_row_width_differs")
    return table.fieldnames, rows


def _labels(names, human, registry, cap, check):
    """Read the authentic common labels, no mixed participant role inventory."""
    spec = read_json(registry, cap)
    named = read_json(names / "label_feature_report.json", cap)
    human_report = read_json(human / "human_metadata_report.json", cap)
    require(spec["record_id"] == "ep18_feature_recipe_20261003" and spec["label_panel"] == list(LABEL)
            and named["status"] == "all1854_label_text_features_completed"
            and named["completed_concepts"] == 1854 and named["feature_shape"] == [1854, 512]
            and human_report["status"] == "source_crosswalk_complete_no_missing_values"
            and human_report["frozen_main_concepts"] == 1854, "complete_label_artifacts_required")
    ids = tuple(row["uniqueID"] for row in read_json(names / "concept_ids_and_tokens.json", cap))
    require(len(ids) == len(set(ids)) == 1854 and all(isinstance(key, str) and key for key in ids),
            "unique_label_inventory_required")
    fields, crosswalk = _table(human / "concept_crosswalk.tsv", cap)
    require(fields == ["uniqueID", "frozen_concept_number", "provider_Word", "provider_WordContext"],
            "authentic_concept_crosswalk_required")
    concept_ids = [None] * 1854
    for row in crosswalk:
        number = int(row["frozen_concept_number"])
        require(0 <= number < 1854 and concept_ids[number] is None, "duplicate_or_invalid_concept_number")
        concept_ids[number] = row["uniqueID"]
    require(len(crosswalk) == len(set(concept_ids)) == 1854 and set(concept_ids) == set(ids),
            "concept_crosswalk_coverage_differs")
    fields, rows = _table(human / "human_features_untransformed.tsv", cap)
    properties = spec["features"][LABEL[1]]["selected_columns"]
    categories = [name for name in fields if name.startswith("category53::")]
    require(len(properties) == 12 and len(categories) == 53
            and categories == ["category53::" + name for name in human_report["category_columns"]],
            "registered_human_columns_required")
    by_id = {row["uniqueID"]: row for row in rows}
    require(len(rows) == len(by_id) == 1854 and set(by_id) == set(ids), "human_label_join_differs")
    check()
    name_path = names / "concept_name_clip_text.npy"
    require(name_path.stat().st_size <= 1854 * 512 * 4 + 65536, "name_feature_byte_cap")
    name_values = np.load(name_path, allow_pickle=False)
    require(name_values.shape == (1854, 512) and name_values.dtype == np.float32
            and np.isfinite(name_values).all(), "qualified_name_features_differ")
    order = {key: i for i, key in enumerate(ids)}
    values = np.asarray([[float(by_id[key][name]) for name in properties + categories]
                         for key in concept_ids], dtype=np.float64)
    require(np.isfinite(values).all() and np.isin(values[:, 12:], (0., 1.)).all(), "invalid_human_values_no_imputation")
    return tuple(concept_ids), MappingProxyType({LABEL[0]: readonly(name_values[[order[key] for key in concept_ids]]),
        LABEL[1]: readonly(values[:, :12]), LABEL[2]: readonly(values[:, 12:])})


def _event_arrays(values):
    require(set(values) == set(EVENT_FIELDS) | {"concept_by_block_image_row"}, "event_array_names_differ")
    require(all(values[name].shape == (N_IMAGES,) and values[name].dtype == np.int64 for name in EVENT_FIELDS),
            "event_array_shape_or_type_differs")
    ranges = {"concept": 1854, "exemplar": 12, "block": 12, "physical_sequence": 72,
              "within_block_sequence": 6, "presentation": 309, "target_flag": 2, "response_flag": 2}
    require(all(np.all((values[name] >= 0) & (values[name] < high)) for name, high in ranges.items()),
            "event_identity_or_binary_flag_differs")
    require(np.array_equal(values["sample_one_based"], values["sample_zero_based"] + 1)
            and np.all(values["sample_zero_based"] >= 0)
            and len(np.unique(values["sample_zero_based"])) == N_IMAGES, "event_clock_differs")
    groups = values["concept_by_block_image_row"]
    require(groups.shape == (1854, 12) and groups.dtype == np.int32
            and np.array_equal(np.sort(groups.ravel()), np.arange(N_IMAGES))
            and np.array_equal(values["concept"][groups], np.broadcast_to(np.arange(1854)[:, None], groups.shape))
            and np.array_equal(values["block"][groups], np.broadcast_to(np.arange(12), groups.shape)),
            "complete_concept_block_join_required")
    require(np.array_equal(np.sort(values["exemplar"][groups], axis=1), np.broadcast_to(np.arange(12), groups.shape)),
            "all_twelve_distinct_exemplars_required")


@dataclass
class ReleasedInputs:
    release: dict
    recipe: dict
    people: tuple[str, ...]
    main_ids: tuple[str, ...]
    main_to_feature: np.ndarray
    images: dict
    routing: dict
    concept_ids: tuple[str, ...]
    labels: MappingProxyType
    eeg_recipe: SequencePreparationRecipe
    channel_order: tuple[str, ...]
    event_records: dict
    check: object
    label_cache: dict

    def load_participant(self, person):
        """Exactly one full participant; no all-cohort EEG preload or fallback."""
        require_compute()
        require(person in self.people, "development_participant_required")
        check, cap = self.check, self.recipe["npz_byte_cap"]
        check()
        root = Path(self.release["partition_inputs"])
        record = self.event_records[person]
        path = root / (person + ".npz")
        require(Path(record["path"]) == path and path.stat().st_size == record["bytes"]
                and record["main_images"] == 22248 and record["blocks"] == 12
                and record["physical_sequences"] == 72, "prepared_participant_event_binding_differs")
        events = _npz(path, (*EVENT_FIELDS, "concept_by_block_image_row"), cap, check)
        _event_arrays(events)
        candidate = Path(self.release["false_candidate_dirs"][person])
        meta = read_json(candidate / "candidate_bank.json", self.recipe["metadata_byte_cap"])
        arrays = _npz(candidate / "candidate_bank.npz",
            ("primary", "random_capacity_diagnostic", "acceptance_diagnostics"), cap, check)
        primary = arrays["primary"]
        require(primary.shape == (8, 1854, 12) and primary.dtype == np.int32, "complete_primary_bank_required")
        groups = np.empty((8, N_IMAGES), dtype=np.int64)
        for index, assignment in enumerate(primary):
            check()
            require(np.array_equal(np.sort(assignment.ravel()), np.arange(N_IMAGES))
                    and np.array_equal(events["block"][assignment], np.broadcast_to(np.arange(12), assignment.shape))
                    and np.all(np.diff(np.sort(events["concept"][assignment], axis=1), axis=1) > 0),
                    "false_partition_membership_differs")
            require(hashlib.sha256(np.asarray(assignment, dtype="<i4").tobytes()).hexdigest()
                    == meta["primary_bank_fingerprints"][index], "false_candidate_fingerprint_differs")
            groups[index, assignment.ravel()] = np.repeat(np.arange(1854), 12)
        del arrays, primary
        sequences = load_prepared_participant(Path(self.release["eeg_cache"]) / person,
            participant_id=person, channel_order=self.channel_order, recipe=self.eeg_recipe, budget_check=check)
        participant = BoundParticipant(self, person, sequences, MappingProxyType(events), readonly(groups))
        participant.verify_sequences(sequences, expected_count=72)
        return participant


def bind_released_inputs(release, recipe, *, budget_check):
    """Actual existing-format reader; call only after coordinator admission."""
    require_compute()
    require(callable(budget_check), "bounded_compute_callback_required")
    recipe = validate_recipe(recipe)
    cap = recipe["metadata_byte_cap"]
    release = read_json(absolute(release), cap)
    fields = {"development_manifest", "eeg_cache", "partition_inputs", "false_candidate_dirs",
              "feature_bank", "private_routing", "name_features", "human_features", "feature_registry"}
    require(set(release) == fields and isinstance(release["false_candidate_dirs"], dict), "explicit_release_fields_required")
    for name in fields - {"false_candidate_dirs"}:
        episode_path(release[name])
    development = read_json(release["development_manifest"], cap)
    people = tuple(development["participants"])
    require(development["role"] == "development" and len(people) == len(set(people)) == 30
            and list(people) == sorted(people) and development["event_count"] == 739440
            and set(release["false_candidate_dirs"]) == set(people), "complete_development_role_required")
    eeg = read_json(Path(release["eeg_cache"]) / "complete.json", cap)
    require(eeg["status"] == "complete_development_EEG_preprocessing" and eeg["participant_count"] == 30
            and set(eeg["participants"]) == set(people) and eeg["physical_sequences"] == 2160
            and eeg["main_events"] == 667440 and eeg["source_role"] == "development_only"
            and eeg["sample_unit"] == "V" and eeg["projection_or_training_transform_applied"] is False
            and eeg["model_scores_computed"] is False and eeg["audit_access"] is False
            and eeg["span_recipe"] == asdict(SPAN), "complete_original_EEG_preparation_required")
    prepared_recipe = dict(eeg["preparation_recipe"])
    require(prepared_recipe.pop("coefficient_sha256") == COEFFICIENT_SHA, "frozen_FIR_recipe_required")
    eeg_recipe = SequencePreparationRecipe(**prepared_recipe)
    require(len(eeg_recipe.fir_coefficients) == 129 and eeg_recipe.fir_origin_sample == 0
            and eeg_recipe.decimation_phase_mod4 == 0 and eeg_recipe.detrend == "none"
            and eeg_recipe.source_sample_rate_hz == 1000 and eeg_recipe.target_sample_rate_hz == 250
            and eeg_recipe.input_sample_unit == "V" and eeg_recipe.operation_order == "detrend_then_fir_then_decimate"
            and hashlib.sha256(np.asarray(eeg_recipe.fir_coefficients, dtype="<f8").tobytes()).hexdigest() == COEFFICIENT_SHA,
            "original_preparation_conventions_required")
    channel_order = tuple(eeg["channel_order"])
    require(len(channel_order) == len(set(channel_order)) == 62, "frozen_62_channel_order_required")
    inputs = Path(release["partition_inputs"])
    report, settings = read_json(inputs / "report.json", cap), read_json(inputs / "settings.json", cap)
    require(report["status"] == "false_partition_inputs_complete" and set(report["participants"]) == set(people)
            and report["main_images"] == 22248 and report["feature_images"] == 22448
            and report["all_main_images_once_per_participant"] is True
            and report["all_original_UUIDs_preserved"] is True and report["audit_view_read"] is False
            and not (inputs / "failure.json").exists(), "complete_partition_inputs_required")
    original, original_people, _, _ = load_release(settings["release_manifest"])
    require(tuple(original_people) == people and all(Path(original[key]) == Path(release[key])
            for key in ("feature_bank", "private_routing", "development_manifest")), "source_release_binding_differs")
    ids, families = feature_binding(release["feature_bank"])
    index = read_json(inputs / "image_index.json", cap)
    routing, main_ids = load_routing(release["private_routing"], ids)
    require(index["all_feature_opaque_ids"] == ids and index["main_opaque_ids"] == main_ids
            and index["families"] == families, "prepared_image_identity_binding_differs")
    positions = {key: i for i, key in enumerate(ids)}
    mapping = np.asarray(index["main_to_feature_row"])
    require(mapping.dtype.kind in "iu" and np.array_equal(mapping, [positions[key] for key in main_ids]),
            "image_feature_row_order_differs")
    # Read every person's completion metadata before any neural/feature values.
    for person in people:
        budget_check()
        candidate = episode_path(release["false_candidate_dirs"][person])
        state, meta, selected = (read_json(candidate / name, cap)
                                for name in ("report.json", "candidate_bank.json", "settings.json"))
        config = selected["config"]
        _require_candidate_context_recipe(config)
        config_id = hashlib.sha256((json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
                                  + "\n").encode()).hexdigest()
        require(state["status"] == "complete_image_context_candidate_bank" and state["primary_members"] == 8
                and state["participant_id"] == person and state["full_false_bank_qualified"] is False
                and meta["participant_id"] == person and meta["main_opaque_ids"] == main_ids
                and len(meta["primary_bank_fingerprints"]) == 8 and meta["config_id"] == config_id
                and meta["full_false_bank_qualified"] is False and meta["selected_design_recheck_required"] is True
                and meta["regeneration_after_selected_design_recheck"] is False and meta["EEG_read"] is False
                and selected["phase"] == "search" and Path(config["inputs_dir"]) == inputs
                and config["participant_id"] == person, "complete_provisional_candidate_binding_required")
        require(Path(eeg["participants"][person]["directory"]) == Path(release["eeg_cache"]) / person,
                "EEG_participant_path_differs")
    concept_ids, labels = _labels(Path(release["name_features"]), Path(release["human_features"]),
                                   Path(release["feature_registry"]), cap, budget_check)
    require(sum(shape["feature_bytes"] for shape in families.values()) + sum(a.nbytes for a in labels.values())
            <= recipe["working_array_byte_cap"], "released_feature_array_admission")
    images = {name: np.load(info["path"], mmap_mode="r", allow_pickle=False) for name, info in families.items()}
    return ReleasedInputs(release, recipe, people, tuple(main_ids), readonly(mapping), images, routing,
                          concept_ids, labels, eeg_recipe, channel_order, report["participants"], budget_check, {})


@dataclass(frozen=True)
class TemporalBasis:
    name: str

    @property
    def width(self):
        return BASIS_WIDTHS[self.name]

    def __call__(self, lags):
        lags = np.asarray(lags, dtype=np.float64)
        require(lags.ndim == 1 and np.isfinite(lags).all(), "finite_lag_vector_required")
        inside = (lags >= 0.) & (lags <= .8)
        result = np.zeros((len(lags), self.width))
        if self.name == "binned_FIR_20ms":
            # The producer passes exact acquisition-tick differences. This
            # conversion prevents binary-float drift at integer 20-ms edges.
            ticks = np.rint(lags[inside] * 1000).astype(np.int64)
            require(np.all(np.abs(lags[inside] * 1000 - ticks) < 1e-7), "acquisition_tick_lags_required")
            result[np.flatnonzero(inside), np.minimum(ticks // 20, 39)] = 1.
        else:
            spacing = .8 / (self.width - 1)
            distance = (lags[:, None] - np.linspace(0., .8, self.width)) / spacing
            keep = (np.abs(distance) <= 1.) & inside[:, None]
            result[keep] = .5 * (1. + np.cos(np.pi * distance[keep]))
        return result


def _kernels(ticks, onsets, basis):
    values = []
    for onset in onsets:
        lo, hi = int(np.searchsorted(ticks, onset, "left")), int(np.searchsorted(ticks, onset + 800, "right"))
        values.append((lo, hi, readonly(basis((ticks[lo:hi] - onset) / 1000.))))
    return tuple(values)


def _expanded(kernels, amplitudes, rows, width):
    result = np.zeros((rows, amplitudes.shape[1] * width))
    for amplitude, (lo, hi, kernel) in zip(amplitudes, kernels):
        if hi > lo:
            result[lo:hi] += (amplitude[None, :, None] * kernel[:, None, :]).reshape(hi - lo, -1)
    return result


def _compose(reference, nuisance, terms, basis):
    """Same existing operator, sharing immutable kernels/base arrays exactly."""
    design = copy.copy(reference)
    design.nuisance, design.terms = nuisance, {}
    schema, cursor = [], nuisance.shape[1]
    for name, term in terms.items():
        width = term.features.shape[1] * basis.width
        design.terms[name] = replace(term, column_slice=slice(cursor, cursor + width))
        schema.append((name, term.features.shape[1], basis.width, (0., .8), basis))
        cursor += width
    design.schema, design.shape = (nuisance.shape[1], tuple(schema)), (len(reference.sample_times), cursor)
    return design


@dataclass
class BoundParticipant:
    inputs: ReleasedInputs
    participant_id: str
    sequences: tuple
    events: MappingProxyType
    false_groups: np.ndarray

    def verify_sequences(self, sequences, *, expected_count):
        require(len(sequences) == expected_count, "complete_sequence_subset_required")
        seen, ids, result = set(), set(), {}
        by_sample = {int(sample): i for i, sample in enumerate(self.events["sample_zero_based"])}
        for sequence in sequences:
            self.inputs.check()
            event_input = sequence.preparation_input
            require(isinstance(event_input, SequenceEventInputs) and event_input.participant_id == self.participant_id
                    and len(event_input.events) == 309 and sequence.sequence_id not in ids
                    and sequence.block == event_input.span.block and sequence.within_block == event_input.span.within_block,
                    "prepared_event_sequence_identity_differs")
            rows = []
            for presentation, event in enumerate(event_input.events):
                sample = event.sample_zero_based
                require(sample in by_sample, "prepared_event_sample_not_in_source_flags")
                row = by_sample[sample]
                frozen = event.frozen_row
                require(row not in seen and frozen["participant"] == self.participant_id
                        and self.inputs.routing[frozen["canonical_image"]] == self.inputs.main_ids[row]
                        and self.inputs.concept_ids[int(self.events["concept"][row])] == frozen["canonical_image"].split("/", 1)[0]
                        and self.events["sample_one_based"][row] == int(frozen["sample"])
                        and sample == int(frozen["sample_zero_based"])
                        and self.events["concept"][row] == int(frozen["concept_number"])
                        and self.events["exemplar"][row] == int(frozen["exemplar_number"])
                        and self.events["block"][row] == sequence.block
                        and self.events["physical_sequence"][row] == event_input.span.physical_sequence
                        and self.events["within_block_sequence"][row] == sequence.within_block
                        and self.events["presentation"][row] == presentation == int(frozen["presentation"]),
                        "EEG_flag_UUID_concept_block_join_differs")
                seen.add(row)
                rows.append(row)
            require(np.array_equal(event_input.source_sample_indices, self.events["sample_zero_based"][rows])
                    and np.array_equal(event_input.onsets_seconds,
                        (self.events["sample_zero_based"][rows] - event_input.span.start_sample) / 1000.),
                    "preserved_event_clock_differs")
            ids.add(sequence.sequence_id)
            result[sequence.sequence_id] = readonly(np.asarray(rows, dtype=np.int64))
        require(len(seen) == expected_count * 309, "complete_event_subset_required")
        return result

    def prepare_training(self, training, *, split, budget_check):
        require_compute()
        recipe, inputs = self.inputs.recipe, self.inputs
        require(split == _split(split.outer_index, split.inner_index), "original_nested_split_required")
        expected = tuple(item.sequence_id for item in self.sequences if item.block in split.training_blocks)
        require(tuple(item.sequence_id for item in training) == expected
                and len(training) == len(split.training_blocks) * 6, "exact_training_sequence_roster_required")
        rows_by_sequence = self.verify_sequences(self.sequences, expected_count=72)
        selected = np.concatenate([rows_by_sequence[item.sequence_id] for item in training])
        selected = np.sort(selected)
        require(len(np.unique(selected)) == len(selected)
                and set(self.events["block"][selected]) == set(split.training_blocks), "unique_training_images_required")
        concept_numbers = np.unique(self.events["concept"][selected])
        require(np.array_equal(concept_numbers, np.arange(1854)) and all(
            np.array_equal(np.sort(self.events["concept"][selected[self.events["block"][selected] == block]]), np.arange(1854))
            for block in split.training_blocks), "all_concepts_in_each_training_block_required")
        order = PANELS[recipe["image_panel"]]
        image_ids = tuple(inputs.main_ids[i] for i in selected)
        # Explicit array admission; native full-SVD workspace is caller-owned.
        peak_bound = 8 * (6 * len(selected) * max(DIMENSIONS[name] for name in order)
                          + N_IMAGES * sum(recipe["image_ranks"].values()) + 12 * 1854**2)
        width = BASIS_WIDTHS[recipe["temporal_basis"]]
        # Shared kernels still retain at most 201 samples for each 0..800-ms
        # event; all 72 designs and nine categorical slots are budgeted here.
        design_bound = 8 * (N_IMAGES * (10 * 1854 + sum(recipe["image_ranks"].values())
                            + sum(recipe["label_ranks"].values()) + 201 * width)
            + sum(len(item.sample_times_seconds) for item in self.sequences) * (3 * (4 + 7 * width) + 8)
            + 72 * 65536)
        source_bytes = sum(value.nbytes for value in inputs.images.values()) + sum(value.nbytes for value in inputs.labels.values())
        require(source_bytes + max(peak_bound, design_bound) <= recipe["working_array_byte_cap"],
                "training_transform_and_design_array_admission")
        budget_check()
        image_rows = {name: inputs.images[name][inputs.main_to_feature[selected]] for name in order}
        panel = fit_feature_panel(image_rows, training_unit_ids={name: image_ids for name in order},
            requested_ranks=recipe["image_ranks"], panel=recipe["image_panel"])
        del image_rows
        budget_check()
        transformed_images = panel.apply({name: inputs.images[name][inputs.main_to_feature] for name in order},
            row_ids={name: inputs.main_ids for name in order})
        key = (tuple(recipe["label_ranks"][name] for name in LABEL), recipe["label_kernel_bandwidth"],
               tuple(sorted(recipe["rbf_numerics"].items())))
        if key not in inputs.label_cache:
            label_panel = fit_feature_panel(inputs.labels,
                training_unit_ids={name: inputs.concept_ids for name in LABEL},
                requested_ranks=recipe["label_ranks"], panel="label")
            labels = label_panel.apply(inputs.labels, row_ids={name: inputs.concept_ids for name in LABEL})
            budget_check()
            kernel = fit_label_rbf_basis(labels, training_unit_ids=inputs.concept_ids,
                label_panel_transform=label_panel, bandwidth_multiplier=recipe["label_kernel_bandwidth"],
                **recipe["rbf_numerics"])
            inputs.label_cache[key] = (label_panel, readonly(labels), kernel)
        label_panel, transformed_labels, rbf = inputs.label_cache[key]
        budget_check()
        # The builder retains predictors only, never self.sequences/held-out
        # sensor samples. Its frozen metadata is sufficient for exact joins.
        predictor_metadata = MappingProxyType({item.sequence_id:
            (item.preparation_input.span, item.sample_times_seconds) for item in self.sequences})
        builder = TrainingDesignBuilder(self.participant_id, self.events, self.false_groups,
            predictor_metadata, split, recipe, panel, label_panel, rbf,
            readonly(transformed_images), transformed_labels, rows_by_sequence, TemporalBasis(recipe["temporal_basis"]),
            budget_check)
        return TuningPreparation(expected, builder)


@dataclass
class TrainingDesignBuilder:
    participant_id: str
    events: MappingProxyType
    false_groups: np.ndarray
    predictor_metadata: MappingProxyType
    split: object
    recipe: dict
    image_transform: object
    label_transform: object
    label_kernel: object
    image_features: np.ndarray
    label_features: np.ndarray
    rows_by_sequence: dict
    basis: TemporalBasis
    check: object

    def __call__(self, sequence):
        require(isinstance(sequence, SequencePredictors) and sequence.sequence_id in self.rows_by_sequence,
                "predictor_only_bound_sequence_required")
        self.check()
        rows = self.rows_by_sequence[sequence.sequence_id]
        event_input = sequence.preparation_input
        span, original_times = self.predictor_metadata[sequence.sequence_id]
        require(isinstance(event_input, SequenceEventInputs)
                and event_input.participant_id == self.participant_id and event_input.span == span
                and sequence.block == int(self.events["block"][rows[0]])
                and sequence.within_block == int(self.events["within_block_sequence"][rows[0]])
                and np.array_equal(sequence.sample_times_seconds, original_times)
                and np.array_equal(event_input.source_sample_indices, self.events["sample_zero_based"][rows]),
                "frozen_sequence_predictor_binding_differs")
        ticks = np.rint(sequence.sample_times_seconds * 1000).astype(np.int64)
        require(np.all(np.abs(ticks / 1000. - sequence.sample_times_seconds) < 1e-10), "original_acquisition_grid_required")
        onsets = self.events["sample_zero_based"][rows] - event_input.span.start_sample
        kernels = _kernels(ticks, onsets, self.basis)
        boundaries = _kernels(ticks, (0, int(onsets[-1]) + 50), self.basis)
        length = (event_input.span.stop_sample - event_input.span.start_sample) / 1000.
        x = 2 * sequence.sample_times_seconds / length - 1
        drift = np.column_stack((np.ones(len(x)), x, .5 * (3*x*x-1), .5 * (5*x*x*x-3*x)))
        position = 2 * np.arange(309) / 308. - 1
        positions = np.column_stack((np.ones(309), position, .5 * (3*position*position-1)))
        flags = np.column_stack((self.events["target_flag"][rows], self.events["response_flag"][rows]))
        reduced = readonly(np.column_stack((drift, _expanded(kernels, positions, len(x), self.basis.width),
                           _expanded(boundaries, np.eye(2), len(x), self.basis.width))))
        nuisance = readonly(np.column_stack((reduced, _expanded(kernels, flags, len(x), self.basis.width))))
        reference = SequenceDesign(sequence.sequence_id, sequence.sample_times_seconds, nuisance, {})
        numbers = self.events["concept"][rows]
        features = {"image": readonly(self.image_features[rows]), "label": readonly(self.label_features[numbers]),
                    "label_kernel": readonly(self.label_kernel.training_features[numbers])}
        for name, assignment in [("true_concept", numbers)] + [
                (f"false_group_{i+1}", self.false_groups[i, rows]) for i in range(8)]:
            values = np.zeros((309, 1854))
            values[np.arange(309), assignment] = 1.
            values.setflags(write=False)
            features[name] = values
        terms = {name: _PreparedTerm(values, kernels, self.basis.width, slice(0, 0)) for name, values in features.items()}
        image = {"image": terms["image"]}
        label = {**image, "label": terms["label"]}
        supplied = {"nuisance": reference,
            "reduced_task_nuisance": _compose(reference, reduced, {}, self.basis),
            "image": _compose(reference, nuisance, image, self.basis),
            "label_linear": _compose(reference, nuisance, label, self.basis),
            "label_capacity_matched": _compose(reference, nuisance, {**label, "label_kernel": terms["label_kernel"]}, self.basis),
            "true_concept": _compose(reference, nuisance, {**label, "true_concept": terms["true_concept"]}, self.basis),
            "image_plus_true_concept": _compose(reference, nuisance, {**image, "true_concept": terms["true_concept"]}, self.basis)}
        supplied.update({name: _compose(reference, nuisance, {**label, name: terms[name]}, self.basis)
                         for name in (f"false_group_{i}" for i in range(1, 9))})
        require(set(supplied) == set(FIXED_PANEL_MODELS), "complete_fifteen_model_panel_required")
        self.check()
        return supplied


def make_tuning_recipe(recipe):
    """Construct only the explicitly selected callbacks; never select a trial."""
    recipe = validate_recipe(recipe)
    width = BASIS_WIDTHS[recipe["temporal_basis"]]
    nuisance_columns, reduced_columns = 4 + 7 * width, 4 + 5 * width

    def base(dimensions):
        require(dimensions.columns["nuisance"] == nuisance_columns
                and dimensions.columns["reduced_task_nuisance"] == reduced_columns,
                "frozen_nuisance_columns_differ")
        nuisance = (recipe["intercept_penalty"],) + (recipe["nuisance_penalty"],) * (nuisance_columns - 1)
        reduced = nuisance[:reduced_columns]
        label = nuisance + (recipe["shared_feature_penalty"],) * (dimensions.columns["label_linear"] - nuisance_columns)
        return BasePenaltyPlan(nuisance, reduced, label)

    def covariance(residuals, weights):
        mean = weights @ residuals / weights.sum()
        centered = residuals - mean
        trace = float(np.sum(weights[:, None] * centered * centered) / weights.sum())
        return WhiteningChoices(True, float(weights.sum()), np.eye(61) * (trace / 61.), recipe["shrinkage_strength"])

    def match(view):
        index = view.feasible.concept_index
        reference = view.capacity.curve("true_concept").term_edf[index]
        def nearest(name, allowed):
            values = view.capacity.curve(name).term_edf
            return min(allowed, key=lambda j: (abs(values[j] - reference), -RIDGE_GRID[j], j))
        return MatchedIndices(index, nearest("label_capacity_matched", view.feasible.label_indices),
            tuple(nearest(f"false_group_{i+1}", allowed) for i, allowed in enumerate(view.feasible.false_indices)))

    def select(view):
        concept = min(view.feasible_concept_indices,
            key=lambda j: (view.matched_label_mean_losses[j], -RIDGE_GRID[j], j))
        image = min(range(13), key=lambda j: (view.image_mean_losses[j], -RIDGE_GRID[j], j))
        return BaselineChoice(concept, image)

    def explanatory(view):
        return (view.nuisance_penalties
            + (RIDGE_GRID[view.selected_image_index],) * (view.image_columns - len(view.nuisance_penalties))
            + (RIDGE_GRID[view.selected_concept_index],) * (view.full_columns - view.image_columns))

    return TuningRecipe(base, covariance, match, select, explanatory,
        recipe["fit_weight_scale"], recipe["fit_weight_multiplier"], recipe["normalization"],
        recipe["capacity_tie_break"] + ";" + recipe["baseline_tie_break"],
        recipe["base_rank_relative_tolerance"], recipe["dense_design_byte_limit"])
