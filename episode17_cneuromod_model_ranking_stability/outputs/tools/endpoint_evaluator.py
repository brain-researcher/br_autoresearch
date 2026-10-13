"""EP17 primary endpoint composition on supplied, already qualified arrays.

Freeze ``EndpointFamily`` from calibration-defined supports BEFORE binding a
pair's error sums. ``family.bind(phase, errors, pair_id=...)`` supplies the statistic missing
between prediction_core and uncertainty_core. It retains seven eligible primary
endpoints and applicable H components, with no model fitting or endpoint refit.

Each error matrix contains whole-concept-block sums over equal-repetition image
means, around the corresponding development-training mean. The caller must
establish that baseline, actual concept/role provenance, response nondegeneracy,
prediction alignment, resources and effective reader isolation. Metadata equality
here cannot authenticate those facts. There is no file reader, private artifact
loader, candidate selection, audit-opening path or scientific acceptance gate.
Correlation/CKA and final audit adjudication remain separate components.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

import measurement_core as measurement
import support_core as support
import uncertainty_core as uncertainty

SUBJECTS = measurement.SUBJECTS
ROIS = measurement.ROIS
GROUPS = tuple((person, roi) for person in SUBJECTS for roi in ROIS)
STAGES = ("B", "C", "D")
PRIMARY = ("B_Aall_raw", "C_Aall_raw", "D_Aall_raw", "D_AN_raw",
           "D_RN_raw", "D_VNC_raw", "D_VNC_normalized")
H_COMPONENTS = ("H_BC", "H_CD")
BETA_EDGES = {"B_to_C": ("B", "C", "H_BC"), "C_to_D": ("C", "D", "H_CD")}
EDGES = (("B_Aall_raw", "C_Aall_raw"), ("C_Aall_raw", "D_Aall_raw"),
         ("D_Aall_raw", "D_AN_raw"), ("D_AN_raw", "D_RN_raw"),
         ("D_VNC_raw", "D_VNC_normalized"))
GAIN_INAPPLICABILITY = frozenset(("undefined_calibration_gain", "calibration_gain_variance_zero",
                                "residual_gain_variance_below_floor", "gain_design_rank_deficient"))
# Exact signs from model_pair_registry_specification.md; importing the feature
# loader merely for these identifiers would unnecessarily require Torch/Pillow.
PAIR_SIGNS = MappingProxyType({"P1": ("DINO_R50", "SUP_R50_V1"),
                              "P2": ("SWAV_R50_800", "SUP_R50_V1"),
                              "P3": ("DINO_R50", "SWAV_R50_800"),
                              "F1": ("SUP_R50_V1", "RANDOM_R50_S17")})


class EndpointError(ValueError):
    """Refuse the supplied comparison; never delete a block, person or voxel."""


def require(condition, code):
    if not condition:
        raise EndpointError(code)


def frozen_array(value, *, integer=False):
    raw = np.asarray(value)
    require(raw.dtype.kind in ("iu" if integer else "iuf"), "array_dtype_invalid")
    # Immutable bytes back the snapshot, so input mutation or changing a public
    # writeable flag cannot change the admitted arrays during a bootstrap.
    converted = np.asarray(raw, dtype=np.int64 if integer else np.float64)
    require(np.isfinite(converted).all(), "array_nonfinite")
    return np.frombuffer(converted.tobytes(order="C"), dtype=converted.dtype).reshape(converted.shape)


def column_ids(value, name, *, allow_empty=False):
    raw = np.asarray(value)
    require(raw.dtype.kind in "iu" and raw.ndim == 1 and (allow_empty or raw.size > 0), name + "_invalid")
    require((raw >= 0).all() and (raw <= np.iinfo(np.int64).max).all(), name + "_invalid")
    out = frozen_array(raw, integer=True)
    require(len(np.unique(out)) == len(out), name + "_duplicate")
    return out


@dataclass(frozen=True)
class GainStrata:
    """Original native IDs from the already frozen calibration median split."""
    low_columns: np.ndarray
    high_columns: np.ndarray


@dataclass(frozen=True)
class InapplicableGain:
    """Explicit calibration inapplicability; missing/unknown is not this state."""
    reason: str


@dataclass(frozen=True)
class FrozenSupport:
    """Inputs correspond to derive_private_operators' per-group array keys.

    A_N_weights is the mean inclusion weight of all 32 original banks. This
    interface consumes that admitted operator; it does not reconstruct or
    requalify the banks from outcomes. beta_strata must explicitly specify both
    H_BC and H_CD, each as GainStrata or a known InapplicableGain.
    """
    A_all: np.ndarray
    A_all_spatial_bins: np.ndarray
    A_N_weights: np.ndarray
    R_N_weights: np.ndarray
    V_NC: np.ndarray
    V_NC_ceiling_denominator: np.ndarray
    beta_strata: Mapping[str, GainStrata | InapplicableGain]


@dataclass(frozen=True)
class ConceptBlock:
    block_id: str | int
    image_ids: tuple[str, ...]
    concept_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairedBlockErrors:
    """One group, one A-minus-B pair, shared image/column axes at B/C/D.

    sse_a, sse_b and sst each map B/C/D to (blocks, A_all columns) arrays.
    Both models must use the same observed images and development-training
    baseline, hence there is one SST per stage. Sums ALREADY include all images;
    image counts are never multiplied into these sums a second time.
    """
    model_order: tuple[str, str]
    column_ids: np.ndarray
    blocks: tuple[ConceptBlock, ...]
    sse_a: Mapping[str, np.ndarray]
    sse_b: Mapping[str, np.ndarray]
    sst: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _BoundSupport:
    columns: np.ndarray
    anatomical_weights: np.ndarray
    reliable_weights: np.ndarray
    ceiling_positions: np.ndarray
    ceiling_denominators: np.ndarray
    strata: Mapping[str, tuple[np.ndarray, np.ndarray] | InapplicableGain]
    spatial_bins: np.ndarray


def bind_support(group, value):
    require(isinstance(value, FrozenSupport), "frozen_support_required")
    columns = column_ids(value.A_all, "A_all")
    require(len(columns) >= support.N and (columns < support.PARTICIPANT_COLUMNS[group[0]]).all(),
            "frozen_A_all_invalid")
    bins = frozen_array(value.A_all_spatial_bins, integer=True)
    require(bins.shape == columns.shape and set(bins.tolist()) == set(range(1, 9))
            and all(np.count_nonzero(bins == b) >= support.Q for b in range(1, 9)), "frozen_spatial_bins_invalid")
    anatomical = frozen_array(value.A_N_weights)
    reliable = frozen_array(value.R_N_weights)
    for name, weights in (("A_N", anatomical), ("R_N", reliable)):
        require(weights.shape == columns.shape and (weights >= 0).all()
                and np.isclose(weights.sum(), 1., rtol=0, atol=1e-12), name + "_weights_invalid")
        require(all(np.isclose(weights[bins == b].sum(), support.Q / support.N, rtol=0, atol=1e-12)
                    for b in range(1, 9)), name + "_spatial_quota_changed")
    inclusion = anatomical * (len(support.BANKS) * support.N)
    require((inclusion <= len(support.BANKS)).all()
            and np.allclose(inclusion, np.rint(inclusion), rtol=0, atol=1e-12), "A_N_inclusion_weights_invalid")
    require(np.count_nonzero(reliable) == support.N
            and np.array_equal(reliable[reliable > 0], np.full(support.N, 1. / support.N)),
            "R_N_count_or_weights_changed")
    lookup = {int(column): index for index, column in enumerate(columns)}

    def positions(ids, name, *, allow_empty=False):
        ids = column_ids(ids, name, allow_empty=allow_empty)
        require(all(int(column) in lookup for column in ids), name + "_outside_A_all")
        return frozen_array([lookup[int(column)] for column in ids], integer=True) if len(ids) else frozen_array(np.array([], dtype=int), integer=True)

    ceiling = positions(value.V_NC, "V_NC", allow_empty=True)
    denominator = frozen_array(value.V_NC_ceiling_denominator)
    require(denominator.shape == ceiling.shape and (denominator >= .10).all(), "ceiling_denominator_invalid")
    require(isinstance(value.beta_strata, Mapping) and set(value.beta_strata) == set(H_COMPONENTS),
            "both_gain_dispositions_required")
    strata = {}
    for component in H_COMPONENTS:
        entry = value.beta_strata[component]
        if isinstance(entry, InapplicableGain):
            require(entry.reason in GAIN_INAPPLICABILITY, "gain_inapplicability_unresolved")
            strata[component] = entry
        else:
            require(isinstance(entry, GainStrata), "gain_disposition_unresolved")
            low = positions(entry.low_columns, "gain_low")
            high = positions(entry.high_columns, "gain_high")
            require(len(low) == len(columns) // 2 and len(high) == len(columns) - len(low)
                    and np.array_equal(np.sort(np.r_[low, high]), np.arange(len(columns))),
                    "gain_strata_changed_or_incomplete")
            strata[component] = (low, high)
    return _BoundSupport(columns, anatomical, reliable, ceiling, denominator, MappingProxyType(strata), bins)


@dataclass(frozen=True, init=False, eq=False)
class EndpointFamily:
    """Freeze the complete eligible family from supplied calibration supports."""
    def __init__(self, supports):
        require(isinstance(supports, Mapping) and set(supports) == set(GROUPS), "eight_participant_ROIs_required")
        object.__setattr__(self, "_supports", MappingProxyType({group: bind_support(group, supports[group]) for group in GROUPS}))
        ceiling_eligible = all(len(value.ceiling_positions) > 0 for value in self._supports.values())
        object.__setattr__(self, "primary_endpoint_ids", PRIMARY if ceiling_eligible else PRIMARY[:5])
        object.__setattr__(self, "h_component_ids", tuple(key for key in H_COMPONENTS
                           if all(not isinstance(value.strata[key], InapplicableGain)
                                  for value in self._supports.values())))
        object.__setattr__(self, "component_ids", self.primary_endpoint_ids + self.h_component_ids)
        object.__setattr__(self, "score_scales", tuple("ceiling_normalized_R2" if name == "D_VNC_normalized" else
                           "beta_high_minus_low_gain" if name in H_COMPONENTS else "raw_R2"
                           for name in self.component_ids))
        object.__setattr__(self, "directed_edges", tuple(edge for edge in EDGES
                           if all(name in self.primary_endpoint_ids for name in edge)))
        # This records calibration-defined applicability only. Required 15-trial
        # initial coverage is NOT waived if an edge is unavailable.
        object.__setattr__(self, "calibration_inapplicability", MappingProxyType({
            "ceiling_empty_groups": tuple(group for group, value in self._supports.items()
                                          if not len(value.ceiling_positions)),
            "gain_reasons": MappingProxyType({(group, key): entry.reason
                                              for group, value in self._supports.items()
                                              for key, entry in value.strata.items()
                                              if isinstance(entry, InapplicableGain)})}))

    def bind(self, phase, errors, *, pair_id):
        return EndpointStatistic(self, phase, errors, pair_id=pair_id)


def bind_blocks(raw, phase):
    expected_blocks, concepts_per_block = (24, 20) if phase == "development" else (12, 10)
    require(isinstance(raw, (tuple, list)) and len(raw) == expected_blocks, "complete_concept_blocks_required")
    out, seen_blocks, seen_images, seen_concepts = [], set(), set(), set()
    for block in raw:
        require(isinstance(block, ConceptBlock) and
                ((isinstance(block.block_id, str) and bool(block.block_id)) or
                 (isinstance(block.block_id, Integral) and not isinstance(block.block_id, (bool, np.bool_))))
                and block.block_id not in seen_blocks, "block_identity_invalid")
        images, concepts = tuple(block.image_ids), tuple(block.concept_ids)
        require(images and len(images) == len(concepts) and all(isinstance(x, str) and x for x in images + concepts),
                "image_concept_identity_invalid")
        require(len(set(images)) == len(images) and not seen_images.intersection(images), "image_identity_duplicate")
        require(len(set(concepts)) == concepts_per_block and not seen_concepts.intersection(concepts),
                "whole_concept_block_assignment_changed")
        seen_blocks.add(block.block_id)
        seen_images.update(images)
        seen_concepts.update(concepts)
        out.append(ConceptBlock(block.block_id, images, concepts))
    return tuple(out)


def multiplicities(value, count, name):
    raw = np.asarray(value)
    require(raw.dtype.kind in "iu" and raw.shape == (count,) and (raw >= 0).all(), name + "_invalid")
    weights = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(weights).all() and np.isfinite(weights.sum()) and weights.sum() > 0, name + "_empty_or_overflow")
    return weights


@dataclass(frozen=True, init=False, eq=False)
class EndpointStatistic:
    """One bound model pair; callable by the fixed crossed bootstrap.

    Multiplicities may also describe a single participant or leave-one-out
    panel. Retained participant/block weights are normalized by their own sum;
    omitted participants never cause available-case ROI or voxel averaging.
    """
    def __init__(self, family, phase, errors, *, pair_id):
        require(isinstance(family, EndpointFamily), "frozen_endpoint_family_required")
        require(phase in ("development", "audit"), "phase_invalid")
        require(isinstance(pair_id, str) and pair_id in PAIR_SIGNS, "registered_pair_required")
        require(isinstance(errors, Mapping) and set(errors) == set(GROUPS), "eight_participant_ROIs_required")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "model_order", PAIR_SIGNS[pair_id])
        bound, blocks = {}, None
        for group in GROUPS:
            entry = errors[group]
            require(isinstance(entry, PairedBlockErrors), "paired_block_errors_required")
            require(isinstance(entry.model_order, (tuple, list)) and tuple(entry.model_order) == self.model_order,
                    "fixed_pair_sign_or_model_identity_changed")
            columns = column_ids(entry.column_ids, "error_columns")
            require(np.array_equal(columns, family._supports[group].columns), "error_support_column_identity_changed")
            current = bind_blocks(entry.blocks, phase)
            if blocks is None:
                blocks = current
            require(current == blocks, "paired_image_concept_block_alignment_changed")
            arrays = []
            for name, stage_values in (("sse_a", entry.sse_a), ("sse_b", entry.sse_b), ("sst", entry.sst)):
                require(isinstance(stage_values, Mapping) and set(stage_values) == set(STAGES), "complete_B_C_D_required")
                stage_arrays = []
                for stage in STAGES:
                    value = frozen_array(stage_values[stage])
                    require(value.shape == (len(blocks), len(columns)), name + "_block_column_shape_changed")
                    require((value > 0).all() if name == "sst" else (value >= 0).all(), name + "_invalid")
                    stage_arrays.append(value)
                arrays.append(tuple(stage_arrays))
            bound[group] = tuple(arrays)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "_errors", MappingProxyType(bound))
        object.__setattr__(self, "component_ids", family.component_ids)
        object.__setattr__(self, "primary_endpoint_ids", family.primary_endpoint_ids)
        object.__setattr__(self, "score_scales", family.score_scales)

    def _advantages(self, group, weights):
        a, b, denominator = self._errors[group]
        scores = []
        for stage in range(3):
            sst = weights @ denominator[stage]
            sse_a, sse_b = weights @ a[stage], weights @ b[stage]
            require(np.isfinite(sst).all() and (sst > 0).all()
                    and np.isfinite(sse_a).all() and np.isfinite(sse_b).all(), "pooled_error_sums_invalid")
            # Preserve separate model R² and all negative values. Pool block
            # sums before the division; image counts already enter those sums.
            advantage = (1. - sse_a / sst) - (1. - sse_b / sst)
            require(np.isfinite(advantage).all(), "pooled_advantage_nonfinite")
            scores.append(advantage)
        return scores

    def participant_effects(self, block_multiplicities):
        weights = multiplicities(block_multiplicities, len(self.blocks), "block_multiplicities")
        values = np.zeros((4, len(self.component_ids)))
        for subject_index, person in enumerate(SUBJECTS):
            for roi in ROIS:
                group = (person, roi)
                frozen = self.family._supports[group]
                b, c, d = self._advantages(group, weights)
                endpoints = {"B_Aall_raw": b.mean(), "C_Aall_raw": c.mean(), "D_Aall_raw": d.mean(),
                             "D_AN_raw": frozen.anatomical_weights @ d,
                             "D_RN_raw": frozen.reliable_weights @ d}
                if "D_VNC_raw" in self.primary_endpoint_ids:
                    selected = d[frozen.ceiling_positions]
                    endpoints["D_VNC_raw"] = selected.mean()
                    endpoints["D_VNC_normalized"] = (selected / frozen.ceiling_denominators).mean()
                for component, change in (("H_BC", c - b), ("H_CD", d - c)):
                    if component in self.family.h_component_ids:
                        low, high = frozen.strata[component]
                        endpoints[component] = change[high].mean() - change[low].mean()
                values[subject_index] += np.array([endpoints[key] for key in self.component_ids]) / 2.
        require(np.isfinite(values).all(), "endpoint_nonfinite")
        return frozen_array(values)

    def __call__(self, participant_multiplicities, block_multiplicities):
        weights = multiplicities(participant_multiplicities, 4, "participant_multiplicities")
        value = (weights / weights.sum()) @ self.participant_effects(block_multiplicities)
        require(np.isfinite(value).all(), "endpoint_nonfinite")
        return value

    def group_beta_block_changes(self, group, edge):
        """Blockwise targets for the existing beta forecast, never pooled R².

        Development callers may pass these 24-by-voxel arrays to the already
        qualified measurement.fit_beta_forecast. Audit applies a frozen forecast
        and must not fit it. This method neither fits nor authorizes either use.
        """
        require(group in GROUPS and edge in BETA_EDGES, "beta_group_or_edge_invalid")
        before, after, component = BETA_EDGES[edge]
        require(component in self.family.h_component_ids, "beta_explanation_inapplicable")
        a, b, denominator = self._errors[group]
        changes = []
        for stage in (STAGES.index(before), STAGES.index(after)):
            changes.append((1. - a[stage] / denominator[stage]) - (1. - b[stage] / denominator[stage]))
        value = changes[1] - changes[0]
        require(np.isfinite(value).all(), "beta_block_change_nonfinite")
        return frozen_array(value)

    def beta_block_mean(self, edge, participant_multiplicities, block_multiplicities):
        """Equal-block companion target; separate from pooled-image endpoints."""
        people = multiplicities(participant_multiplicities, 4, "participant_multiplicities")
        blocks = multiplicities(block_multiplicities, len(self.blocks), "block_multiplicities")
        people, blocks = people / people.sum(), blocks / blocks.sum()
        result = 0.
        for index, person in enumerate(SUBJECTS):
            for roi in ROIS:
                result += people[index] * float(blocks @ self.group_beta_block_changes((person, roi), edge).mean(axis=1)) / 2.
        require(np.isfinite(result), "beta_block_mean_nonfinite")
        return float(result)

    def crossed_bounds(self, family="complete"):
        """Reuse the registered 10,000-draw engine, with no arbitrary subset.

        complete chooses F_raw_normalized when its endpoint is eligible, else
        F_raw. F_raw remains available for the registered raw-scale comparisons.
        Unselected H contrasts stay inside the family but never become primary
        stability endpoints.
        """
        require(family in ("complete", "F_raw", "F_raw_normalized"), "inference_family_invalid")
        normalized = "D_VNC_normalized" in self.primary_endpoint_ids
        require(family != "F_raw_normalized" or normalized, "normalized_family_inapplicable")
        indices = tuple(index for index, key in enumerate(self.component_ids)
                        if family != "F_raw" or key != "D_VNC_normalized")
        ids = tuple(self.component_ids[index] for index in indices)
        scales = tuple(self.score_scales[index] for index in indices)
        primary = tuple(key for key in self.primary_endpoint_ids if key in ids)
        return uncertainty.crossed_bounds(lambda people, blocks: self(people, blocks)[list(indices)],
                                          ids, scales, self.phase, primary_endpoint_ids=primary)
