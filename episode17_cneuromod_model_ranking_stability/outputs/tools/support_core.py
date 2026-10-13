"""EP17 fixed spatial-quota support construction; supplied metadata/arrays only.

The caller binds these records to structural_support/attempt001 and supplies
calibration-only A_all validity and D reliability under the declared reader
boundary. This module has no filesystem/source reader, score or selector.
Outputs are calibration-derived support artifacts; the caller owns release.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real

PARTICIPANT_COLUMNS = {"sub-01": 227214, "sub-02": 236701,
                       "sub-03": 210349, "sub-06": 207095}
ROI_LABELS = {"left": 1007, "right": 2007}
BINS = tuple(f"BIN{i:02d}" for i in range(1, 9))
BANKS = tuple(f"BANK{i:02d}" for i in range(1, 33))
Q = 32
N = 256
SEED = 17
CORRELATION_ROUNDOFF = 1e-12


class SupportError(ValueError):
    """Refuse the affected support; never authorize quota or membership rescue."""


def require(condition, code):
    if not condition:
        raise SupportError(code)


def column_id(value):
    require(isinstance(value, Integral) and not isinstance(value, bool), "invalid_original_column_id")
    value = int(value)
    require(value >= 0, "negative_original_column_id")
    return value


@dataclass(frozen=True)
class StructuralColumn:
    participant: str
    roi: str
    column_id: int
    spatial_bin: str
    atlas_label: int


@dataclass(frozen=True)
class _Group:
    participant: str
    roi: str
    columns: tuple[StructuralColumn, ...]
    a_all_ids: tuple[int, ...]
    eligible_by_bin: tuple[tuple[int, ...], ...]

    @property
    def column_ids(self):
        return tuple(row.column_id for row in self.columns)

    @property
    def spatial_bins(self):
        return tuple(row.spatial_bin for row in self.columns)


def _prepare(columns, a_all_ids):
    columns = tuple(columns)
    require(bool(columns) and all(isinstance(row, StructuralColumn) for row in columns), "structural_columns_required")
    participant, roi = columns[0].participant, columns[0].roi
    require(participant in PARTICIPANT_COLUMNS and roi in ROI_LABELS, "unregistered_participant_or_ROI")
    seen, normalized = set(), []
    for row in columns:
        require(row.participant == participant and row.roi == roi, "mixed_participant_or_ROI")
        require(isinstance(row.atlas_label, Integral) and not isinstance(row.atlas_label, bool)
                and int(row.atlas_label) == ROI_LABELS[roi], "ROI_label_mismatch")
        ident = column_id(row.column_id)
        require(ident < PARTICIPANT_COLUMNS[participant], "column_outside_original_axis")
        require(ident not in seen, "duplicate_original_column")
        require(row.spatial_bin in BINS, "unregistered_spatial_bin")
        seen.add(ident)
        normalized.append(StructuralColumn(participant, roi, ident, row.spatial_bin, int(row.atlas_label)))
    normalized.sort(key=lambda row: row.column_id)
    eligible = tuple(column_id(value) for value in a_all_ids)
    require(len(eligible) == len(set(eligible)), "duplicate_A_all_column")
    require(set(eligible).issubset(seen), "A_all_column_not_in_structural_group")
    eligible = tuple(sorted(eligible))
    eligible_set = set(eligible)
    by_bin = tuple(tuple(row.column_id for row in normalized
                         if row.spatial_bin == spatial_bin and row.column_id in eligible_set)
                   for spatial_bin in BINS)
    require(all(len(ids) >= Q for ids in by_bin), "insufficient_A_all_bin_quota")
    return _Group(participant, roi, tuple(normalized), eligible, by_bin)


def _digest(participant, roi, spatial_bin, bank, ident):
    text = f"ep17-anatomical-bank-v1|17|{participant}|{roi}|{spatial_bin}|{bank}|{ident}"
    return hashlib.sha256(text.encode("utf-8")).digest()


@dataclass(frozen=True)
class AnatomicalBank:
    participant: str
    roi: str
    column_ids: tuple[int, ...]
    spatial_bins: tuple[str, ...]
    a_all_ids: tuple[int, ...]
    # Each tuple contains the original column IDs of one complete subset,
    # sorted for identity-preserving gathering. Bank order is BANK01..BANK32.
    banks: tuple[tuple[int, ...], ...]
    inclusion_counts: tuple[int, ...]

    @property
    def member_weight(self):
        return 1 / N

    @property
    def mean_weights(self):
        """Weights aligned to column_ids for the arithmetic mean of32 means."""
        return tuple(count / (len(BANKS) * N) for count in self.inclusion_counts)

    @property
    def a_all_weights(self):
        included = set(self.a_all_ids)
        return tuple(1 / len(included) if ident in included else 0.0 for ident in self.column_ids)

    def overlap_summary(self):
        sets = tuple(set(bank) for bank in self.banks)
        return {"bank_count": len(BANKS), "unique_bank_count": len(set(self.banks)),
                "union_column_count": len(set().union(*sets)),
                "pairwise_intersection_counts": tuple(tuple(len(a & b) for b in sets) for a in sets),
                "bank_members_are_independent_replicates": False}


def build_anatomical_bank(columns, a_all_ids):
    """Reliability-independent bank: there is deliberately no reliability input."""
    group = _prepare(columns, a_all_ids)
    selected_banks = []
    for bank in BANKS:
        selected = []
        for spatial_bin, eligible in zip(BINS, group.eligible_by_bin):
            ranked = sorted(eligible, key=lambda ident: (
                _digest(group.participant, group.roi, spatial_bin, bank, ident), ident))
            selected.extend(ranked[:Q])
        require(len(selected) == N and len(set(selected)) == N, "anatomical_subset_identity_failure")
        selected_banks.append(tuple(sorted(selected)))
    counts = Counter(ident for bank in selected_banks for ident in bank)
    return AnatomicalBank(group.participant, group.roi, group.column_ids, group.spatial_bins,
                          group.a_all_ids, tuple(selected_banks),
                          tuple(counts[ident] for ident in group.column_ids))


@dataclass(frozen=True)
class ReliableSupport:
    participant: str
    roi: str
    column_ids: tuple[int, ...]
    spatial_bins: tuple[str, ...]
    a_all_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    finite_eligible_counts_by_bin: tuple[int, ...]

    @property
    def weights(self):
        selected = set(self.selected_ids)
        return tuple(1 / N if ident in selected else 0.0 for ident in self.column_ids)


def build_reliable_support(columns, a_all_ids, d_reliability_by_id):
    """Select top finite, un-Fisher-clipped D correlations with column-ID ties.

    Reliability must supply exactly the structural group's original-ID keys.
    Nonfinite values remain unselected for R_N; they never alter A_all or the
    anatomical bank. No value is repaired, clipped or re-estimated here.
    """
    group = _prepare(columns, a_all_ids)
    require(hasattr(d_reliability_by_id, "items"), "reliability_ID_mapping_required")
    reliability = {}
    for raw_ident, value in d_reliability_by_id.items():
        ident = column_id(raw_ident)
        require(ident not in reliability, "duplicate_reliability_column")
        require(isinstance(value, Real) and not isinstance(value, bool), "invalid_D_reliability_type")
        reliability[ident] = float(value)
    require(set(reliability) == set(group.column_ids), "reliability_original_ID_mismatch")
    selected, counts = [], []
    for eligible in group.eligible_by_bin:
        finite = [ident for ident in eligible if math.isfinite(reliability[ident])]
        require(all(abs(reliability[ident]) <= 1 + CORRELATION_ROUNDOFF for ident in finite), "D_reliability_outside_Pearson_range")
        require(len(finite) >= Q, "insufficient_finite_D_bin_quota")
        counts.append(len(finite))
        ranked = sorted(finite, key=lambda ident: (-reliability[ident], ident))
        selected.extend(ranked[:Q])
    require(len(selected) == N and len(set(selected)) == N, "reliable_support_identity_failure")
    return ReliableSupport(group.participant, group.roi, group.column_ids, group.spatial_bins,
                           group.a_all_ids, tuple(sorted(selected)), tuple(counts))


def compare_supports(anatomical, reliable):
    """Membership/weight contrasts only; this function consumes no model score."""
    require(isinstance(anatomical, AnatomicalBank) and isinstance(reliable, ReliableSupport), "support_result_types_required")
    identity = lambda value: (value.participant, value.roi, value.column_ids, value.spatial_bins, value.a_all_ids)
    require(identity(anatomical) == identity(reliable), "support_group_or_A_all_mismatch")
    selected = set(reliable.selected_ids)
    intersections = tuple(len(set(bank) & selected) for bank in anatomical.banks)
    return {"column_ids": anatomical.column_ids,
            "reliable_minus_anatomical_mean_weights": tuple(r - a for r, a in zip(reliable.weights, anatomical.mean_weights)),
            "bank_reliable_intersection_counts": intersections,
            "bank_reliable_jaccard": tuple(count / (2 * N - count) for count in intersections),
            "mean_bank_reliable_intersection_count": sum(intersections) / len(BANKS),
            "independent_trial_count_added": 0}
