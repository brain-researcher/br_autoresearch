#!/usr/bin/env python3
"""Deterministic controller grammar for the EP10 v3 development search.

The only filesystem input is the episode-local target-blind 194-cell identity
inventory frozen before M2 outcomes. Model fitters supply trial results; the
controller fixes the prospective ordering, counts, tie rules, patience, null
seeds, and exploratory pair selector.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


RHO_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
ALPHA_GRID = (0.25, 0.5, 0.75)
LOW_RANKS = (1, 2, 3, 4, 6, 8)
FAMILY_ORDER = {
    "lasso": 0,
    "elastic_net": 1,
    "group_lasso": 2,
    "low_rank": 3,
    "sparse_plus_low_rank": 4,
}
CORE_FAMILIES = frozenset(("lasso", "elastic_net", "group_lasso"))
GROUP_SIZES = {
    "212074": 6,
    "220187": 35,
    "220188": 25,
    "220190": 16,
    "221091": 33,
    "221225": 39,
    "221418": 12,
    "221514": 5,
    "221539": 1,
    "221588": 7,
    "221751": 13,
    "233281": 2,
}
PAIR_COUNT = 120
MINIMUM_VALID_TRIALS = 40
MAXIMUM_VALID_TRIALS = 80
REFERENCE_MINIMUM = 8
REFERENCE_MAXIMUM = 16
ASSOCIATION_MINIMUM = 32
ASSOCIATION_MAXIMUM = 72
COVERAGE_TRIALS = 16
PATIENCE = 14
PATIENCE_GAIN = 0.002
MAX_INVALID_RETRIES = 12
NULL_SEEDS = tuple(range(2026093001, 2026093100))
TARGET_ORDER = (
    "allen_993_MOs__ipsilateral",
    "allen_993_MOs__contralateral",
    "allen_453_SS__ipsilateral",
    "allen_453_SS__contralateral",
    "allen_31_ACA__ipsilateral",
    "allen_31_ACA__contralateral",
    "allen_477_STR__ipsilateral",
    "allen_477_STR__contralateral",
    "allen_549_TH__ipsilateral",
    "allen_549_TH__contralateral",
    "allen_313_MB__ipsilateral",
    "allen_313_MB__contralateral",
    "allen_771_P__ipsilateral",
    "allen_771_P__contralateral",
    "allen_354_MY__ipsilateral",
    "allen_354_MY__contralateral",
)
PAIR_NAMES = tuple(
    f"{left}--{right}" for left, right in itertools.combinations(TARGET_ORDER, 2)
)
STAGE_CODES = {
    "M0_refit": 1,
    "M1_refit": 2,
    "M2_fit": 3,
    "optional_diagnostic": 4,
    "candidate_selector": 5,
}
PURPOSE_CODES = {
    "model_fit_or_initialization": 1,
    "fitted_M1_categorical_sampling": 2,
    "Rademacher_design": 3,
    "whole_set_permutation": 4,
    "deterministic_tie_auxiliary": 5,
}
REPLAY_PURPOSE_ROUTES = (
    ("M0_refit", "model_fit_or_initialization"),
    ("M1_refit", "model_fit_or_initialization"),
    ("M1_refit", "fitted_M1_categorical_sampling"),
    ("M2_fit", "model_fit_or_initialization"),
    ("M2_fit", "Rademacher_design"),
    ("M2_fit", "whole_set_permutation"),
    ("optional_diagnostic", "deterministic_tie_auxiliary"),
    ("candidate_selector", "deterministic_tie_auxiliary"),
)
STAGE0_DIR = Path(__file__).resolve().parent.parent
FROZEN_CELL_INVENTORY_PATH = STAGE0_DIR / "v3_m2_frozen_cell_inventory.tsv"
EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES = (
    ("10000", "220187"),
    ("10001", "220187"),
    ("10002", "220187"),
    ("10012", "220188"),
    ("10014", "220188"),
    ("10015", "220188"),
    ("10016", "220188"),
    ("10032", "220190"),
    ("10038", "220190"),
    ("10042", "220190"),
    ("10546", "221091"),
    ("10548", "221091"),
    ("10549", "221091"),
    ("10550", "221091"),
    ("10551", "221091"),
    ("10552", "221091"),
    ("10553", "221091"),
    ("10554", "221091"),
    ("10555", "221091"),
    ("10556", "221091"),
    ("10557", "221091"),
    ("10558", "221091"),
    ("10559", "221091"),
    ("10560", "221091"),
    ("10561", "221091"),
    ("10562", "221091"),
    ("10563", "221091"),
    ("10564", "221091"),
    ("10565", "221091"),
    ("10566", "221091"),
    ("10567", "221091"),
    ("10568", "221091"),
    ("10569", "221091"),
    ("10570", "221091"),
    ("10571", "221091"),
    ("10572", "221091"),
    ("10573", "221091"),
    ("10574", "221091"),
    ("10686", "221225"),
    ("10687", "221225"),
    ("10688", "221225"),
    ("10695", "221225"),
    ("10696", "221225"),
    ("10698", "221225"),
    ("10839", "221418"),
    ("10842", "221418"),
    ("10843", "221418"),
    ("11590", "221751"),
    ("11593", "221751"),
    ("12172", "220187"),
    ("12173", "220187"),
    ("12175", "220187"),
    ("12181", "220187"),
    ("12182", "220187"),
    ("12183", "220187"),
    ("12186", "220187"),
    ("12194", "220188"),
    ("12196", "220188"),
    ("12197", "220188"),
    ("12229", "220190"),
    ("12230", "220190"),
    ("12231", "220190"),
    ("12232", "220190"),
    ("12233", "220190"),
    ("12236", "220190"),
    ("12349", "221091"),
    ("14076", "212074"),
    ("14077", "212074"),
    ("14085", "220187"),
    ("14086", "220187"),
    ("14088", "220187"),
    ("14094", "220187"),
    ("14095", "220187"),
    ("14096", "220187"),
    ("14097", "220187"),
    ("14098", "220187"),
    ("14111", "220188"),
    ("14112", "220188"),
    ("14115", "220188"),
    ("14117", "220188"),
    ("14118", "220188"),
    ("14119", "220188"),
    ("14138", "220190"),
    ("14139", "220190"),
    ("14682", "221225"),
    ("14683", "221225"),
    ("14684", "221225"),
    ("14685", "221225"),
    ("14686", "221225"),
    ("14687", "221225"),
    ("14688", "221225"),
    ("14689", "221225"),
    ("14690", "221225"),
    ("14691", "221225"),
    ("14692", "221225"),
    ("14693", "221225"),
    ("14694", "221225"),
    ("14704", "221225"),
    ("14705", "221225"),
    ("14710", "221225"),
    ("14711", "221225"),
    ("14712", "221225"),
    ("14713", "221225"),
    ("14720", "221225"),
    ("14721", "221225"),
    ("14722", "221225"),
    ("14723", "221225"),
    ("14724", "221225"),
    ("14725", "221225"),
    ("14726", "221225"),
    ("14727", "221225"),
    ("14729", "221225"),
    ("15537", "221539"),
    ("17287", "220187"),
    ("17289", "220187"),
    ("17290", "220187"),
    ("17294", "220187"),
    ("17312", "220188"),
    ("17313", "220188"),
    ("17315", "220188"),
    ("17316", "220188"),
    ("17317", "220188"),
    ("17347", "220190"),
    ("17348", "220190"),
    ("17349", "220190"),
    ("17350", "220190"),
    ("17691", "221514"),
    ("17693", "221514"),
    ("17694", "221514"),
    ("17695", "221514"),
    ("17696", "221514"),
    ("18168", "233281"),
    ("18171", "233281"),
    ("1923", "212074"),
    ("1924", "212074"),
    ("1947", "220187"),
    ("1948", "220187"),
    ("1949", "220187"),
    ("1950", "220187"),
    ("1951", "220187"),
    ("1952", "220187"),
    ("1972", "220187"),
    ("1981", "220188"),
    ("1986", "220188"),
    ("1987", "220188"),
    ("1988", "220188"),
    ("1989", "220188"),
    ("1991", "220188"),
    ("2000", "220190"),
    ("2909", "221091"),
    ("2910", "221091"),
    ("2955", "221225"),
    ("2958", "221225"),
    ("2959", "221225"),
    ("2961", "221225"),
    ("3288", "221418"),
    ("3289", "221418"),
    ("3290", "221418"),
    ("3292", "221418"),
    ("3293", "221418"),
    ("3294", "221418"),
    ("3295", "221418"),
    ("3296", "221418"),
    ("3297", "221418"),
    ("4187", "221588"),
    ("4188", "221588"),
    ("4189", "221588"),
    ("4200", "221588"),
    ("4204", "221588"),
    ("4205", "221588"),
    ("4206", "221588"),
    ("5525", "221751"),
    ("5526", "221751"),
    ("5527", "221751"),
    ("5528", "221751"),
    ("5529", "221751"),
    ("5530", "221751"),
    ("5532", "221751"),
    ("5533", "221751"),
    ("5535", "221751"),
    ("5536", "221751"),
    ("5537", "221751"),
    ("7102", "212074"),
    ("7103", "212074"),
    ("7117", "220188"),
    ("7297", "221091"),
    ("7304", "221091"),
    ("7318", "221225"),
    ("9988", "220187"),
    ("9989", "220187"),
    ("9990", "220187"),
    ("9997", "220187"),
    ("9998", "220187"),
    ("9999", "220187"),
)


class ControllerError(RuntimeError):
    """A prospective controller invariant was violated."""


@dataclass(frozen=True)
class Configuration:
    family: str
    rho: float
    alpha: float | None = None
    rank: int | None = None
    sparse_rho: float | None = None
    sparse_edges: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILY_ORDER:
            raise ControllerError("unknown association family")
        if not math.isfinite(self.rho) or self.rho not in RHO_GRID:
            raise ControllerError("rho is outside the frozen grid")
        if self.family == "elastic_net":
            if self.alpha not in ALPHA_GRID:
                raise ControllerError("elastic-net alpha is outside the frozen grid")
        elif self.alpha is not None:
            raise ControllerError("alpha is only defined for elastic net")
        if self.family in {"low_rank", "sparse_plus_low_rank"}:
            if self.rank not in LOW_RANKS:
                raise ControllerError("optional-family rank is outside the frozen grid")
        elif self.rank is not None:
            raise ControllerError("rank is only defined for optional families")
        if self.family == "sparse_plus_low_rank":
            if self.sparse_rho not in RHO_GRID:
                raise ControllerError("hybrid sparse rho is outside the frozen grid")
            if not 1 <= len(self.sparse_edges) <= 16:
                raise ControllerError("hybrid sparse edge set must contain 1..16 edges")
            if tuple(sorted(set(self.sparse_edges))) != self.sparse_edges or not all(
                0 <= value < PAIR_COUNT for value in self.sparse_edges
            ):
                raise ControllerError("hybrid sparse edge set is malformed")
        elif self.sparse_rho is not None or self.sparse_edges:
            raise ControllerError("sparse-rho/edges are only defined for the hybrid")

    @property
    def key(self) -> str:
        alpha = "none" if self.alpha is None else f"{self.alpha:.12g}"
        rank = "none" if self.rank is None else str(self.rank)
        sparse_rho = "none" if self.sparse_rho is None else f"{self.sparse_rho:.12g}"
        edges = "none" if not self.sparse_edges else "_".join(map(str, self.sparse_edges))
        return (
            f"{self.family}__alpha_{alpha}__rank_{rank}__rho_{self.rho:.12g}"
            f"__sparse_rho_{sparse_rho}__edges_{edges}"
        )


@dataclass(frozen=True)
class Trial:
    valid_index: int
    role: str
    kind: str
    phase: str
    falsifier: bool
    configuration: Configuration | None = None
    seed: int | None = None
    cycle: int | None = None
    matched_configuration: Configuration | None = None
    coverage_binding: CoverageBinding | None = None

    def __post_init__(self) -> None:
        if self.valid_index < 1:
            raise ControllerError("valid trial indexes are one based")
        if self.valid_index <= COVERAGE_TRIALS:
            if self.coverage_binding is not None:
                raise ControllerError("core trials cannot carry post-core evidence binding")
        elif self.coverage_binding is None:
            raise ControllerError("every post-core trial requires coverage binding")
        if self.role not in {"reference", "association"}:
            raise ControllerError("trial role is invalid")
        allowed_kinds = {
            "reference_anchor",
            "anchor",
            "successor",
            "permutation",
            "rademacher_noise",
        }
        if self.kind not in allowed_kinds:
            raise ControllerError("trial kind is invalid")
        if self.kind in {"anchor", "successor", "permutation"}:
            if self.configuration is None or self.role != "association":
                raise ControllerError("association trial lacks configuration")
        elif self.configuration is not None:
            raise ControllerError("reference trial unexpectedly has a configuration")
        if self.kind == "rademacher_noise":
            if self.role != "reference" or self.matched_configuration is None:
                raise ControllerError("noise reference lacks its matched configuration")
        elif self.matched_configuration is not None:
            raise ControllerError("only noise references have matched configurations")
        if self.kind == "reference_anchor":
            if (
                self.role != "reference"
                or self.falsifier
                or self.seed is not None
                or self.phase != "coverage"
                or self.cycle is not None
            ):
                raise ControllerError(
                    "reference anchor role/falsifier/seed/phase/cycle changed"
                )
        elif self.kind == "anchor":
            if (
                self.falsifier
                or self.seed is not None
                or self.phase not in {"coverage", "extension_coverage"}
                or self.cycle is not None
            ):
                raise ControllerError(
                    "association anchor falsifier/seed/phase/cycle changed"
                )
        elif self.kind == "successor":
            if self.falsifier or self.seed is not None or self.cycle is None:
                raise ControllerError("successor role/falsifier/seed/cycle changed")
        elif self.kind in {"permutation", "rademacher_noise"}:
            if not self.falsifier or self.seed is None or self.cycle is None:
                raise ControllerError("falsifier role/seed/cycle changed")
        if self.kind in {"successor", "permutation", "rademacher_noise"}:
            assert self.cycle is not None
            expected_phase = "postcoverage" if self.cycle in {1, 2} else "patience"
            if self.cycle < 1 or self.phase != expected_phase:
                raise ControllerError("adaptive trial phase is not bound to its cycle")
            if self.kind == "rademacher_noise" and self.cycle not in {1, 2}:
                raise ControllerError("Rademacher references occur only in cycles one and two")


@dataclass(frozen=True)
class TrialResult:
    trial: Trial
    gain_bits_per_cell: float
    active_pair_degrees: int
    valid: bool = True
    coverage_binding: CoverageBinding | None = None

    def __post_init__(self) -> None:
        if self.valid:
            if not math.isfinite(self.gain_bits_per_cell):
                raise ControllerError("valid trial gain must be finite")
            if self.active_pair_degrees < 0:
                raise ControllerError("active degree count is negative")
        if self.coverage_binding is not None and self.trial.valid_index != 16:
            raise ControllerError("result-level coverage binding belongs only to trial 16")


_PAIR_CONTRAST_ROW_SEAL = object()


def _canonical_pair_contrast_fields(
    cell_key: str,
    group: str,
    k: int,
    values: Sequence[float],
) -> tuple[str, str, int, tuple[float, ...]]:
    if type(cell_key) is not str or not cell_key:
        raise ControllerError("pair-contrast row has an empty cell key")
    if type(group) is not str or group not in GROUP_SIZES:
        raise ControllerError("pair-contrast row has an unknown group")
    if type(k) is not int or not 0 <= k <= 16:
        raise ControllerError("pair-contrast row K is outside 0..16")
    if isinstance(values, (str, bytes)):
        raise ControllerError("pair-contrast row is malformed or nonfinite")
    try:
        canonical_values = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ControllerError("pair-contrast row is malformed or nonfinite") from error
    if len(canonical_values) != PAIR_COUNT or not all(
        math.isfinite(value) for value in canonical_values
    ):
        raise ControllerError("pair-contrast row is malformed or nonfinite")
    if k in {0, 1, 16} and any(value != 0.0 for value in canonical_values):
        raise ControllerError("K in {0,1,16} pair contrasts must be exact zero")
    return cell_key, group, k, canonical_values


@dataclass(frozen=True, init=False)
class PairContrastRow:
    cell_key: str
    group: str
    k: int
    values: tuple[float, ...]
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        cell_key: str,
        group: str,
        k: int,
        values: tuple[float, ...],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PAIR_CONTRAST_ROW_SEAL:
            raise ControllerError(
                "pair-contrast rows must be issued by make_pair_contrast_row"
            )
        canonical = _canonical_pair_contrast_fields(cell_key, group, k, values)
        if type(values) is not tuple or any(type(value) is not float for value in values):
            raise ControllerError("issued pair-contrast values are not canonical")
        object.__setattr__(self, "cell_key", canonical[0])
        object.__setattr__(self, "group", canonical[1])
        object.__setattr__(self, "k", canonical[2])
        object.__setattr__(self, "values", canonical[3])
        object.__setattr__(self, "_seal", _seal)


def _make_pair_contrast_row_authority():
    records: dict[
        int,
        tuple[
            weakref.ReferenceType[PairContrastRow],
            tuple[str, str, int, tuple[float, ...]],
        ],
    ] = {}

    def issue(
        cell_key: str,
        group: str,
        k: int,
        values: Sequence[float],
    ) -> PairContrastRow:
        canonical = _canonical_pair_contrast_fields(cell_key, group, k, values)
        row = PairContrastRow(
            *canonical,
            _seal=_PAIR_CONTRAST_ROW_SEAL,
        )
        identity = id(row)

        def remove(reference: weakref.ReferenceType[PairContrastRow]) -> None:
            current = records.get(identity)
            if current is not None and current[0] is reference:
                records.pop(identity, None)

        records[identity] = (weakref.ref(row, remove), canonical)
        return row

    def require(row: PairContrastRow) -> PairContrastRow:
        if type(row) is not PairContrastRow:
            raise ControllerError("pair selection requires exact PairContrastRow values")
        record = records.get(id(row))
        if record is None or record[0]() is not row:
            raise ControllerError("pair-contrast row was not issued by this controller")
        if row._seal is not _PAIR_CONTRAST_ROW_SEAL:
            raise ControllerError("pair-contrast row lost its controller authority")
        if (
            type(row.cell_key) is not str
            or not row.cell_key
            or type(row.group) is not str
            or row.group not in GROUP_SIZES
            or type(row.k) is not int
            or not 0 <= row.k <= 16
            or type(row.values) is not tuple
            or len(row.values) != PAIR_COUNT
            or any(type(value) is not float for value in row.values)
            or not all(math.isfinite(value) for value in row.values)
            or (row.k in {0, 1, 16} and any(value != 0.0 for value in row.values))
        ):
            raise ControllerError("pair-contrast row changed into a malformed value")
        current = (row.cell_key, row.group, row.k, row.values)
        if current != record[1]:
            raise ControllerError("pair-contrast row changed after controller issuance")
        return row

    return issue, require


make_pair_contrast_row, _require_canonical_pair_contrast_row = (
    _make_pair_contrast_row_authority()
)


@dataclass(frozen=True)
class FrozenCellInventory:
    """Episode-frozen target-blind identity/group inventory for pair selection."""

    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, tuple) or len(entry) != 2 for entry in self.entries
        ):
            raise ControllerError("frozen inventory must be an immutable pair tuple")
        if len(self.entries) != sum(GROUP_SIZES.values()):
            raise ControllerError("frozen inventory requires exactly 194 cells")
        if any(
            not isinstance(cell_key, str)
            or not cell_key
            or not isinstance(group, str)
            or group not in GROUP_SIZES
            for cell_key, group in self.entries
        ):
            raise ControllerError("frozen inventory has an invalid cell key or group")
        if self.entries != tuple(sorted(self.entries)):
            raise ControllerError("frozen inventory must use canonical identity order")
        if len({cell_key for cell_key, _ in self.entries}) != len(self.entries):
            raise ControllerError("frozen inventory contains duplicate cell keys")
        counts = {group: 0 for group in GROUP_SIZES}
        for _, group in self.entries:
            counts[group] += 1
        if counts != GROUP_SIZES:
            raise ControllerError("frozen inventory group-size vector changed")


def _load_frozen_cell_inventory() -> FrozenCellInventory:
    """Load the one episode-local inventory; callers cannot inject a substitute."""
    path = FROZEN_CELL_INVENTORY_PATH
    if not path.is_file() or path.is_symlink():
        raise ControllerError("frozen target-blind cell inventory is unavailable")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("cell_key", "conservative_group"):
            raise ControllerError("frozen target-blind cell inventory header changed")
        entries = tuple(
            sorted(
                (str(row["cell_key"]), str(row["conservative_group"]))
                for row in reader
            )
        )
    inventory = FrozenCellInventory(entries)
    if inventory.entries != EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES:
        raise ControllerError(
            "frozen target-blind cell inventory differs from its embedded identity seal"
        )
    return inventory


FROZEN_CELL_INVENTORY = _load_frozen_cell_inventory()


@dataclass(frozen=True)
class NullReplayState:
    replicate_index: int
    seed: int
    trial_history: tuple[str, ...] = ()
    M0_refit_required: bool = True
    M1_refit_required: bool = True
    optional_trigger_replay_required: bool = True
    coverage_replay_required: bool = True
    falsifier_replay_required: bool = True
    candidate_selector_replay_required: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.replicate_index <= 99:
            raise ControllerError("null replicate index is outside 1..99")
        if self.seed != NULL_SEEDS[self.replicate_index - 1]:
            raise ControllerError("null replicate seed changed")
        if self.trial_history:
            raise ControllerError("a new null replay must start with empty history")
        if not all(
            (
                self.M0_refit_required,
                self.M1_refit_required,
                self.optional_trigger_replay_required,
                self.coverage_replay_required,
                self.falsifier_replay_required,
                self.candidate_selector_replay_required,
            )
        ):
            raise ControllerError("a null replay cannot disable a required branch")


@dataclass(frozen=True)
class OptionalDiagnostics:
    leading_energy: float
    leading_edge_participation_ratio: float
    median_leave_one_group_cosine: float
    leave_one_group_cosines_at_least_0_70: int

    def __post_init__(self) -> None:
        finite = (
            self.leading_energy,
            self.leading_edge_participation_ratio,
            self.median_leave_one_group_cosine,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ControllerError("optional-family diagnostic is nonfinite")
        if not 0.0 <= self.leading_energy <= 1.0:
            raise ControllerError("leading energy is outside 0..1")
        if not -1.0 <= self.median_leave_one_group_cosine <= 1.0:
            raise ControllerError("median cosine is outside -1..1")
        if not 1.0 <= self.leading_edge_participation_ratio <= PAIR_COUNT:
            raise ControllerError("participation ratio is outside 1..120")
        if (
            isinstance(self.leave_one_group_cosines_at_least_0_70, bool)
            or not isinstance(self.leave_one_group_cosines_at_least_0_70, int)
            or not 0 <= self.leave_one_group_cosines_at_least_0_70 <= 12
        ):
            raise ControllerError("optional-family cosine count is outside 0..12")


@dataclass(frozen=True)
class CoverageBinding:
    """Bit-exact immutable identity for the core scores and branch evidence."""

    core_signature: tuple[tuple[object, ...], ...]
    diagnostic_signature: tuple[str, str, str, int]
    contrast_signature: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.core_signature) != COVERAGE_TRIALS:
            raise ControllerError("coverage binding core signature changed")
        if len(self.diagnostic_signature) != 4:
            raise ControllerError("coverage binding diagnostic signature changed")
        if len(self.contrast_signature) != PAIR_COUNT:
            raise ControllerError("coverage binding contrast signature changed")


@dataclass(frozen=True)
class CoverageEvidence:
    """Immutable trial-16 evidence that determines the optional branch.

    The exact scored core prefix is carried with the target-free diagnostic and
    equal-group contrasts so a caller cannot substitute a different branch
    decision after core coverage.  Construction is intentionally possible only
    after all 16 core anchors have valid results.
    """

    core_results: tuple[TrialResult, ...]
    diagnostic: OptionalDiagnostics
    equal_group_pair_contrasts: tuple[float, ...]
    binding: CoverageBinding

    def __post_init__(self) -> None:
        if not isinstance(self.core_results, tuple) or len(self.core_results) != 16:
            raise ControllerError("coverage evidence requires the exact 16-result core")
        if any(not result.valid for result in self.core_results):
            raise ControllerError("coverage evidence core contains an invalid result")
        if tuple(result.trial for result in self.core_results) != _core_coverage_trials():
            raise ControllerError("coverage evidence core identity or order changed")
        if any(result.coverage_binding is not None for result in self.core_results[:-1]):
            raise ControllerError("coverage binding appeared before trial 16")
        if self.core_results[-1].coverage_binding != self.binding:
            raise ControllerError("trial 16 does not carry the evidence binding")
        contrasts = self.equal_group_pair_contrasts
        if not isinstance(contrasts, tuple):
            raise ControllerError("coverage evidence contrasts must be immutable")
        if len(contrasts) != PAIR_COUNT or not all(
            math.isfinite(float(value)) for value in contrasts
        ):
            raise ControllerError("coverage evidence requires 120 finite contrasts")
        unbound_core = (
            *self.core_results[:-1],
            replace(self.core_results[-1], coverage_binding=None),
        )
        if _coverage_binding(unbound_core, self.diagnostic, contrasts) != self.binding:
            raise ControllerError("coverage evidence differs from its bit-exact binding")

    @property
    def localized(self) -> LocalizedSparseSelection:
        return localized_sparse_selection(self.equal_group_pair_contrasts)

    @property
    def activation(self) -> OptionalActivation:
        return optional_activation(self.diagnostic, self.localized)


@dataclass(frozen=True)
class LocalizedSparseSelection:
    eligible: bool
    edges: tuple[int, ...]
    total_absolute_contrast: float
    captured_fraction: float


@dataclass(frozen=True)
class OptionalActivation:
    low_rank: bool
    sparse_plus_low_rank: bool


@dataclass(frozen=True)
class IncumbentChallengerDecision:
    incumbent: TrialResult
    challenger: TrialResult
    winner: TrialResult


def _config(family: str, rho: float, alpha: float | None = None) -> Configuration:
    return Configuration(family=family, rho=rho, alpha=alpha)


def anchor_configurations() -> tuple[Configuration, ...]:
    """The 15 association anchors; with the M1 reference, coverage is 16."""
    values = (
        _config("lasso", 1.0),
        _config("lasso", 0.1),
        _config("lasso", 0.01),
        _config("lasso", 0.001),
        _config("elastic_net", 1.0, 0.25),
        _config("elastic_net", 0.001, 0.25),
        _config("elastic_net", 1.0, 0.5),
        _config("elastic_net", 0.1, 0.5),
        _config("elastic_net", 0.001, 0.5),
        _config("elastic_net", 1.0, 0.75),
        _config("elastic_net", 0.001, 0.75),
        _config("group_lasso", 1.0),
        _config("group_lasso", 0.1),
        _config("group_lasso", 0.01),
        _config("group_lasso", 0.001),
    )
    if len({value.key for value in values}) != 15:
        raise ControllerError("association anchors are not unique")
    return values


def _core_coverage_trials() -> tuple[Trial, ...]:
    return (
        Trial(1, "reference", "reference_anchor", "coverage", False),
        *(
            Trial(index, "association", "anchor", "coverage", False, configuration)
            for index, configuration in enumerate(anchor_configurations(), start=2)
        ),
    )


def _configuration_signature(
    configuration: Configuration | None,
) -> tuple[object, ...] | None:
    if configuration is None:
        return None
    return (
        configuration.family,
        float(configuration.rho).hex(),
        None if configuration.alpha is None else float(configuration.alpha).hex(),
        configuration.rank,
        None
        if configuration.sparse_rho is None
        else float(configuration.sparse_rho).hex(),
        configuration.sparse_edges,
    )


def _trial_signature(trial: Trial) -> tuple[object, ...]:
    if trial.coverage_binding is not None:
        raise ControllerError("core binding input must contain unbound trials")
    return (
        trial.valid_index,
        trial.role,
        trial.kind,
        trial.phase,
        trial.falsifier,
        _configuration_signature(trial.configuration),
        trial.seed,
        trial.cycle,
        _configuration_signature(trial.matched_configuration),
        None,
    )


def _coverage_binding(
    core_results: Sequence[TrialResult],
    diagnostic: OptionalDiagnostics,
    equal_group_pair_contrasts: Sequence[float],
) -> CoverageBinding:
    if len(core_results) != COVERAGE_TRIALS:
        raise ControllerError("coverage binding requires exactly 16 core results")
    core_signature: list[tuple[object, ...]] = []
    for result in core_results:
        if result.coverage_binding is not None:
            raise ControllerError("coverage binding factory requires an unbound core")
        core_signature.append(_unbound_result_signature(result))
    return CoverageBinding(
        tuple(core_signature),
        (
            float(diagnostic.leading_energy).hex(),
            float(diagnostic.leading_edge_participation_ratio).hex(),
            float(diagnostic.median_leave_one_group_cosine).hex(),
            diagnostic.leave_one_group_cosines_at_least_0_70,
        ),
        tuple(float(value).hex() for value in equal_group_pair_contrasts),
    )


def _unbound_result_signature(result: TrialResult) -> tuple[object, ...]:
    if result.coverage_binding is not None:
        raise ControllerError("result signature requires an unbound core result")
    return (
        _trial_signature(result.trial),
        float(result.gain_bits_per_cell).hex(),
        result.active_pair_degrees,
        result.valid,
        None,
    )


def bind_coverage_evidence(
    core_results: Sequence[TrialResult],
    diagnostic: OptionalDiagnostics,
    equal_group_pair_contrasts: Sequence[float],
) -> CoverageEvidence:
    """Canonically bind one unbound exact core to its trial-16 evidence."""
    core = tuple(core_results)
    contrasts = tuple(float(value) for value in equal_group_pair_contrasts)
    if tuple(result.trial for result in core) != _core_coverage_trials():
        raise ControllerError("coverage binding core identity or order changed")
    if any(not result.valid for result in core):
        raise ControllerError("coverage binding core contains an invalid result")
    binding = _coverage_binding(core, diagnostic, contrasts)
    bound_core = (
        *core[:-1],
        replace(core[-1], coverage_binding=binding),
    )
    return CoverageEvidence(bound_core, diagnostic, contrasts, binding)


def localized_sparse_selection(
    equal_group_pair_contrasts: Sequence[float],
) -> LocalizedSparseSelection:
    """Apply the frozen smallest-80%-cumulative selector with index ties."""
    values = tuple(float(value) for value in equal_group_pair_contrasts)
    if len(values) != PAIR_COUNT or not all(math.isfinite(value) for value in values):
        raise ControllerError("localized selector requires 120 finite contrasts")
    magnitudes = tuple(abs(value) for value in values)
    total = math.fsum(magnitudes)
    if total == 0.0:
        return LocalizedSparseSelection(False, (), 0.0, 0.0)
    order = sorted(range(PAIR_COUNT), key=lambda index: (-magnitudes[index], index))
    selected: list[int] = []
    cumulative = 0.0
    for index in order:
        selected.append(index)
        cumulative += magnitudes[index]
        if cumulative >= 0.80 * total:
            break
    edges = tuple(sorted(selected))
    fraction = cumulative / total
    return LocalizedSparseSelection(len(edges) <= 16, edges, total, fraction)


def optional_activation(
    diagnostic: OptionalDiagnostics,
    localized: LocalizedSparseSelection,
) -> OptionalActivation:
    """Apply the frozen diffuse and independent localized trigger rules."""
    low_rank = (
        diagnostic.leading_energy >= 0.40
        and diagnostic.leading_edge_participation_ratio >= 16.0
        and diagnostic.median_leave_one_group_cosine >= 0.80
        and diagnostic.leave_one_group_cosines_at_least_0_70 >= 9
    )
    return OptionalActivation(low_rank, low_rank and localized.eligible)


def coverage_trials(
    *,
    coverage_evidence: CoverageEvidence | None = None,
    low_rank_anchor_results: Sequence[TrialResult] = (),
) -> tuple[Trial, ...]:
    """Core coverage plus the mandatory extension known at this evidence stage.

    With no evidence this returns only the prospectively known 16 core trials.
    Hybrid evidence first exposes the eight low-rank anchors; after those exact
    results are supplied it also exposes the four score/contrast-bound hybrids.
    """
    trials = list(_core_coverage_trials())
    if coverage_evidence is None:
        if low_rank_anchor_results:
            raise ControllerError("low-rank scores require coverage evidence")
        optional: tuple[Configuration, ...] = ()
    else:
        optional = bound_optional_anchor_configurations(
            coverage_evidence, low_rank_anchor_results
        )
    for configuration in optional:
        trials.append(
            Trial(
                len(trials) + 1,
                "association",
                "anchor",
                "extension_coverage",
                False,
                configuration,
                coverage_binding=coverage_evidence.binding,
            )
        )
    return tuple(trials)


def _optional_anchor_configurations(
    *,
    low_rank_triggered: bool,
    hybrid_rank: int | None = None,
    hybrid_edges: Sequence[int] = (),
) -> tuple[Configuration, ...]:
    """Mandatory extension anchors inserted before coverage can complete."""
    values: list[Configuration] = []
    if low_rank_triggered:
        values.extend(
            Configuration("low_rank", 0.1, rank=rank) for rank in LOW_RANKS
        )
        values.extend(
            (
                Configuration("low_rank", 1.0, rank=LOW_RANKS[0]),
                Configuration("low_rank", 0.001, rank=LOW_RANKS[-1]),
            )
        )
    edge_tuple = tuple(int(value) for value in hybrid_edges)
    if edge_tuple:
        if not low_rank_triggered or hybrid_rank not in LOW_RANKS:
            raise ControllerError("hybrid anchors require an activated low-rank rank")
        for low_rho, sparse_rho in (
            (1.0, 1.0),
            (1.0, 0.001),
            (0.001, 1.0),
            (0.001, 0.001),
        ):
            values.append(
                Configuration(
                    "sparse_plus_low_rank",
                    low_rho,
                    rank=hybrid_rank,
                    sparse_rho=sparse_rho,
                    sparse_edges=edge_tuple,
                )
            )
    if len({value.key for value in values}) != len(values):
        raise ControllerError("optional anchor inventory is duplicated")
    return tuple(values)


def low_rank_anchor_configurations() -> tuple[Configuration, ...]:
    """The eight mandatory low-rank anchors, before outcome-bound rank choice."""
    return _optional_anchor_configurations(low_rank_triggered=True)


def best_low_rank_anchor_rank(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence,
) -> int:
    """Select the rank of the best complete valid low-rank anchor inventory."""
    expected = _optional_anchor_configurations(low_rank_triggered=True)
    if len(results) != len(expected):
        raise ControllerError("hybrid rank selection requires all eight low-rank anchors")
    expected_trials = _extension_anchor_trials(
        expected,
        start_valid_index=COVERAGE_TRIALS + 1,
        coverage_binding=coverage_evidence.binding,
    )
    if tuple(result.trial for result in results) != expected_trials:
        raise ControllerError("low-rank anchor scores are not bound to this evidence")
    by_key = {result.trial.configuration.key: result for result in results if result.trial.configuration is not None}
    if len(by_key) != len(expected) or set(by_key) != {value.key for value in expected}:
        raise ControllerError("low-rank anchor result inventory changed")
    ordered = tuple(by_key[value.key] for value in expected)
    if any(
        not result.valid
        or result.trial.kind != "anchor"
        or result.trial.falsifier
        or result.trial.configuration is None
        or result.trial.configuration.family != "low_rank"
        for result in ordered
    ):
        raise ControllerError("hybrid rank selection received an invalid low-rank anchor")
    winner = choose_incumbent(ordered)
    assert winner.trial.configuration is not None
    assert winner.trial.configuration.rank is not None
    return winner.trial.configuration.rank


def bound_optional_anchor_configurations(
    coverage_evidence: CoverageEvidence,
    low_rank_anchor_results: Sequence[TrialResult] = (),
) -> tuple[Configuration, ...]:
    """Derive optional anchors; rank and edge identity are never caller inputs."""
    localized = coverage_evidence.localized
    activation = coverage_evidence.activation
    if not activation.low_rank:
        if low_rank_anchor_results:
            raise ControllerError("low-rank results supplied when its trigger failed")
        return ()
    if not activation.sparse_plus_low_rank:
        if low_rank_anchor_results:
            raise ControllerError("low-rank-only coverage does not accept rank scores")
        return _optional_anchor_configurations(low_rank_triggered=True)
    if not low_rank_anchor_results:
        return _optional_anchor_configurations(low_rank_triggered=True)
    rank = best_low_rank_anchor_rank(low_rank_anchor_results, coverage_evidence)
    return _optional_anchor_configurations(
        low_rank_triggered=True,
        hybrid_rank=rank,
        hybrid_edges=localized.edges,
    )


def _bind_coverage_evidence(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None,
) -> CoverageEvidence:
    if coverage_evidence is None:
        raise ControllerError("trial-16 coverage evidence is required")
    if len(results) < COVERAGE_TRIALS:
        raise ControllerError("coverage evidence cannot bind before the full core")
    core = tuple(results[:COVERAGE_TRIALS])
    if any(result.coverage_binding is not None for result in core[:-1]):
        raise ControllerError("coverage binding appeared before trial 16")
    if core[-1].coverage_binding != coverage_evidence.binding:
        raise ControllerError("trial 16 carries different coverage evidence")
    unbound_core = (*core[:-1], replace(core[-1], coverage_binding=None))
    if _coverage_binding(
        unbound_core,
        coverage_evidence.diagnostic,
        coverage_evidence.equal_group_pair_contrasts,
    ) != coverage_evidence.binding:
        raise ControllerError("coverage evidence is not bound to this scored core")
    return coverage_evidence


def _complete_coverage_trials(
    coverage_evidence: CoverageEvidence,
    result_history: Sequence[TrialResult] | None = None,
) -> tuple[Trial, ...]:
    activation = coverage_evidence.activation
    if not activation.sparse_plus_low_rank:
        return coverage_trials(coverage_evidence=coverage_evidence)
    if result_history is None or len(result_history) < COVERAGE_TRIALS + 8:
        raise ControllerError(
            "hybrid coverage requires the same scored low-rank anchor history"
        )
    _bind_coverage_evidence(result_history, coverage_evidence)
    return coverage_trials(
        coverage_evidence=coverage_evidence,
        low_rank_anchor_results=result_history[
            COVERAGE_TRIALS : COVERAGE_TRIALS + 8
        ],
    )


def _validate_coverage_prefix(
    trials: Sequence[Trial],
    coverage_evidence: CoverageEvidence | None,
    *,
    result_history: Sequence[TrialResult] | None = None,
) -> int:
    """Validate complete coverage against the sole branch-evidence authority."""
    if coverage_evidence is None:
        raise ControllerError("schedule validation requires trial-16 coverage evidence")
    if tuple(trials[:COVERAGE_TRIALS]) != _core_coverage_trials():
        raise ControllerError("core coverage trial identity, phase, or order changed")
    expected = _complete_coverage_trials(coverage_evidence, result_history)
    if len(trials) < len(expected) or tuple(trials[: len(expected)]) != expected:
        raise ControllerError(
            "coverage branch, optional rank, edges, identity, or order changed"
        )
    return len(expected)


def _validate_result_bound_coverage(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None,
) -> int:
    """Require the complete evidence-selected coverage in this score stream."""
    coverage_length = _validate_available_coverage_prefix(
        results, coverage_evidence
    )
    if coverage_length is None:
        raise ControllerError("coverage result stream is incomplete")
    return coverage_length


def configuration_order_key(configuration: Configuration) -> tuple[object, ...]:
    alpha_index = (
        ALPHA_GRID.index(configuration.alpha)
        if configuration.alpha in ALPHA_GRID
        else -1
    )
    rank_index = LOW_RANKS.index(configuration.rank) if configuration.rank in LOW_RANKS else -1
    rho_index = RHO_GRID.index(configuration.rho) if configuration.rho in RHO_GRID else len(RHO_GRID)
    sparse_rho_index = (
        RHO_GRID.index(configuration.sparse_rho)
        if configuration.sparse_rho in RHO_GRID
        else -1
    )
    return (
        FAMILY_ORDER[configuration.family],
        alpha_index,
        rank_index,
        rho_index,
        sparse_rho_index,
        configuration.sparse_edges,
        configuration.key,
    )


def _incumbent_key(result: TrialResult) -> tuple[object, ...]:
    configuration = result.trial.configuration
    if configuration is None:
        raise ControllerError("incumbent candidates must be association models")
    return (-result.gain_bits_per_cell, *configuration_order_key(configuration))


def choose_incumbent(results: Iterable[TrialResult]) -> TrialResult:
    eligible = [
        result
        for result in results
        if result.valid
        and result.trial.role == "association"
        and not result.trial.falsifier
        and result.trial.kind in {"anchor", "successor"}
    ]
    if not eligible:
        raise ControllerError("no valid real association challenger exists")
    return min(eligible, key=_incumbent_key)


def incumbent_challenger_decision(
    results: Sequence[TrialResult],
    cycle: int,
    *,
    coverage_evidence: CoverageEvidence | None = None,
) -> IncumbentChallengerDecision:
    """Make the required frozen decision after successor cycle one or two."""
    if cycle not in {1, 2}:
        raise ControllerError("only the two minimum-schedule decisions are defined")
    valid = [result for result in results if result.valid]
    if len(valid) < COVERAGE_TRIALS:
        raise ControllerError("decision precedes core coverage")
    coverage_length = _validate_result_bound_coverage(valid, coverage_evidence)
    expected_length = coverage_length + 12 * cycle
    if len(valid) != expected_length:
        raise ControllerError("incumbent-challenger decision is off its frozen time")
    _validate_cycle_successors(valid, cycle, coverage_evidence)
    prior = [
        result
        for result in valid
        if result.trial.cycle is None or result.trial.cycle < cycle
    ]
    challengers = [
        result
        for result in valid
        if result.trial.cycle == cycle and result.trial.kind == "successor"
    ]
    incumbent = choose_incumbent(prior)
    challenger = choose_incumbent(challengers)
    winner = min((incumbent, challenger), key=_incumbent_key)
    return IncumbentChallengerDecision(incumbent, challenger, winner)


def lattice_configurations() -> tuple[Configuration, ...]:
    values: list[Configuration] = []
    values.extend(_config("lasso", rho) for rho in RHO_GRID)
    for alpha in ALPHA_GRID:
        values.extend(_config("elastic_net", rho, alpha) for rho in RHO_GRID)
    values.extend(_config("group_lasso", rho) for rho in RHO_GRID)
    return tuple(values)


def _adjacent_values(value: object, ordered: Sequence[object]) -> tuple[object, ...]:
    try:
        index = ordered.index(value)
    except ValueError as error:
        raise ControllerError("configuration value is outside its frozen lattice") from error
    neighbors = []
    if index > 0:
        neighbors.append(ordered[index - 1])
    if index + 1 < len(ordered):
        neighbors.append(ordered[index + 1])
    return tuple(neighbors)


def neighbor_configurations(parent: Configuration) -> tuple[Configuration, ...]:
    """Return exactly the frozen one-step lattice neighbors of one parent."""
    values: list[Configuration] = []
    for rho in _adjacent_values(parent.rho, RHO_GRID):
        if parent.family == "sparse_plus_low_rank":
            values.append(
                Configuration(
                    parent.family,
                    float(rho),
                    rank=parent.rank,
                    sparse_rho=parent.sparse_rho,
                    sparse_edges=parent.sparse_edges,
                )
            )
        else:
            values.append(
                Configuration(
                    parent.family,
                    float(rho),
                    alpha=parent.alpha,
                    rank=parent.rank,
                )
            )
    if parent.family == "elastic_net":
        assert parent.alpha is not None
        for alpha in _adjacent_values(parent.alpha, ALPHA_GRID):
            values.append(Configuration(parent.family, parent.rho, alpha=float(alpha)))
    if parent.family in {"low_rank", "sparse_plus_low_rank"}:
        assert parent.rank is not None
        for rank in _adjacent_values(parent.rank, LOW_RANKS):
            if parent.family == "low_rank":
                values.append(Configuration(parent.family, parent.rho, rank=int(rank)))
            else:
                values.append(
                    Configuration(
                        parent.family,
                        parent.rho,
                        rank=int(rank),
                        sparse_rho=parent.sparse_rho,
                        sparse_edges=parent.sparse_edges,
                    )
                )
    if parent.family == "sparse_plus_low_rank":
        assert parent.sparse_rho is not None
        for sparse_rho in _adjacent_values(parent.sparse_rho, RHO_GRID):
            values.append(
                Configuration(
                    parent.family,
                    parent.rho,
                    rank=parent.rank,
                    sparse_rho=float(sparse_rho),
                    sparse_edges=parent.sparse_edges,
                )
            )
    return tuple(sorted({value.key: value for value in values}.values(), key=configuration_order_key))


def _successor_configurations_unchecked(
    parent_results: Sequence[TrialResult],
    tried: Iterable[Configuration],
    *,
    count: int = 7,
) -> tuple[Configuration, ...]:
    """Enumerate one-step neighbors across ranked eligible parents."""
    if count < 1:
        raise ControllerError("successor count must be positive")
    used = {value.key for value in tried}
    parents = [
        result
        for result in parent_results
        if result.valid
        and result.trial.role == "association"
        and not result.trial.falsifier
        and result.trial.configuration is not None
    ]
    if not parents:
        raise ControllerError("successor proposal requires valid association parents")
    parents.sort(key=_incumbent_key)
    selected: list[Configuration] = []
    seen = set(used)
    for parent in parents:
        assert parent.trial.configuration is not None
        for candidate in neighbor_configurations(parent.trial.configuration):
            if candidate.key in seen:
                continue
            selected.append(candidate)
            seen.add(candidate.key)
            if len(selected) == count:
                return tuple(selected)
    return tuple(selected)


def successor_configurations(
    parent_results: Sequence[TrialResult],
    tried: Iterable[Configuration] | None = None,
    *,
    count: int = 7,
    coverage_evidence: CoverageEvidence | None = None,
) -> tuple[Configuration, ...]:
    """Propose successors only at the two frozen adaptive decision times."""
    if count != 7:
        raise ControllerError("public successor decisions require exactly seven trials")
    coverage_length = _validate_result_bound_coverage(
        parent_results, coverage_evidence
    )
    bound_tried = _bind_tried_configurations(parent_results, tried)
    tail_length = len(parent_results) - coverage_length
    if tail_length == 0:
        validate_available_result_prefix(parent_results, coverage_evidence)
    elif tail_length == 7 and parent_results[coverage_length].trial.valid_index == (
        coverage_length + 5
    ):
        assert coverage_evidence is not None
        _validate_compact_cycle_one_results(
            parent_results, coverage_length, coverage_evidence.binding
        )
    else:
        raise ControllerError(
            "successor proposals occur only after coverage or seven compact cycle-one scores"
        )
    return _successor_configurations_unchecked(
        parent_results, bound_tried, count=count
    )


def _minimum_cycle_trials(
    configurations: Sequence[Configuration],
    *,
    cycle: int,
    start_valid_index: int,
    coverage_binding: CoverageBinding,
) -> tuple[Trial, ...]:
    """Construct one exact 12-trial minimum-schedule cycle."""
    if len(configurations) != 7 or cycle not in {1, 2}:
        raise ControllerError("minimum cycle requires seven successors and cycle 1 or 2")
    noise_ordinals = range(1, 5) if cycle == 1 else range(5, 8)
    trials: list[Trial] = []
    for offset, seed_ordinal in enumerate(noise_ordinals):
        trials.append(
            Trial(
                start_valid_index + offset,
                "reference",
                "rademacher_noise",
                "postcoverage",
                True,
                seed=2026092800 + seed_ordinal,
                cycle=cycle,
                matched_configuration=configurations[offset],
                coverage_binding=coverage_binding,
            )
        )
    for configuration in configurations:
        trials.append(
            Trial(
                start_valid_index + len(trials),
                "association",
                "successor",
                "postcoverage",
                False,
                configuration=configuration,
                cycle=cycle,
                coverage_binding=coverage_binding,
            )
        )
    permutation_families = ("lasso",) if cycle == 1 else (
        "elastic_net",
        "group_lasso",
    )
    for family in permutation_families:
        trials.append(
            Trial(
                start_valid_index + len(trials),
                "association",
                "permutation",
                "postcoverage",
                True,
                configuration=_config(
                    family, 0.1, 0.5 if family == "elastic_net" else None
                ),
                seed=2026092811 + FAMILY_ORDER[family],
                cycle=cycle,
                coverage_binding=coverage_binding,
            )
        )
    if len(trials) != 12:
        raise ControllerError("minimum cycle trial count changed")
    return tuple(trials)


def postcoverage_trials(
    cycle_one: Sequence[Configuration],
    cycle_two: Sequence[Configuration],
    *,
    start_valid_index: int = 17,
    coverage_evidence: CoverageEvidence,
) -> tuple[Trial, ...]:
    if len(cycle_one) != 7 or len(cycle_two) != 7:
        raise ControllerError("each required successor cycle has seven trials")
    if len({value.key for value in (*cycle_one, *cycle_two)}) != 14:
        raise ControllerError("successor cycles overlap")
    return (
        *_minimum_cycle_trials(
            cycle_one,
            cycle=1,
            start_valid_index=start_valid_index,
            coverage_binding=coverage_evidence.binding,
        ),
        *_minimum_cycle_trials(
            cycle_two,
            cycle=2,
            start_valid_index=start_valid_index + 12,
            coverage_binding=coverage_evidence.binding,
        ),
    )


def _coverage_trials_bound_to_results(
    coverage_results: Sequence[TrialResult],
    *,
    coverage_evidence: CoverageEvidence | None,
) -> tuple[Trial, ...]:
    """Rebuild coverage using only scores in the submitted result stream."""
    if not coverage_results or any(not result.valid for result in coverage_results):
        raise ControllerError("coverage result stream is empty or contains invalid trials")
    evidence = _bind_coverage_evidence(coverage_results, coverage_evidence)
    low_rank_results: tuple[TrialResult, ...] = ()
    if evidence.activation.sparse_plus_low_rank:
        if len(coverage_results) < COVERAGE_TRIALS + 8:
            raise ControllerError(
                "hybrid coverage lacks its scored low-rank anchor prefix"
            )
        low_rank_results = tuple(
            coverage_results[COVERAGE_TRIALS : COVERAGE_TRIALS + 8]
        )
    return coverage_trials(
        coverage_evidence=evidence,
        low_rank_anchor_results=low_rank_results,
    )


def minimum_schedule(
    cycle_one: Sequence[Configuration],
    cycle_two: Sequence[Configuration],
    *,
    coverage_results: Sequence[TrialResult],
    cycle_one_results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None = None,
) -> tuple[Trial, ...]:
    coverage = _coverage_trials_bound_to_results(
        coverage_results,
        coverage_evidence=coverage_evidence,
    )
    if tuple(result.trial for result in coverage_results) != coverage:
        raise ControllerError("minimum schedule coverage results changed or are incomplete")
    if _validate_result_bound_coverage(
        coverage_results, coverage_evidence
    ) != len(coverage):
        raise ControllerError("minimum schedule coverage result stream changed")
    expected_one = successor_configurations(
        coverage_results,
        _real_tried_configurations(coverage_results),
        count=7,
        coverage_evidence=coverage_evidence,
    )
    if tuple(cycle_one) != expected_one:
        raise ControllerError("minimum schedule cycle one is not the ranked proposal")
    if len(cycle_one_results) != 7:
        raise ControllerError("minimum schedule cycle-one result inventory changed")
    before_two = (*coverage_results, *cycle_one_results)
    assert coverage_evidence is not None
    _validate_compact_cycle_one_results(
        before_two, len(coverage), coverage_evidence.binding
    )
    expected_two = successor_configurations(
        before_two,
        _real_tried_configurations(before_two),
        count=7,
        coverage_evidence=coverage_evidence,
    )
    if tuple(cycle_two) != expected_two:
        raise ControllerError("minimum schedule cycle two is not the ranked proposal")
    trials = coverage + postcoverage_trials(
        cycle_one,
        cycle_two,
        start_valid_index=len(coverage) + 1,
        coverage_evidence=coverage_evidence,
    )
    validate_schedule(
        trials,
        result_history=coverage_results,
        coverage_evidence=coverage_evidence,
    )
    return trials


def validate_schedule(
    trials: Sequence[Trial],
    *,
    result_history: Sequence[TrialResult] | None = None,
    coverage_evidence: CoverageEvidence | None = None,
) -> None:
    if coverage_evidence is None:
        raise ControllerError("schedule validation requires trial-16 coverage evidence")
    if any(
        trial.valid_index >= 17
        and trial.coverage_binding != coverage_evidence.binding
        for trial in trials
    ):
        raise ControllerError("post-core trial carries different coverage evidence")
    if tuple(trial.valid_index for trial in trials) != tuple(range(1, len(trials) + 1)):
        raise ControllerError("valid trial indexes are not contiguous")
    if len(trials) < MINIMUM_VALID_TRIALS or len(trials) > MAXIMUM_VALID_TRIALS:
        raise ControllerError("schedule is outside the valid-trial bounds")
    coverage_length = _validate_coverage_prefix(
        trials,
        coverage_evidence,
        result_history=result_history,
    )
    if result_history is not None:
        if (
            len(result_history) < coverage_length
            or len(result_history) > len(trials)
            or tuple(result.trial for result in result_history)
            != tuple(trials[: len(result_history)])
        ):
            raise ControllerError(
                "schedule and scored result history do not share one prefix"
            )
        if _validate_partial_result_history(
            result_history, coverage_evidence=coverage_evidence
        ) != coverage_length:
            raise ControllerError("scored coverage history changed")
    branch_minimum = coverage_length + 24
    if len(trials) < branch_minimum:
        raise ControllerError("triggered-family coverage displaced the minimum schedule")
    references = sum(trial.role == "reference" for trial in trials)
    associations = sum(trial.role == "association" for trial in trials)
    if not REFERENCE_MINIMUM <= references <= REFERENCE_MAXIMUM:
        raise ControllerError("reference-trial count is outside policy")
    if not ASSOCIATION_MINIMUM <= associations <= ASSOCIATION_MAXIMUM:
        raise ControllerError("association-trial count is outside policy")
    real_configurations = [
        trial.configuration
        for trial in trials
        if trial.kind in {"anchor", "successor"} and trial.configuration is not None
    ]
    if len({value.key for value in real_configurations}) != len(real_configurations):
        raise ControllerError("a real anchor/successor configuration was reused")
    post = trials[coverage_length:]
    required_falsifiers = math.ceil(0.4 * len(post))
    if sum(trial.falsifier for trial in post) < required_falsifiers:
        raise ControllerError("postcoverage falsifier fraction is below 40 percent")
    base = post[:24]
    expected_kinds = (
        *("rademacher_noise" for _ in range(4)),
        *("successor" for _ in range(7)),
        "permutation",
        *("rademacher_noise" for _ in range(3)),
        *("successor" for _ in range(7)),
        "permutation",
        "permutation",
    )
    if len(base) != 24 or tuple(trial.kind for trial in base) != expected_kinds:
        raise ControllerError("minimum postcoverage trial order changed")
    if tuple(trial.cycle for trial in base) != (1,) * 12 + (2,) * 12:
        raise ControllerError("minimum postcoverage cycle labels changed")
    cycle_one = tuple(
        trial.configuration for trial in base[4:11] if trial.configuration is not None
    )
    cycle_two = tuple(
        trial.configuration for trial in base[15:22] if trial.configuration is not None
    )
    if len(cycle_one) != 7 or len(cycle_two) != 7 or tuple(base) != postcoverage_trials(
        cycle_one,
        cycle_two,
        start_valid_index=coverage_length + 1,
        coverage_evidence=coverage_evidence,
    ):
        raise ControllerError(
            "minimum postcoverage trial identity, phase, cycle, or order changed"
        )
    noise = [trial for trial in base if trial.kind == "rademacher_noise"]
    if tuple(trial.seed for trial in noise) != tuple(range(2026092801, 2026092808)):
        raise ControllerError("Rademacher seed order changed")
    permutations = [trial for trial in base if trial.kind == "permutation"]
    if tuple(trial.seed for trial in permutations) != (
        2026092811,
        2026092812,
        2026092813,
    ) or tuple(trial.configuration for trial in permutations) != (
        _config("lasso", 0.1),
        _config("elastic_net", 0.1, 0.5),
        _config("group_lasso", 0.1),
    ):
        raise ControllerError("whole-set permutation configuration changed")
    if len(trials) == branch_minimum:
        if references != 8 or associations != branch_minimum - 8:
            raise ControllerError("branch minimum schedule has wrong role counts")
        if sum(trial.falsifier for trial in post) != 10:
            raise ControllerError("branch minimum schedule must contain ten falsifiers")

    continuation = post[24:]
    exhaustion_seeds: list[int] = []
    offset = 0
    block = 0
    while offset < len(continuation):
        block_trials = continuation[offset : offset + 5]
        cycle = 3 + block
        if any(trial.phase != "patience" or trial.cycle != cycle for trial in block_trials):
            raise ControllerError("continuation phase or cycle label changed")
        for slot, trial in enumerate(block_trials):
            if slot < 2:
                expected = _permutation_trial(
                    trial.valid_index,
                    2 * block + slot + 1,
                    cycle,
                    coverage_evidence.binding,
                )
                if trial != expected:
                    raise ControllerError("mandatory continuation falsifier changed")
            elif trial.kind == "permutation":
                if trial.seed is None or trial.seed < 2026092837:
                    raise ControllerError("exhaustion replacement seed changed")
                expected = _permutation_trial(
                    trial.valid_index,
                    trial.seed - 2026092820,
                    cycle,
                    coverage_evidence.binding,
                )
                if trial != expected:
                    raise ControllerError("exhaustion replacement configuration changed")
                exhaustion_seeds.append(trial.seed)
            elif trial.kind != "successor":
                raise ControllerError("continuation successor slot has an invalid kind")
        first_exhaustion = next(
            (index for index, trial in enumerate(block_trials[2:]) if trial.kind == "permutation"),
            None,
        )
        if first_exhaustion is not None and any(
            trial.kind != "permutation" for trial in block_trials[2 + first_exhaustion :]
        ):
            raise ControllerError("successor appears after exhaustion within a block")
        offset += len(block_trials)
        block += 1
    if exhaustion_seeds != list(range(2026092837, 2026092837 + len(exhaustion_seeds))):
        raise ControllerError("exhaustion replacement seeds are not contiguous")


def _permutation_trial(
    valid_index: int,
    ordinal: int,
    cycle: int,
    coverage_binding: CoverageBinding,
) -> Trial:
    family = tuple(sorted(CORE_FAMILIES, key=FAMILY_ORDER.__getitem__))[(ordinal - 1) % 3]
    alpha = 0.5 if family == "elastic_net" else None
    return Trial(
        valid_index,
        "association",
        "permutation",
        "patience",
        True,
        configuration=_config(family, 0.1, alpha),
        seed=2026092820 + ordinal,
        cycle=cycle,
        coverage_binding=coverage_binding,
    )


def _continuation_block_unchecked(
    start_valid_index: int,
    parent_results: Sequence[TrialResult],
    tried: Iterable[Configuration],
    *,
    coverage_length: int = COVERAGE_TRIALS,
    coverage_binding: CoverageBinding,
) -> tuple[Trial, ...]:
    """Construct one F,F,R,R,R block after caller-side history validation."""
    if coverage_length not in {16, 24, 28}:
        raise ControllerError("continuation received an invalid coverage length")
    if start_valid_index != len(parent_results) + 1:
        raise ControllerError("continuation must start at the exact next valid index")
    first_start = coverage_length + 25
    if start_valid_index not in range(first_start, 81, 5):
        raise ControllerError("continuation block start is off the branch lattice")
    block = (start_valid_index - first_start) // 5
    cycle = 3 + block
    first_ordinal = 2 * block + 1
    slots = min(5, MAXIMUM_VALID_TRIALS - start_valid_index + 1)
    trials = [
        _permutation_trial(
            start_valid_index, first_ordinal, cycle, coverage_binding
        ),
        _permutation_trial(
            start_valid_index + 1, first_ordinal + 1, cycle, coverage_binding
        ),
    ][:slots]
    successors = _successor_configurations_unchecked(
        parent_results, tried, count=max(1, slots - 2)
    ) if slots > 2 else ()
    for offset, configuration in enumerate(successors, start=2):
        trials.append(
            Trial(
                start_valid_index + offset,
                "association",
                "successor",
                "patience",
                False,
                configuration=configuration,
                cycle=cycle,
                coverage_binding=coverage_binding,
            )
        )
    # Ordinals 1..16 are reserved for the two mandatory falsifiers in each
    # block. Exhaustion replacements consume the disjoint range 17..40 without
    # gaps, based on the complete prior valid-result history.
    prior_exhaustion = sorted(
        result.trial.seed
        for result in parent_results
        if result.valid
        and result.trial.kind == "permutation"
        and result.trial.seed is not None
        and result.trial.seed >= 2026092837
    )
    if prior_exhaustion != list(range(2026092837, 2026092837 + len(prior_exhaustion))):
        raise ControllerError("prior exhaustion seed history is not contiguous")
    next_ordinal = 17 + len(prior_exhaustion)
    while len(trials) < slots:
        trials.append(
            _permutation_trial(
                start_valid_index + len(trials),
                next_ordinal,
                cycle,
                coverage_binding,
            )
        )
        next_ordinal += 1
    return tuple(trials)


def continuation_block(
    start_valid_index: int,
    parent_results: Sequence[TrialResult],
    tried: Iterable[Configuration] | None = None,
    *,
    coverage_length: int = COVERAGE_TRIALS,
    coverage_evidence: CoverageEvidence | None = None,
) -> tuple[Trial, ...]:
    """Construct one frozen continuation block from a validated history."""
    if (
        _validate_partial_result_history(
            parent_results,
            require_minimum=True,
            coverage_evidence=coverage_evidence,
        )
        != coverage_length
    ):
        raise ControllerError("continuation coverage evidence changed")
    bound_tried = _bind_tried_configurations(parent_results, tried)
    return _continuation_block_unchecked(
        start_valid_index,
        parent_results,
        bound_tried,
        coverage_length=coverage_length,
        coverage_binding=coverage_evidence.binding,
    )


def _real_tried_configurations(
    results: Sequence[TrialResult],
) -> tuple[Configuration, ...]:
    return tuple(
        result.trial.configuration
        for result in results
        if result.valid
        and result.trial.kind in {"anchor", "successor"}
        and result.trial.configuration is not None
    )


def _bind_tried_configurations(
    parent_results: Sequence[TrialResult],
    tried: Iterable[Configuration] | None,
) -> tuple[Configuration, ...]:
    """Derive tried configurations and reject any caller-supplied mismatch."""
    expected = _real_tried_configurations(parent_results)
    if tried is not None and tuple(tried) != expected:
        raise ControllerError("caller tried inventory differs from scored parent history")
    return expected


def _cycle_one_configurations(
    coverage_results: Sequence[TrialResult],
) -> tuple[Configuration, ...]:
    return _successor_configurations_unchecked(
        coverage_results,
        _real_tried_configurations(coverage_results),
        count=7,
    )


def _validate_compact_cycle_one_results(
    results: Sequence[TrialResult],
    coverage_length: int,
    coverage_binding: CoverageBinding,
) -> None:
    """Validate the coverage+7-successor score stream used to propose cycle two."""
    if len(results) != coverage_length + 7:
        raise ControllerError("compact cycle-one result stream has the wrong length")
    coverage_results = tuple(results[:coverage_length])
    cycle_one = _cycle_one_configurations(coverage_results)
    expected = _minimum_cycle_trials(
        cycle_one,
        cycle=1,
        start_valid_index=coverage_length + 1,
        coverage_binding=coverage_binding,
    )[4:11]
    actual = tuple(result.trial for result in results[coverage_length:])
    if actual != expected or any(not result.valid for result in results):
        raise ControllerError(
            "compact cycle-one scores do not have exact successor trial identities"
        )


def _extension_anchor_trials(
    configurations: Sequence[Configuration],
    *,
    start_valid_index: int,
    coverage_binding: CoverageBinding,
) -> tuple[Trial, ...]:
    return tuple(
        Trial(
            start_valid_index + offset,
            "association",
            "anchor",
            "extension_coverage",
            False,
            configuration,
            coverage_binding=coverage_binding,
        )
        for offset, configuration in enumerate(configurations)
    )


def _validate_available_coverage_prefix(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None,
) -> int | None:
    """Authenticate evidence-selected coverage, including incomplete prefixes."""
    trials = tuple(result.trial for result in results)
    core = _core_coverage_trials()
    core_length = min(len(trials), COVERAGE_TRIALS)
    if trials[:core_length] != core[:core_length]:
        raise ControllerError("available core-coverage prefix changed")
    if len(trials) < COVERAGE_TRIALS:
        if coverage_evidence is not None:
            expected = coverage_evidence.core_results[: len(results)]
            if tuple(
                _unbound_result_signature(result) for result in results
            ) != tuple(_unbound_result_signature(result) for result in expected):
                raise ControllerError(
                    "early core prefix conflicts with coverage evidence"
                )
        return None
    evidence = _bind_coverage_evidence(results, coverage_evidence)
    activation = evidence.activation
    if not activation.low_rank:
        return COVERAGE_TRIALS

    low_rank_configurations = low_rank_anchor_configurations()
    expected_low_rank = _extension_anchor_trials(
        low_rank_configurations,
        start_valid_index=COVERAGE_TRIALS + 1,
        coverage_binding=evidence.binding,
    )
    low_rank_length = min(len(trials) - COVERAGE_TRIALS, 8)
    if trials[COVERAGE_TRIALS : COVERAGE_TRIALS + low_rank_length] != (
        expected_low_rank[:low_rank_length]
    ):
        raise ControllerError("available low-rank coverage prefix changed")
    if len(trials) < COVERAGE_TRIALS + 8:
        return None
    low_rank_end = COVERAGE_TRIALS + 8
    if not activation.sparse_plus_low_rank:
        return low_rank_end

    selected_rank = best_low_rank_anchor_rank(
        results[COVERAGE_TRIALS:low_rank_end], evidence
    )
    edges = evidence.localized.edges
    hybrid_configurations = _optional_anchor_configurations(
        low_rank_triggered=True,
        hybrid_rank=selected_rank,
        hybrid_edges=edges,
    )[8:]
    expected_hybrid = _extension_anchor_trials(
        hybrid_configurations,
        start_valid_index=low_rank_end + 1,
        coverage_binding=evidence.binding,
    )
    hybrid_length = min(len(trials) - low_rank_end, 4)
    if trials[low_rank_end : low_rank_end + hybrid_length] != expected_hybrid[
        :hybrid_length
    ]:
        raise ControllerError("available hybrid coverage is not score-bound")
    if len(trials) < low_rank_end + 4:
        return None
    return low_rank_end + 4


def _validate_adaptive_result_prefix(
    results: Sequence[TrialResult],
    coverage_length: int,
    coverage_binding: CoverageBinding,
) -> None:
    """Authenticate every available adaptive-cycle and continuation trial."""
    coverage_results = tuple(results[:coverage_length])
    tail_length = len(results) - coverage_length
    cycle_one = _cycle_one_configurations(coverage_results)
    expected_one = _minimum_cycle_trials(
        cycle_one,
        cycle=1,
        start_valid_index=coverage_length + 1,
        coverage_binding=coverage_binding,
    )
    first_length = min(tail_length, 12)
    if tuple(
        result.trial
        for result in results[coverage_length : coverage_length + first_length]
    ) != expected_one[:first_length]:
        raise ControllerError("cycle-one trial identity or order changed")
    if tail_length < 12:
        return

    before_two = tuple(results[: coverage_length + 12])
    cycle_two = _successor_configurations_unchecked(
        before_two,
        _real_tried_configurations(before_two),
        count=7,
    )
    expected_two = _minimum_cycle_trials(
        cycle_two,
        cycle=2,
        start_valid_index=coverage_length + 13,
        coverage_binding=coverage_binding,
    )
    second_length = min(tail_length - 12, 12)
    if tuple(
        result.trial
        for result in results[
            coverage_length + 12 : coverage_length + 12 + second_length
        ]
    ) != expected_two[:second_length]:
        raise ControllerError("cycle-two trial identity or order changed")
    if tail_length < 24:
        return

    offset = coverage_length + 24
    while offset < len(results):
        parent_results = tuple(results[:offset])
        expected = _continuation_block_unchecked(
            offset + 1,
            parent_results,
            _real_tried_configurations(parent_results),
            coverage_length=coverage_length,
            coverage_binding=coverage_binding,
        )
        actual = tuple(
            result.trial for result in results[offset : offset + len(expected)]
        )
        if actual != expected[: len(actual)]:
            raise ControllerError("continuation block differs from frozen replay")
        offset += len(actual)


def validate_available_result_prefix(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None = None,
) -> int | None:
    """Authenticate every nonempty available valid-result history prefix."""
    if not results:
        return None
    if any(not result.valid for result in results):
        raise ControllerError("available result prefix contains invalid trials")
    coverage_length = _validate_available_coverage_prefix(
        results, coverage_evidence
    )
    if coverage_length is None:
        return None
    assert coverage_evidence is not None
    _validate_adaptive_result_prefix(
        results, coverage_length, coverage_evidence.binding
    )
    return coverage_length


def _validate_partial_result_history(
    results: Sequence[TrialResult],
    *,
    require_minimum: bool = False,
    coverage_evidence: CoverageEvidence | None = None,
) -> int:
    """Require completed coverage after authenticating the available prefix."""
    coverage_length = validate_available_result_prefix(results, coverage_evidence)
    if coverage_length is None:
        raise ControllerError("available result history has incomplete coverage")
    if require_minimum and len(results) < coverage_length + 24:
        raise ControllerError("minimum history lacks two complete cycles")
    return coverage_length


def _validate_cycle_successors(
    results: Sequence[TrialResult],
    through_cycle: int,
    coverage_evidence: CoverageEvidence | None,
) -> None:
    if through_cycle not in {1, 2}:
        raise ControllerError("cycle validation is defined only for one or two")
    coverage_length = _validate_partial_result_history(
        results, coverage_evidence=coverage_evidence
    )
    required_length = coverage_length + 12 * through_cycle
    if len(results) < required_length:
        raise ControllerError("successor cycle is incomplete")


def validate_result_history(
    results: Sequence[TrialResult],
    *,
    coverage_evidence: CoverageEvidence | None = None,
) -> None:
    """Validate the complete score-dependent controller history."""
    if not results or any(not result.valid for result in results):
        raise ControllerError("valid-result history cannot contain invalid attempts")
    trials = tuple(result.trial for result in results)
    validate_schedule(
        trials,
        result_history=results,
        coverage_evidence=coverage_evidence,
    )
    _validate_partial_result_history(
        results,
        require_minimum=True,
        coverage_evidence=coverage_evidence,
    )


def patience_streak(
    results: Sequence[TrialResult],
    coverage_evidence: CoverageEvidence | None = None,
) -> int:
    valid = [result for result in results if result.valid]
    coverage_length = validate_available_result_prefix(valid, coverage_evidence)
    if coverage_length is None:
        return 0
    branch_minimum = coverage_length + 24
    if len(valid) <= branch_minimum:
        return 0
    history = valid[:branch_minimum]
    incumbent = choose_incumbent(history)
    streak = 0
    for result in valid[branch_minimum:]:
        streak += 1
        if (
            result.trial.role == "association"
            and not result.trial.falsifier
            and result.trial.kind == "successor"
            and result.gain_bits_per_cell
            >= incumbent.gain_bits_per_cell + PATIENCE_GAIN
        ):
            incumbent = result
            streak = 0
    return streak


def stop_decision(
    results: Sequence[TrialResult],
    invalid_retries: int,
    coverage_evidence: CoverageEvidence | None = None,
) -> str:
    valid = [result for result in results if result.valid]
    coverage_length = validate_available_result_prefix(valid, coverage_evidence)
    # ``MAX_INVALID_RETRIES`` is the number of failed retry attempts the
    # frozen controller permits, not the first forbidden retry ordinal.  Stop
    # as soon as all 12 have been consumed; allowing a thirteenth attempt
    # would exceed the prospective search budget.
    if invalid_retries < 0 or invalid_retries >= MAX_INVALID_RETRIES:
        return "technical_stop_invalid_retry_ceiling"
    if len(valid) < MINIMUM_VALID_TRIALS:
        return "continue_minimum_not_met"
    if coverage_length is None:
        raise ControllerError("minimum trial count reached before coverage completed")
    if len(valid) < coverage_length + 24:
        return "continue_branch_minimum_not_met"
    validate_result_history(valid, coverage_evidence=coverage_evidence)
    if len(valid) >= MAXIMUM_VALID_TRIALS:
        return "incomplete_budget_stop"
    if patience_streak(valid, coverage_evidence) >= PATIENCE:
        return "ordinary_patience_stop"
    return "continue"


def null_replay_seeds() -> tuple[int, ...]:
    if len(NULL_SEEDS) != 99 or len(set(NULL_SEEDS)) != 99:
        raise ControllerError("null replay seed inventory changed")
    return NULL_SEEDS


def new_null_replay(replicate_index: int) -> NullReplayState:
    if not 1 <= replicate_index <= len(NULL_SEEDS):
        raise ControllerError("null replicate index is outside 1..99")
    return NullReplayState(replicate_index, NULL_SEEDS[replicate_index - 1])


def seedsequence_substream(
    replicate_index: int,
    stage_name: str,
    fold_ordinal: int,
    trial_ordinal: int,
    purpose: str,
) -> tuple[int, int, int, int]:
    """Derive the frozen integer-only SeedSequence substream state."""
    try:
        import numpy as np
    except ImportError as error:
        raise ControllerError("NumPy is required only for SeedSequence replay") from error
    replay = new_null_replay(replicate_index)
    if stage_name not in STAGE_CODES or purpose not in PURPOSE_CODES:
        raise ControllerError("unknown null-replay stage or purpose")
    if not 0 <= fold_ordinal <= 79 or not 0 <= trial_ordinal <= MAXIMUM_VALID_TRIALS:
        raise ControllerError("null-replay fold/trial ordinal is outside its frozen range")
    sequence = np.random.SeedSequence(
        replay.seed,
        spawn_key=(
            STAGE_CODES[stage_name],
            int(fold_ordinal),
            int(trial_ordinal),
            PURPOSE_CODES[purpose],
        ),
    )
    return tuple(int(value) for value in sequence.generate_state(4, dtype=np.uint32))


def run_target_free_replay_conformance(
    replicate_index: int, optional_branch: str
) -> Mapping[str, object]:
    """Exercise one complete empty-history controller replay on synthetic scores.

    This is a grammar conformance fixture, not a scientific null replicate. It
    traverses trigger binding, coverage, both decisions, falsifiers, adaptive
    successors, patience, the pair selector, and every RNG-purpose route.
    """
    if optional_branch not in {"none", "low_rank", "hybrid"}:
        raise ControllerError("unknown replay conformance branch")
    replay = new_null_replay(replicate_index)
    if replay.trial_history:
        raise ControllerError("replay fixture did not start empty")

    if optional_branch == "none":
        diagnostic = OptionalDiagnostics(0.39, 16.0, 0.80, 9)
        contrasts = (0.0,) * PAIR_COUNT
    elif optional_branch == "low_rank":
        diagnostic = OptionalDiagnostics(0.40, 16.0, 0.80, 9)
        contrasts = (1.0,) * PAIR_COUNT
    else:
        diagnostic = OptionalDiagnostics(0.40, 16.0, 0.80, 9)
        raw = [0.0] * PAIR_COUNT
        raw[0], raw[1], raw[2], raw[3] = 0.4, 0.3, 0.2, 0.1
        contrasts = tuple(raw)

    def anchor_gain(trial: Trial) -> float:
        configuration = trial.configuration
        if (
            configuration is not None
            and configuration.family == "low_rank"
            and configuration.rank == 2
            and configuration.rho == 0.1
        ):
            return 0.03
        return 0.01 - 0.00001 * trial.valid_index

    unbound_core_results = tuple(
        TrialResult(trial, anchor_gain(trial), trial.valid_index % 5)
        for trial in coverage_trials()
    )
    coverage_evidence = bind_coverage_evidence(
        unbound_core_results, diagnostic, tuple(contrasts)
    )
    core_results = coverage_evidence.core_results
    staged_coverage = coverage_trials(coverage_evidence=coverage_evidence)
    low_results: tuple[TrialResult, ...] = ()
    if coverage_evidence.activation.sparse_plus_low_rank:
        low_results = tuple(
            TrialResult(
                trial,
                anchor_gain(trial),
                4,
            )
            for trial in staged_coverage[COVERAGE_TRIALS:]
        )
    coverage = coverage_trials(
        coverage_evidence=coverage_evidence,
        low_rank_anchor_results=low_results,
    )
    coverage_results = (
        *core_results,
        *(
            TrialResult(trial, anchor_gain(trial), trial.valid_index % 5)
            for trial in coverage[COVERAGE_TRIALS:]
        ),
    )
    cycle_one = successor_configurations(
        coverage_results,
        _real_tried_configurations(coverage_results),
        count=7,
        coverage_evidence=coverage_evidence,
    )
    coverage_length = len(coverage)
    cycle_one_results = tuple(
        TrialResult(
            Trial(
                coverage_length + 5 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                configuration,
                cycle=1,
                coverage_binding=coverage_evidence.binding,
            ),
            0.02 - 0.0001 * index,
            4,
        )
        for index, configuration in enumerate(cycle_one)
    )
    cycle_two = successor_configurations(
        (*coverage_results, *cycle_one_results),
        (*_real_tried_configurations(coverage_results), *cycle_one),
        count=7,
        coverage_evidence=coverage_evidence,
    )
    schedule = minimum_schedule(
        cycle_one,
        cycle_two,
        coverage_results=coverage_results,
        cycle_one_results=cycle_one_results,
        coverage_evidence=coverage_evidence,
    )
    results: list[TrialResult] = list(coverage_results)
    for trial in schedule[coverage_length:]:
        if trial.kind == "successor" and trial.cycle == 1:
            gain = 0.02 - 0.0001 * cycle_one.index(trial.configuration)
        else:
            gain = 0.01 if trial.role == "association" and not trial.falsifier else 0.0
        results.append(TrialResult(trial, gain, 4))
    first_decision = incumbent_challenger_decision(
        results[: coverage_length + 12],
        1,
        coverage_evidence=coverage_evidence,
    )
    second_decision = incumbent_challenger_decision(
        results, 2, coverage_evidence=coverage_evidence
    )

    target_length = min(MAXIMUM_VALID_TRIALS, len(schedule) + PATIENCE)
    while len(results) < target_length:
        block = continuation_block(
            len(results) + 1,
            results,
            _real_tried_configurations(results),
            coverage_length=coverage_length,
            coverage_evidence=coverage_evidence,
        )
        for trial in block:
            if len(results) >= target_length:
                break
            gain = 0.01 if trial.kind == "successor" else 0.0
            results.append(TrialResult(trial, gain, 4))
    validate_result_history(results, coverage_evidence=coverage_evidence)
    disposition = stop_decision(results, 0, coverage_evidence)
    if target_length < MAXIMUM_VALID_TRIALS and disposition != "ordinary_patience_stop":
        raise ControllerError("replay conformance fixture missed its patience stop")

    rows = tuple(
        make_pair_contrast_row(cell_key, group, 2, (0.0,) * PAIR_COUNT)
        for cell_key, group in FROZEN_CELL_INVENTORY.entries
    )
    selected = select_pair_contrast(rows)

    fold_ordinals = (0, 1, 79)
    substreams: dict[
        tuple[str, int, int, str], tuple[int, int, int, int]
    ] = {}
    exercised_routes: set[tuple[str, str]] = set()

    def exercise_route(stage: str, trial_ordinal: int, purpose: str) -> None:
        exercised_routes.add((stage, purpose))
        for fold_ordinal in fold_ordinals:
            key = (stage, fold_ordinal, trial_ordinal, purpose)
            first = seedsequence_substream(
                replicate_index, stage, fold_ordinal, trial_ordinal, purpose
            )
            repeat = seedsequence_substream(
                replicate_index, stage, fold_ordinal, trial_ordinal, purpose
            )
            if first != repeat:
                raise ControllerError("replay conformance substream is nondeterministic")
            if key in substreams:
                raise ControllerError("replay conformance repeated a route key")
            substreams[key] = first

    exercise_route("M0_refit", 0, "model_fit_or_initialization")
    exercise_route("M1_refit", 0, "model_fit_or_initialization")
    exercise_route("M1_refit", 0, "fitted_M1_categorical_sampling")
    exercise_route("optional_diagnostic", 0, "deterministic_tie_auxiliary")
    exercise_route("candidate_selector", 0, "deterministic_tie_auxiliary")
    for result in results:
        trial = result.trial
        purpose = (
            "Rademacher_design"
            if trial.kind == "rademacher_noise"
            else "whole_set_permutation"
            if trial.kind == "permutation"
            else "model_fit_or_initialization"
        )
        exercise_route("M2_fit", trial.valid_index, purpose)
    if exercised_routes != set(REPLAY_PURPOSE_ROUTES):
        raise ControllerError("replay conformance missed a declared RNG purpose route")
    if len(set(substreams.values())) != len(substreams):
        raise ControllerError("replay conformance substreams collided")
    return {
        "replicate_index": replicate_index,
        "started_empty": True,
        "optional_branch": optional_branch,
        "coverage_length": coverage_length,
        "valid_trials": len(results),
        "decision_1_winner": first_decision.winner.trial.configuration.key,
        "decision_2_winner": second_decision.winner.trial.configuration.key,
        "disposition": disposition,
        "falsifiers": sum(result.trial.falsifier for result in results),
        "selected_pair_index": selected["pair_index"],
        "substreams_derived": len(substreams),
        "fold_ordinals_exercised": fold_ordinals,
        "purpose_routes_exercised": REPLAY_PURPOSE_ROUTES,
        "substream_determinism_verified": True,
        "M0_refit_required": replay.M0_refit_required,
        "M1_refit_required": replay.M1_refit_required,
        "all_required_branches_replayed": all(
            (
                replay.optional_trigger_replay_required,
                replay.coverage_replay_required,
                replay.falsifier_replay_required,
                replay.candidate_selector_replay_required,
            )
        ),
    }


def select_pair_contrast(
    rows: Sequence[PairContrastRow],
) -> Mapping[str, object]:
    """Select the exploratory identified event-probability contrast.

    The controller owns the prospectively frozen target-blind identity/group
    inventory; callers supply only all 194 cellwise cross-fitted
    Pr_M2(A&B recorded)-Pr_M1(A&B recorded) vectors. Each group receives equal
    weight, irrespective of its cell count; K=0, K=1, and K=16 contrasts are
    exact zero. Pair identity and exact-tie order are internal frozen constants.
    """
    if (
        type(FROZEN_CELL_INVENTORY) is not FrozenCellInventory
        or FROZEN_CELL_INVENTORY.entries
        != EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES
    ):
        raise ControllerError("in-memory frozen cell inventory changed after loading")
    submitted = tuple(rows)
    if len(submitted) != sum(GROUP_SIZES.values()):
        raise ControllerError("pair selection requires the exact 194-cell denominator")
    validated = tuple(
        _require_canonical_pair_contrast_row(row) for row in submitted
    )
    if len({row.cell_key for row in validated}) != len(validated):
        raise ControllerError("pair selection requires 194 unique cross-fitted cells")
    observed_inventory = tuple(
        sorted((row.cell_key, row.group) for row in validated)
    )
    if observed_inventory != EXPECTED_FROZEN_CELL_INVENTORY_ENTRIES:
        raise ControllerError(
            "pair selection rows do not match the frozen target-blind cell inventory"
        )
    by_group: dict[str, list[PairContrastRow]] = {group: [] for group in GROUP_SIZES}
    for row in validated:
        by_group[row.group].append(row)
    if {group: len(values) for group, values in by_group.items()} != GROUP_SIZES:
        raise ControllerError("pair selection group-size vector changed")
    group_means = []
    for group in GROUP_SIZES:
        values = by_group[group]
        group_means.append(
            tuple(
                math.fsum(row.values[index] for row in values) / len(values)
                for index in range(PAIR_COUNT)
            )
        )
    means = tuple(
        math.fsum(group[index] for group in group_means) / 12.0
        for index in range(PAIR_COUNT)
    )
    # max() retains the first index on an exact tie, which is the frozen pair order.
    selected = max(range(PAIR_COUNT), key=lambda index: abs(means[index]))
    value = means[selected]
    direction = "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
    return {
        "pair": PAIR_NAMES[selected],
        "pair_index": selected,
        "equal_group_probability_contrast": value,
        "direction": direction,
        "selection_rule": "maximum_absolute_then_lexicographic",
        "analysis_label": "exploratory_recorded_positive_event",
        "biological_promotion": False,
    }


def run_self_test() -> None:
    core = coverage_trials()
    unbound_coverage_results = tuple(
        TrialResult(trial, 0.0001 * index, index % 5)
        for index, trial in enumerate(core, start=1)
    )
    coverage_evidence = bind_coverage_evidence(
        unbound_coverage_results,
        OptionalDiagnostics(0.39, 16.0, 0.80, 9),
        (0.0,) * PAIR_COUNT,
    )
    coverage_results = coverage_evidence.core_results
    cycle_one = successor_configurations(
        coverage_results,
        anchor_configurations(),
        count=7,
        coverage_evidence=coverage_evidence,
    )
    cycle_one_results = [
        TrialResult(
            Trial(
                21 + index,
                "association",
                "successor",
                "postcoverage",
                False,
                value,
                cycle=1,
                coverage_binding=coverage_evidence.binding,
            ),
            0.003 - 0.00001 * index,
            3,
        )
        for index, value in enumerate(cycle_one)
    ]
    tried = (*anchor_configurations(), *cycle_one)
    cycle_two = successor_configurations(
        (*coverage_results, *cycle_one_results),
        tried,
        count=7,
        coverage_evidence=coverage_evidence,
    )
    schedule = minimum_schedule(
        cycle_one,
        cycle_two,
        coverage_results=coverage_results,
        cycle_one_results=cycle_one_results,
        coverage_evidence=coverage_evidence,
    )
    assert len(schedule) == 40
    assert sum(trial.role == "reference" for trial in schedule) == 8
    assert sum(trial.role == "association" for trial in schedule) == 32
    assert sum(trial.falsifier for trial in schedule[16:]) == 10
    assert len(null_replay_seeds()) == 99


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.self_test:
        raise ControllerError("this pure controller CLI only supports --self-test")
    run_self_test()
    print("v3 search controller self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
