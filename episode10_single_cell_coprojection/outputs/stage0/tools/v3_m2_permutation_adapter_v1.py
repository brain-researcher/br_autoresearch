#!/usr/bin/env python3
"""Deterministic whole-set permutation primitive for EP10 v3 M2.

This module implements only the outcome-permutation mapping frozen in
``v3_m2_numerical_falsifier_clarification_v4.yaml``.  It accepts an already
materialized, canonical 194-cell in-memory M1 view; it has no data loader and
does not fit M1 or M2.  In particular, importing or calling this module cannot
open observed, final, unassigned, or morphology files.

The mapping shuffles complete ``observed_index`` values within
``(conservative_group, segment_unmapped, K_i_obs)`` strata.  Recipient
metadata remain unchanged.  The returned :class:`PermutationAssignment` is a
read-only description that can be reused by every fold and M1 path belonging
to the same controller-trial objective.  Production execution authority and
one-shot capabilities remain responsibilities of a later named adapter.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_ROWS = 194
UINT64_MODULUS = 1 << 64
OBSERVED_ROUTE = "observed_or_target_free_qualification"
NULL_REPLAY_ROUTE = "null_replay"
PERMUTATION_STAGE_CODE = 3
NONAPPLICABLE_FOLD_ORDINAL = 0
PERMUTATION_PURPOSE_CODE = 4
SEGMENT_ORDER = (0, 1)
K_ORDER = tuple(range(m1.PANEL_SIZE + 1))
STRATUM_KEYS = tuple(
    (group, segment_unmapped, k)
    for group in m1.GROUP_ORDER
    for segment_unmapped in SEGMENT_ORDER
    for k in K_ORDER
)
FROZEN_MEMBERSHIP_INVENTORY = controller.FROZEN_CELL_INVENTORY.entries


class PermutationAdapterError(m1.M1Error):
    """A frozen whole-set permutation invariant failed closed."""


def _require_static_contract_alignment() -> None:
    if (
        tuple(controller.GROUP_SIZES) != tuple(m1.GROUP_ORDER)
        or controller.GROUP_SIZES != m1.GROUP_SIZES
        or len(FROZEN_MEMBERSHIP_INVENTORY) != EXPECTED_ROWS
        or sum(controller.GROUP_SIZES.values()) != EXPECTED_ROWS
        or controller.STAGE_CODES.get("M2_fit") != PERMUTATION_STAGE_CODE
        or controller.PURPOSE_CODES.get("whole_set_permutation")
        != PERMUTATION_PURPOSE_CODE
    ):
        raise PermutationAdapterError(
            "controller, M1, and frozen permutation constants are inconsistent"
        )


_require_static_contract_alignment()


def _is_integer(value: object) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )


def _canonical_cells(
    cells: Sequence[m1.Cell],
) -> tuple[
    tuple[m1.Cell, ...],
    tuple[tuple[str, str], ...],
    tuple[tuple[str, int, int], ...],
    tuple[int, ...],
]:
    """Validate and bind the exact canonical 194-row execution-cache order."""
    rows = tuple(cells)
    if len(rows) != EXPECTED_ROWS or any(type(cell) is not m1.Cell for cell in rows):
        raise PermutationAdapterError(
            "permutation input must contain exactly 194 canonical M1 Cell rows"
        )
    if any(
        type(cell.identifier) is not str or type(cell.group) is not str
        for cell in rows
    ):
        raise PermutationAdapterError("cell identity/group fields are not strings")
    execution_cache_inventory = tuple(
        (cell.identifier, cell.group) for cell in rows
    )
    if tuple(sorted(execution_cache_inventory)) != FROZEN_MEMBERSHIP_INVENTORY:
        raise PermutationAdapterError(
            "permutation rows do not have the exact frozen 194-cell membership"
        )

    recipient_strata: list[tuple[str, int, int]] = []
    observed_indexes: list[int] = []
    for cell in rows:
        if not _is_integer(cell.segment_unmapped) or int(cell.segment_unmapped) not in (
            0,
            1,
        ):
            raise PermutationAdapterError("segment_unmapped is not exact false/true")
        if not _is_integer(cell.k) or not 0 <= int(cell.k) <= m1.PANEL_SIZE:
            raise PermutationAdapterError("K_i_obs is outside the frozen 0..16 range")
        k = int(cell.k)
        if not _is_integer(cell.observed_index):
            raise PermutationAdapterError("observed_index is not an integer")
        observed_index = int(cell.observed_index)
        if not 0 <= observed_index < math.comb(m1.PANEL_SIZE, k):
            raise PermutationAdapterError(
                "observed_index is outside the exact fixed-K candidate order"
            )
        recipient_strata.append((cell.group, int(cell.segment_unmapped), k))
        observed_indexes.append(observed_index)
    return (
        rows,
        execution_cache_inventory,
        tuple(recipient_strata),
        tuple(observed_indexes),
    )


def _require_trial(trial: controller.Trial) -> controller.Trial:
    """Require a structurally valid controller permutation trial.

    Exact schedule/history authentication deliberately belongs to the
    controller/session layer.  This primitive binds the supplied immutable
    trial identity and never invents or changes its seed.
    """
    if (
        type(trial) is not controller.Trial
        or trial.kind != "permutation"
        or trial.role != "association"
        or trial.falsifier is not True
        or trial.configuration is None
        or trial.cycle is None
    ):
        raise PermutationAdapterError(
            "whole-set mapping requires a controller permutation falsifier trial"
        )
    if (
        not _is_integer(trial.valid_index)
        or not 1 <= int(trial.valid_index) <= controller.MAXIMUM_VALID_TRIALS
    ):
        raise PermutationAdapterError("permutation trial index is outside 1..80")
    if (
        not _is_integer(trial.seed)
        or not 0 <= int(trial.seed) < UINT64_MODULUS
    ):
        raise PermutationAdapterError(
            "permutation trial seed must be one unsigned 64-bit integer"
        )
    return trial


@dataclass(frozen=True)
class PermutationStratumAssignment:
    """Recipient and shuffled donor rows for one frozen ordered stratum."""

    group: str
    segment_unmapped: int
    k: int
    recipient_indexes: tuple[int, ...]
    donor_indexes: tuple[int, ...]
    raw_uint64_draws: int

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.group, self.segment_unmapped, self.k)

    def __post_init__(self) -> None:
        if self.key not in STRATUM_KEYS:
            raise PermutationAdapterError("permutation stratum key is not frozen")
        if (
            type(self.recipient_indexes) is not tuple
            or type(self.donor_indexes) is not tuple
            or any(not _is_integer(index) for index in self.recipient_indexes)
            or any(not _is_integer(index) for index in self.donor_indexes)
            or tuple(int(index) for index in self.recipient_indexes)
            != self.recipient_indexes
            or tuple(int(index) for index in self.donor_indexes) != self.donor_indexes
        ):
            raise PermutationAdapterError("permutation row indexes must be integer tuples")
        if tuple(sorted(self.recipient_indexes)) != self.recipient_indexes or len(
            set(self.recipient_indexes)
        ) != len(self.recipient_indexes):
            raise PermutationAdapterError(
                "stratum recipients are not unique canonical-order rows"
            )
        if tuple(sorted(self.donor_indexes)) != tuple(sorted(self.recipient_indexes)):
            raise PermutationAdapterError(
                "stratum donors are not an exact recipient-row bijection"
            )
        if (
            not _is_integer(self.raw_uint64_draws)
            or int(self.raw_uint64_draws) < 0
        ):
            raise PermutationAdapterError("raw draw count is malformed")
        minimum_draws = max(0, len(self.recipient_indexes) - 1)
        if len(self.recipient_indexes) <= 1 and self.raw_uint64_draws != 0:
            raise PermutationAdapterError("empty or singleton stratum consumed RNG")
        if len(self.recipient_indexes) > 1 and self.raw_uint64_draws < minimum_draws:
            raise PermutationAdapterError(
                "Fisher-Yates stratum consumed fewer than one draw per iteration"
            )


@dataclass(frozen=True)
class PermutationAssignment:
    """Immutable provenance and donor mapping for one controller objective."""

    rng_route: str
    scheduled_trial_valid_index: int
    scheduled_trial_seed: int
    scheduled_configuration_key: str
    scheduled_phase: str
    scheduled_cycle: int
    null_replicate_index: int | None
    null_replicate_seed: int | None
    seedsequence_spawn_key: tuple[int, int, int, int] | None
    execution_cache_inventory: tuple[tuple[str, str], ...]
    recipient_strata: tuple[tuple[str, int, int], ...]
    source_observed_indexes: tuple[int, ...]
    strata: tuple[PermutationStratumAssignment, ...]
    recipient_to_donor: tuple[int, ...]
    raw_uint64_draws: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.rng_route not in {OBSERVED_ROUTE, NULL_REPLAY_ROUTE}:
            raise PermutationAdapterError("unknown permutation RNG route")
        if (
            not _is_integer(self.scheduled_trial_valid_index)
            or not 1
            <= int(self.scheduled_trial_valid_index)
            <= controller.MAXIMUM_VALID_TRIALS
            or not _is_integer(self.scheduled_trial_seed)
            or not 0 <= int(self.scheduled_trial_seed) < UINT64_MODULUS
            or not isinstance(self.scheduled_configuration_key, str)
            or not self.scheduled_configuration_key
            or self.scheduled_phase not in {"postcoverage", "patience"}
            or not _is_integer(self.scheduled_cycle)
            or int(self.scheduled_cycle) < 1
        ):
            raise PermutationAdapterError("scheduled permutation provenance is malformed")
        expected_phase = (
            "postcoverage" if int(self.scheduled_cycle) in {1, 2} else "patience"
        )
        if self.scheduled_phase != expected_phase:
            raise PermutationAdapterError(
                "scheduled permutation phase and cycle are inconsistent"
            )
        expected_spawn_key = (
            PERMUTATION_STAGE_CODE,
            NONAPPLICABLE_FOLD_ORDINAL,
            int(self.scheduled_trial_valid_index),
            PERMUTATION_PURPOSE_CODE,
        )
        if self.rng_route == OBSERVED_ROUTE:
            if (
                self.null_replicate_index is not None
                or self.null_replicate_seed is not None
                or self.seedsequence_spawn_key is not None
            ):
                raise PermutationAdapterError(
                    "direct trial-seed mapping carries null-replay provenance"
                )
        else:
            if (
                not _is_integer(self.null_replicate_index)
                or not 1
                <= int(self.null_replicate_index)
                <= len(controller.NULL_SEEDS)
            ):
                raise PermutationAdapterError("null replicate index is outside 1..99")
            replay = controller.new_null_replay(int(self.null_replicate_index))
            if (
                self.null_replicate_seed != replay.seed
                or self.seedsequence_spawn_key != expected_spawn_key
            ):
                raise PermutationAdapterError(
                    "null permutation SeedSequence provenance changed"
                )

        if (
            type(self.execution_cache_inventory) is not tuple
            or len(self.execution_cache_inventory) != EXPECTED_ROWS
            or any(
                type(entry) is not tuple
                or len(entry) != 2
                or type(entry[0]) is not str
                or type(entry[1]) is not str
                for entry in self.execution_cache_inventory
            )
        ):
            raise PermutationAdapterError(
                "execution-cache inventory identity/order is malformed"
            )
        if tuple(sorted(self.execution_cache_inventory)) != FROZEN_MEMBERSHIP_INVENTORY:
            raise PermutationAdapterError(
                "execution-cache inventory membership is not the frozen 194 cells"
            )
        if (
            type(self.recipient_strata) is not tuple
            or len(self.recipient_strata) != EXPECTED_ROWS
            or type(self.source_observed_indexes) is not tuple
            or len(self.source_observed_indexes) != EXPECTED_ROWS
            or type(self.recipient_to_donor) is not tuple
            or len(self.recipient_to_donor) != EXPECTED_ROWS
            or type(self.strata) is not tuple
            or len(self.strata) != len(STRATUM_KEYS)
        ):
            raise PermutationAdapterError(
                "permutation assignment is not bound to the canonical 194-row view"
            )
        for inventory_entry, key, observed_index in zip(
            self.execution_cache_inventory,
            self.recipient_strata,
            self.source_observed_indexes,
            strict=True,
        ):
            if (
                key not in STRATUM_KEYS
                or inventory_entry[1] != key[0]
                or not _is_integer(observed_index)
            ):
                raise PermutationAdapterError(
                    "permutation source stratum or observed index is malformed"
                )
            k = key[2]
            if not 0 <= int(observed_index) < math.comb(m1.PANEL_SIZE, k):
                raise PermutationAdapterError(
                    "permutation source observed index is outside its fixed-K basis"
                )

        rebuilt = [-1] * EXPECTED_ROWS
        total_draws = 0
        for expected_key, record in zip(STRATUM_KEYS, self.strata, strict=True):
            if type(record) is not PermutationStratumAssignment:
                raise PermutationAdapterError("stratum record type changed")
            record.__post_init__()
            expected_recipients = tuple(
                index
                for index, key in enumerate(self.recipient_strata)
                if key == expected_key
            )
            if record.key != expected_key or record.recipient_indexes != expected_recipients:
                raise PermutationAdapterError(
                    "stratum order or canonical recipient membership changed"
                )
            for recipient, donor in zip(
                record.recipient_indexes, record.donor_indexes, strict=True
            ):
                if rebuilt[recipient] != -1:
                    raise PermutationAdapterError(
                        "canonical recipient appears in multiple strata"
                    )
                rebuilt[recipient] = donor
            total_draws += int(record.raw_uint64_draws)
        if (
            tuple(rebuilt) != self.recipient_to_donor
            or any(index < 0 for index in rebuilt)
            or not _is_integer(self.raw_uint64_draws)
            or total_draws != int(self.raw_uint64_draws)
        ):
            raise PermutationAdapterError(
                "global recipient-to-donor mapping or raw draw accounting changed"
            )
        for recipient, donor in enumerate(self.recipient_to_donor):
            if (
                not _is_integer(donor)
                or not 0 <= int(donor) < EXPECTED_ROWS
                or self.recipient_strata[recipient]
                != self.recipient_strata[int(donor)]
            ):
                raise PermutationAdapterError(
                    "permutation donor crosses a frozen group/segment/K stratum"
                )
        expected_mapping, expected_donors, expected_draws, expected_total_draws = (
            _replay_assignment_from_provenance(self)
        )
        if (
            expected_mapping != self.recipient_to_donor
            or expected_total_draws != self.raw_uint64_draws
        ):
            raise PermutationAdapterError(
                "permutation mapping does not replay from its claimed RNG provenance"
            )
        for record, donors, draws in zip(
            self.strata, expected_donors, expected_draws, strict=True
        ):
            if record.donor_indexes != donors or record.raw_uint64_draws != draws:
                raise PermutationAdapterError(
                    "permutation stratum does not replay from its claimed RNG provenance"
                )

    def apply(self, cells: Sequence[m1.Cell]) -> tuple[m1.Cell, ...]:
        """Copy each donor's complete observed set into its recipient row."""
        return apply_permutation(cells, self)


def _bounded_raw_uint64(bit_generator: Any, upper: int) -> tuple[int, int]:
    """Return an unbiased integer in ``range(upper)`` using raw PCG64 words."""
    if isinstance(upper, bool) or not isinstance(upper, int) or upper < 1:
        raise PermutationAdapterError("bounded raw draw requires a positive integer")
    limit = (UINT64_MODULUS // upper) * upper
    draws = 0
    while True:
        try:
            raw_value = bit_generator.random_raw()
        except (AttributeError, TypeError, ValueError) as error:
            raise PermutationAdapterError("bit generator cannot provide raw uint64") from error
        if not _is_integer(raw_value):
            raise PermutationAdapterError("bit generator returned a noninteger raw word")
        value = int(raw_value)
        if not 0 <= value < UINT64_MODULUS:
            raise PermutationAdapterError("bit generator returned a non-uint64 raw word")
        draws += 1
        if value < limit:
            return value % upper, draws


def _fisher_yates(
    indexes: tuple[int, ...], bit_generator: Any
) -> tuple[tuple[int, ...], int]:
    """Apply the exact descending rejection-sampled Fisher-Yates shuffle."""
    shuffled = list(indexes)
    raw_draws = 0
    for index in range(len(shuffled) - 1, 0, -1):
        partner, draws = _bounded_raw_uint64(bit_generator, index + 1)
        raw_draws += draws
        shuffled[index], shuffled[partner] = shuffled[partner], shuffled[index]
    return tuple(shuffled), raw_draws


def _replay_assignment_from_provenance(
    assignment: PermutationAssignment,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    int,
]:
    """Deterministically reconstruct a mapping from its claimed seed route."""
    if assignment.rng_route == OBSERVED_ROUTE:
        bit_generator = np.random.PCG64(int(assignment.scheduled_trial_seed))
    elif assignment.rng_route == NULL_REPLAY_ROUTE:
        assert assignment.null_replicate_seed is not None
        assert assignment.seedsequence_spawn_key is not None
        sequence = np.random.SeedSequence(
            int(assignment.null_replicate_seed),
            spawn_key=assignment.seedsequence_spawn_key,
        )
        bit_generator = np.random.PCG64(sequence)
    else:  # Already rejected above; keep this helper fail-closed in isolation.
        raise PermutationAdapterError("unknown permutation RNG replay route")

    mapping = [-1] * EXPECTED_ROWS
    donors_by_stratum: list[tuple[int, ...]] = []
    draws_by_stratum: list[int] = []
    total_draws = 0
    for key in STRATUM_KEYS:
        recipients = tuple(
            index
            for index, recipient_key in enumerate(assignment.recipient_strata)
            if recipient_key == key
        )
        donors, draws = _fisher_yates(recipients, bit_generator)
        donors_by_stratum.append(donors)
        draws_by_stratum.append(draws)
        total_draws += draws
        for recipient, donor in zip(recipients, donors, strict=True):
            mapping[recipient] = donor
    if any(index < 0 for index in mapping):
        raise PermutationAdapterError("RNG replay did not assign every recipient")
    return (
        tuple(mapping),
        tuple(donors_by_stratum),
        tuple(draws_by_stratum),
        total_draws,
    )


def _build_assignment(
    rows: tuple[m1.Cell, ...],
    execution_cache_inventory: tuple[tuple[str, str], ...],
    recipient_strata: tuple[tuple[str, int, int], ...],
    source_observed_indexes: tuple[int, ...],
    trial: controller.Trial,
    *,
    rng_route: str,
    bit_generator: Any,
    null_replicate_index: int | None,
    null_replicate_seed: int | None,
    seedsequence_spawn_key: tuple[int, int, int, int] | None,
) -> PermutationAssignment:
    del rows  # validation already produced the immutable row bindings above
    recipient_to_donor = [-1] * EXPECTED_ROWS
    strata: list[PermutationStratumAssignment] = []
    total_draws = 0
    for group, segment_unmapped, k in STRATUM_KEYS:
        key = (group, segment_unmapped, k)
        recipients = tuple(
            index
            for index, recipient_key in enumerate(recipient_strata)
            if recipient_key == key
        )
        donors, draws = _fisher_yates(recipients, bit_generator)
        for recipient, donor in zip(recipients, donors, strict=True):
            recipient_to_donor[recipient] = donor
        strata.append(
            PermutationStratumAssignment(
                group=group,
                segment_unmapped=segment_unmapped,
                k=k,
                recipient_indexes=recipients,
                donor_indexes=donors,
                raw_uint64_draws=draws,
            )
        )
        total_draws += draws

    assert trial.configuration is not None and trial.cycle is not None
    assert trial.seed is not None
    return PermutationAssignment(
        rng_route=rng_route,
        scheduled_trial_valid_index=int(trial.valid_index),
        scheduled_trial_seed=int(trial.seed),
        scheduled_configuration_key=trial.configuration.key,
        scheduled_phase=trial.phase,
        scheduled_cycle=int(trial.cycle),
        null_replicate_index=null_replicate_index,
        null_replicate_seed=null_replicate_seed,
        seedsequence_spawn_key=seedsequence_spawn_key,
        execution_cache_inventory=execution_cache_inventory,
        recipient_strata=recipient_strata,
        source_observed_indexes=source_observed_indexes,
        strata=tuple(strata),
        recipient_to_donor=tuple(recipient_to_donor),
        raw_uint64_draws=total_draws,
    )


def build_observed_permutation_assignment(
    cells: Sequence[m1.Cell], trial: controller.Trial
) -> PermutationAssignment:
    """Build the direct-trial-seed observed/qualification mapping."""
    checked_trial = _require_trial(trial)
    rows, execution_inventory, recipient_strata, observed_indexes = _canonical_cells(
        cells
    )
    assert checked_trial.seed is not None
    bit_generator = np.random.PCG64(int(checked_trial.seed))
    return _build_assignment(
        rows,
        execution_inventory,
        recipient_strata,
        observed_indexes,
        checked_trial,
        rng_route=OBSERVED_ROUTE,
        bit_generator=bit_generator,
        null_replicate_index=None,
        null_replicate_seed=None,
        seedsequence_spawn_key=None,
    )


def build_null_permutation_assignment(
    cells: Sequence[m1.Cell],
    trial: controller.Trial,
    *,
    null_replicate_index: int,
) -> PermutationAssignment:
    """Build the frozen null-replay SeedSequence-substream mapping."""
    checked_trial = _require_trial(trial)
    if (
        not _is_integer(null_replicate_index)
        or not 1 <= int(null_replicate_index) <= len(controller.NULL_SEEDS)
    ):
        raise PermutationAdapterError("null replicate index is outside 1..99")
    rows, execution_inventory, recipient_strata, observed_indexes = _canonical_cells(
        cells
    )
    replay = controller.new_null_replay(int(null_replicate_index))
    spawn_key = (
        PERMUTATION_STAGE_CODE,
        NONAPPLICABLE_FOLD_ORDINAL,
        int(checked_trial.valid_index),
        PERMUTATION_PURPOSE_CODE,
    )
    sequence = np.random.SeedSequence(replay.seed, spawn_key=spawn_key)
    bit_generator = np.random.PCG64(sequence)
    return _build_assignment(
        rows,
        execution_inventory,
        recipient_strata,
        observed_indexes,
        checked_trial,
        rng_route=NULL_REPLAY_ROUTE,
        bit_generator=bit_generator,
        null_replicate_index=int(null_replicate_index),
        null_replicate_seed=int(replay.seed),
        seedsequence_spawn_key=spawn_key,
    )


def apply_permutation(
    cells: Sequence[m1.Cell], assignment: PermutationAssignment
) -> tuple[m1.Cell, ...]:
    """Return permuted cells while retaining every recipient field but outcome."""
    if type(assignment) is not PermutationAssignment:
        raise PermutationAdapterError("permutation assignment has the wrong type")
    assignment._validate()
    rows, execution_inventory, recipient_strata, observed_indexes = _canonical_cells(
        cells
    )
    if (
        execution_inventory != assignment.execution_cache_inventory
        or recipient_strata != assignment.recipient_strata
        or observed_indexes != assignment.source_observed_indexes
    ):
        raise PermutationAdapterError(
            "permutation assignment is being applied to a different source view"
        )

    permuted = tuple(
        replace(recipient, observed_index=rows[donor].observed_index)
        for recipient, donor in zip(
            rows, assignment.recipient_to_donor, strict=True
        )
    )
    for index, cell in enumerate(permuted):
        donor = rows[assignment.recipient_to_donor[index]]
        if (
            cell.observed_index != donor.observed_index
            or cell.k != rows[index].k
            or cell.group != rows[index].group
            or cell.segment_unmapped != rows[index].segment_unmapped
        ):
            raise PermutationAdapterError(
                "complete-set copy or recipient stratum preservation failed"
            )
    return permuted


__all__ = [
    "NULL_REPLAY_ROUTE",
    "OBSERVED_ROUTE",
    "PermutationAdapterError",
    "PermutationAssignment",
    "PermutationStratumAssignment",
    "STRATUM_KEYS",
    "apply_permutation",
    "build_null_permutation_assignment",
    "build_observed_permutation_assignment",
]
