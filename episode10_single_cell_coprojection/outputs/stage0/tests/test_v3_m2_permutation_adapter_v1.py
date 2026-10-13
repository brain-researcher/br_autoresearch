#!/usr/bin/env python3
"""Target-free tests for the frozen EP10 v3 whole-set permutation."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import v3_m2_permutation_adapter_v1 as permutation  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


def _coverage_binding() -> controller.CoverageBinding:
    return controller.CoverageBinding(
        tuple((index,) for index in range(controller.COVERAGE_TRIALS)),
        ("0x0.0p+0", "0x0.0p+0", "0x0.0p+0", 0),
        tuple("0x0.0p+0" for _ in range(controller.PAIR_COUNT)),
    )


def _trial(
    *, seed: int = 2026092811, valid_index: int = 28
) -> controller.Trial:
    return controller.Trial(
        valid_index=valid_index,
        role="association",
        kind="permutation",
        phase="postcoverage" if valid_index <= 40 else "patience",
        falsifier=True,
        configuration=controller.Configuration("lasso", 0.1),
        seed=seed,
        cycle=1 if valid_index <= 40 else 3,
        coverage_binding=_coverage_binding(),
    )


def _canonical_cells() -> tuple[m1.Cell, ...]:
    local_by_group = {group: 0 for group in m1.GROUP_ORDER}
    group_ordinal = {group: index for index, group in enumerate(m1.GROUP_ORDER)}
    cells: list[m1.Cell] = []
    execution_order = tuple(
        entry
        for group in m1.GROUP_ORDER
        for entry in sorted(
            (
                entry
                for entry in controller.FROZEN_CELL_INVENTORY.entries
                if entry[1] == group
            ),
            key=lambda entry: int(entry[0]),
        )
    )
    assert execution_order != controller.FROZEN_CELL_INVENTORY.entries
    for row_index, (identifier, group) in enumerate(
        execution_order
    ):
        local = local_by_group[group]
        local_by_group[group] += 1
        k = (5 * local + group_ordinal[group]) % (m1.PANEL_SIZE + 1)
        count = math.comb(m1.PANEL_SIZE, k)
        distances = np.linspace(0.1, 1.6, m1.PANEL_SIZE) + row_index * 1.0e-4
        distances.setflags(write=False)
        cells.append(
            m1.Cell(
                identifier=identifier,
                group=group,
                layer=m1.LAYERS[(row_index + local) % len(m1.LAYERS)],
                hemisphere="left" if row_index % 2 == 0 else "right",
                soma_um=(float(row_index), float(local), float(row_index + local)),
                soma_distance_mm=distances,
                k=k,
                observed_index=(37 * row_index + 11 * local) % count,
                segment_unmapped=(local // 3) % 2,
                sampling_uniform=(row_index + 0.5) / 194.0,
            )
        )
    assert local_by_group == m1.GROUP_SIZES
    return tuple(cells)


def _reference_mapping(
    cells: tuple[m1.Cell, ...], bit_generator: np.random.PCG64
) -> tuple[int, ...]:
    """Small independent transcription of the v4 mapping specification."""
    mapping = [-1] * len(cells)
    modulus = 1 << 64
    for group, segment_unmapped, k in permutation.STRATUM_KEYS:
        recipients = [
            index
            for index, cell in enumerate(cells)
            if (
                cell.group == group
                and cell.segment_unmapped == segment_unmapped
                and cell.k == k
            )
        ]
        donors = list(recipients)
        for index in range(len(donors) - 1, 0, -1):
            upper = index + 1
            limit = (modulus // upper) * upper
            while True:
                raw = int(bit_generator.random_raw())
                if raw < limit:
                    partner = raw % upper
                    break
            donors[index], donors[partner] = donors[partner], donors[index]
        for recipient, donor in zip(recipients, donors, strict=True):
            mapping[recipient] = donor
    assert all(index >= 0 for index in mapping)
    return tuple(mapping)


def _assert_random_state_equal(
    case: unittest.TestCase, left: tuple[object, ...], right: tuple[object, ...]
) -> None:
    case.assertEqual(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    case.assertEqual(left[2:], right[2:])


class _RawWords:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)
        self.calls = 0

    def random_raw(self) -> int:
        self.calls += 1
        if not self.values:
            raise AssertionError("unexpected raw RNG draw")
        return self.values.pop(0)


class PermutationMappingTests(unittest.TestCase):
    def test_observed_mapping_has_exact_order_bijections_and_complete_set_apply(
        self,
    ) -> None:
        cells = _canonical_cells()
        trial = _trial()
        assignment = permutation.build_observed_permutation_assignment(cells, trial)

        self.assertEqual(assignment.rng_route, permutation.OBSERVED_ROUTE)
        self.assertEqual(assignment.scheduled_trial_valid_index, trial.valid_index)
        self.assertEqual(assignment.scheduled_trial_seed, trial.seed)
        self.assertEqual(
            assignment.scheduled_configuration_key, trial.configuration.key
        )
        self.assertIsNone(assignment.null_replicate_index)
        self.assertIsNone(assignment.null_replicate_seed)
        self.assertIsNone(assignment.seedsequence_spawn_key)
        self.assertEqual(
            assignment.execution_cache_inventory,
            tuple((cell.identifier, cell.group) for cell in cells),
        )
        self.assertNotEqual(
            assignment.execution_cache_inventory,
            controller.FROZEN_CELL_INVENTORY.entries,
        )
        self.assertEqual(
            tuple(record.key for record in assignment.strata),
            permutation.STRATUM_KEYS,
        )
        self.assertEqual(len(assignment.strata), 12 * 2 * 17)

        for record in assignment.strata:
            self.assertEqual(
                tuple(sorted(record.donor_indexes)), record.recipient_indexes
            )
            if len(record.recipient_indexes) <= 1:
                self.assertEqual(record.raw_uint64_draws, 0)
            for recipient, donor in zip(
                record.recipient_indexes, record.donor_indexes, strict=True
            ):
                self.assertEqual(
                    (
                        cells[recipient].group,
                        cells[recipient].segment_unmapped,
                        cells[recipient].k,
                    ),
                    (
                        cells[donor].group,
                        cells[donor].segment_unmapped,
                        cells[donor].k,
                    ),
                )

        permuted = assignment.apply(cells)
        self.assertEqual(len(permuted), 194)
        for index, (source, result) in enumerate(zip(cells, permuted, strict=True)):
            donor = cells[assignment.recipient_to_donor[index]]
            self.assertEqual(result.observed_index, donor.observed_index)
            for field in dataclasses.fields(m1.Cell):
                if field.name == "observed_index":
                    continue
                source_value = getattr(source, field.name)
                result_value = getattr(result, field.name)
                if isinstance(source_value, np.ndarray):
                    self.assertIs(result_value, source_value)
                else:
                    self.assertEqual(result_value, source_value)

        for record in assignment.strata:
            before = Counter(
                cells[index].observed_index for index in record.recipient_indexes
            )
            after = Counter(
                permuted[index].observed_index for index in record.recipient_indexes
            )
            self.assertEqual(before, after)

    def test_observed_route_is_exact_repeatable_and_does_not_touch_global_rng(
        self,
    ) -> None:
        cells = _canonical_cells()
        trial = _trial()
        np.random.seed(9173)
        before = np.random.get_state()
        first = permutation.build_observed_permutation_assignment(cells, trial)
        after = np.random.get_state()
        second = permutation.build_observed_permutation_assignment(cells, trial)
        _assert_random_state_equal(self, before, after)
        self.assertEqual(first, second)

        expected = _reference_mapping(cells, np.random.PCG64(trial.seed))
        self.assertEqual(first.recipient_to_donor, expected)
        changed = permutation.build_observed_permutation_assignment(
            cells, _trial(seed=2026092812)
        )
        self.assertNotEqual(first.recipient_to_donor, changed.recipient_to_donor)

    def test_null_route_uses_exact_seedsequence_substream_not_trial_seed(self) -> None:
        cells = _canonical_cells()
        trial = _trial()
        first = permutation.build_null_permutation_assignment(
            cells, trial, null_replicate_index=1
        )
        repeat = permutation.build_null_permutation_assignment(
            cells, trial, null_replicate_index=1
        )
        self.assertEqual(first, repeat)
        self.assertEqual(first.rng_route, permutation.NULL_REPLAY_ROUTE)
        self.assertEqual(first.null_replicate_index, 1)
        self.assertEqual(first.null_replicate_seed, controller.NULL_SEEDS[0])
        self.assertEqual(first.seedsequence_spawn_key, (3, 0, 28, 4))

        expected_generator = np.random.PCG64(
            np.random.SeedSequence(controller.NULL_SEEDS[0], spawn_key=(3, 0, 28, 4))
        )
        self.assertEqual(
            first.recipient_to_donor,
            _reference_mapping(cells, expected_generator),
        )
        self.assertNotEqual(
            first.recipient_to_donor,
            permutation.build_null_permutation_assignment(
                cells, trial, null_replicate_index=2
            ).recipient_to_donor,
        )
        other_trial_index = permutation.build_null_permutation_assignment(
            cells,
            _trial(valid_index=41),
            null_replicate_index=1,
        )
        self.assertEqual(other_trial_index.seedsequence_spawn_key, (3, 0, 41, 4))
        self.assertNotEqual(
            first.recipient_to_donor,
            other_trial_index.recipient_to_donor,
        )

        alternate_scheduled_seed = permutation.build_null_permutation_assignment(
            cells,
            _trial(seed=2026092860),
            null_replicate_index=1,
        )
        self.assertEqual(
            first.recipient_to_donor,
            alternate_scheduled_seed.recipient_to_donor,
        )
        self.assertNotEqual(
            first.scheduled_trial_seed,
            alternate_scheduled_seed.scheduled_trial_seed,
        )

    def test_raw_rejection_singletons_and_complete_identity_are_accepted(self) -> None:
        raw = _RawWords([(1 << 64) - 1, 4])
        selected, draws = permutation._bounded_raw_uint64(raw, 3)
        self.assertEqual((selected, draws, raw.calls), (1, 2, 2))

        forbidden = _RawWords([])
        self.assertEqual(permutation._fisher_yates((), forbidden), ((), 0))
        self.assertEqual(permutation._fisher_yates((9,), forbidden), ((9,), 0))
        self.assertEqual(forbidden.calls, 0)

        identity_raw = _RawWords([5, 4, 3, 2, 1])
        identity, identity_draws = permutation._fisher_yates(
            tuple(range(6)), identity_raw
        )
        self.assertEqual(identity, tuple(range(6)))
        self.assertEqual(identity_draws, 5)

        cells = _canonical_cells()
        (
            rows,
            execution_inventory,
            recipient_strata,
            observed_indexes,
        ) = permutation._canonical_cells(cells)
        identity_words: list[int] = []
        for key in permutation.STRATUM_KEYS:
            size = sum(recipient_key == key for recipient_key in recipient_strata)
            identity_words.extend(range(size - 1, 0, -1))
        complete_identity_source = _RawWords(identity_words)
        # The shuffle primitive accepts fixed points and a complete identity.
        # A finished assignment must additionally replay from its claimed PCG64
        # seed, so a hand-supplied identity stream cannot be relabelled as that
        # seed's authoritative mapping.
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "claimed RNG provenance"
        ):
            permutation._build_assignment(
                rows,
                execution_inventory,
                recipient_strata,
                observed_indexes,
                _trial(),
                rng_route=permutation.OBSERVED_ROUTE,
                bit_generator=complete_identity_source,
                null_replicate_index=None,
                null_replicate_seed=None,
                seedsequence_spawn_key=None,
            )
        self.assertEqual(complete_identity_source.values, [])

    def test_assignment_and_provenance_are_immutable(self) -> None:
        assignment = permutation.build_observed_permutation_assignment(
            _canonical_cells(), _trial()
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            assignment.raw_uint64_draws = 0  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            assignment.strata[0].raw_uint64_draws = 1  # type: ignore[misc]
        self.assertIs(type(assignment.recipient_to_donor), tuple)
        self.assertIs(type(assignment.source_observed_indexes), tuple)

        mapping = list(assignment.recipient_to_donor)
        left = 0
        right = next(
            index
            for index, key in enumerate(assignment.recipient_strata)
            if key != assignment.recipient_strata[left]
        )
        mapping[left], mapping[right] = mapping[right], mapping[left]
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError,
            "global recipient-to-donor|crosses",
        ):
            dataclasses.replace(assignment, recipient_to_donor=tuple(mapping))

        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "claimed RNG provenance"
        ):
            dataclasses.replace(
                assignment,
                scheduled_trial_seed=assignment.scheduled_trial_seed + 1,
            )

        null_assignment = permutation.build_null_permutation_assignment(
            _canonical_cells(), _trial(), null_replicate_index=1
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "claimed RNG provenance"
        ):
            dataclasses.replace(
                null_assignment,
                null_replicate_index=2,
                null_replicate_seed=controller.NULL_SEEDS[1],
            )

        identity_strata = tuple(
            dataclasses.replace(record, donor_indexes=record.recipient_indexes)
            for record in assignment.strata
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "claimed RNG provenance"
        ):
            dataclasses.replace(
                assignment,
                strata=identity_strata,
                recipient_to_donor=tuple(range(len(assignment.recipient_to_donor))),
            )

    def test_reordering_invalid_membership_and_different_source_view_fail_closed(
        self,
    ) -> None:
        cells = _canonical_cells()
        trial = _trial()
        assignment = permutation.build_observed_permutation_assignment(cells, trial)
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "different source view"
        ):
            assignment.apply(cells[::-1])

        reversed_assignment = permutation.build_observed_permutation_assignment(
            cells[::-1], trial
        )
        self.assertEqual(
            reversed_assignment.execution_cache_inventory,
            tuple((cell.identifier, cell.group) for cell in cells[::-1]),
        )
        self.assertNotEqual(
            reversed_assignment.execution_cache_inventory,
            assignment.execution_cache_inventory,
        )

        invalid_membership = list(cells)
        invalid_membership[0] = dataclasses.replace(
            invalid_membership[0], identifier="not_in_frozen_inventory"
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "frozen 194-cell membership"
        ):
            permutation.build_observed_permutation_assignment(
                invalid_membership, trial
            )

        invalid_segment = list(cells)
        invalid_segment[0] = dataclasses.replace(invalid_segment[0], segment_unmapped=2)
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "segment_unmapped"
        ):
            permutation.build_observed_permutation_assignment(invalid_segment, trial)

        invalid_observed = list(cells)
        count = math.comb(m1.PANEL_SIZE, invalid_observed[0].k)
        invalid_observed[0] = dataclasses.replace(
            invalid_observed[0], observed_index=count
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "observed_index"
        ):
            permutation.build_observed_permutation_assignment(invalid_observed, trial)

        changed_source = list(cells)
        replace_index = next(
            index for index, cell in enumerate(cells) if math.comb(16, cell.k) > 1
        )
        changed_source[replace_index] = dataclasses.replace(
            changed_source[replace_index],
            observed_index=(changed_source[replace_index].observed_index + 1)
            % math.comb(16, changed_source[replace_index].k),
        )
        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "different source view"
        ):
            assignment.apply(changed_source)

        with self.assertRaisesRegex(
            permutation.PermutationAdapterError, "outside 1..99"
        ):
            permutation.build_null_permutation_assignment(
                cells, trial, null_replicate_index=0
            )


if __name__ == "__main__":
    unittest.main()
