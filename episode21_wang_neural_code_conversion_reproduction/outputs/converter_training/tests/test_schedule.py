"""Source-semantic checks only; prepared for a coordinated compute allocation.

No empirical files are opened. One check uses the existing report-bound public
models.py ZIP member. The tiny-width model below copies its architecture
literally, including the 3-D reshape and InstanceNorm1d(1); it is a comparator.
"""
from dataclasses import replace
import itertools
from pathlib import Path
import random
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import schedule


class SourceConverter(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        medium_num = input_nc // 2
        self.model = nn.Sequential(
            nn.Linear(input_nc, medium_num),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(0.2),
            nn.Linear(medium_num, medium_num),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(0.2),
            nn.Linear(medium_num, output_nc),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.model(x)
        return x.view(x.size(0), -1)


def source_prepare_indices(num_iterations, features_list):
    """Literal source algorithm, using its global Python RNG and FC zero chunk."""
    chunks_number = {
        'conv1_1': 64, 'conv1_2': 64,
        'conv2_1': 128, 'conv2_2': 128,
        'conv3_1': 256, 'conv3_2': 256, 'conv3_3': 256, 'conv3_4': 256,
        'conv4_1': 512, 'conv4_2': 512, 'conv4_3': 512, 'conv4_4': 512,
        'conv5_1': 512, 'conv5_2': 512, 'conv5_3': 512, 'conv5_4': 512,
        'fc1': 4096, 'fc2': 4096, 'fc3': 1000,
    }
    prepared_indices = {}
    for layer in features_list:
        if "fc" in layer:
            prepared_indices[layer] = [0] * num_iterations
        elif layer in chunks_number:
            current_indices = random.sample(range(chunks_number[layer]), chunks_number[layer])
            indices = []
            for i in range(num_iterations):
                if i % len(current_indices) == 0 and i > 0:
                    current_indices = random.sample(current_indices, len(current_indices))
                indices.append(current_indices[i % len(current_indices)])
            prepared_indices[layer] = indices
    return prepared_indices


class ScheduleTests(unittest.TestCase):
    def test_existing_source_loader_retains_pinned_odd_width_architecture(self):
        factory = schedule.load_pinned_converter_factory()
        with torch.random.fork_rng(devices=[]):
            initial = torch.Generator(device="cpu").manual_seed(42).get_state()
            torch.set_rng_state(initial)
            expected = SourceConverter(9, 11)
            expected_after = torch.get_rng_state().clone()
            torch.set_rng_state(initial)
            actual = factory(9, 11)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_after))
            for key, value in actual.state_dict().items():
                self.assertTrue(torch.equal(value, expected.state_dict()[key]), key)
            x = torch.arange(27, dtype=torch.float32).reshape(3, 9) / 7
            self.assertEqual(tuple(actual(x).shape), (3, 11))
            self.assertTrue(torch.equal(actual(x), expected(x)))

    def test_complete_shared_schedule_matches_tagged_algorithm(self):
        layers = [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8',
        ][::-1]
        previous = random.getstate()
        try:
            random.seed(42)
            expected = source_prepare_indices(1024, layers)
        finally:
            random.setstate(previous)
        actual = schedule.make_channel_schedule()
        self.assertEqual(random.getstate(), previous)
        self.assertEqual(schedule.LAYER_ORDER, tuple(layers))
        for iteration in range(1024):
            self.assertEqual(actual.blocks(iteration), tuple(
                (layer, None if layer.startswith("fc") else expected[layer][iteration])
                for layer in layers))
        self.assertEqual(actual, schedule.make_channel_schedule())

    def test_permutation_cycles_all_FC_units_and_explicit_iteration(self):
        actual = schedule.make_channel_schedule()
        for layer_index, layer in enumerate(schedule.LAYER_ORDER):
            values = actual.channels_by_layer[layer_index]
            if layer.startswith("fc"):
                self.assertEqual(values, (None,) * 1024)
                continue
            block = int(layer[4])
            count = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}[block]
            for start in range(0, 1024, count):
                self.assertEqual(sorted(values[start:start + count]), list(range(count)))
        for invalid in (-1, 1024, True, 0.0):
            with self.assertRaises(ValueError):
                actual.blocks(invalid)
        # Looking up a later step cannot consume a cursor needed by another pair.
        self.assertEqual(actual.blocks(0), schedule.make_channel_schedule().blocks(0))

    def test_canonical_pair_replay_matches_sequential_source_initialization(self):
        people = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")
        widths = dict(zip(people, (6, 8, 10, 12, 14)))
        pairs = tuple(itertools.permutations(people, 2))
        expected = []
        # Independent source-style construction: seed once, construct in order.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(42).get_state())
            for source, target in pairs:
                before = torch.get_rng_state().clone()
                model = SourceConverter(widths[source], widths[target])
                expected.append((before, torch.get_rng_state().clone(),
                                 {key: value.detach().clone() for key, value in model.state_dict().items()}))
                del model
        ambient = torch.get_rng_state().clone()
        records = schedule.capture_pair_initializations(widths, SourceConverter)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        self.assertEqual(len(records), 20)
        self.assertEqual(schedule.PAIR_ORDER, pairs)
        for index, record in enumerate(records):
            self.assertEqual((record.source, record.target), pairs[index])
            self.assertEqual(record.hidden_width, widths[record.source] // 2)
            self.assertTrue(torch.equal(record.cpu_rng_before, expected[index][0]))
            self.assertTrue(torch.equal(record.cpu_rng_after, expected[index][1]))
        # Replay in a different job order, including the last and same-source pairs.
        for index in (19, 1, 4, 0):
            model = schedule.initialize_pair(records[index], SourceConverter)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[index][2][key]), key)
            self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        # Resetting pair one to seed42 would incorrectly reuse pair zero's first layer.
        self.assertFalse(torch.equal(expected[0][2]["model.0.weight"],
                                     expected[1][2]["model.0.weight"]))

    def test_refusals_restore_CPU_rng_and_keep_canonical_width_bindings(self):
        widths = dict(zip(schedule.PARTICIPANTS, (6, 8, 10, 12, 14)))
        ambient = torch.get_rng_state().clone()

        def interrupted_factory(input_width, output_width):
            SourceConverter(input_width, output_width)
            raise KeyboardInterrupt("interrupted source construction")

        with self.assertRaises(KeyboardInterrupt):
            schedule.capture_pair_initializations(widths, interrupted_factory)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        records = schedule.capture_pair_initializations(widths, SourceConverter)

        def extra_rng_factory(input_width, output_width):
            model = SourceConverter(input_width, output_width)
            torch.rand(1)
            return model

        with self.assertRaisesRegex(ValueError, "RNG_consumption"):
            schedule.initialize_pair(records[0], extra_rng_factory)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        with self.assertRaisesRegex(ValueError, "canonical_pair"):
            schedule.initialize_pair(replace(records[0], pair_index=1), SourceConverter)
        with self.assertRaisesRegex(ValueError, "hidden|width"):
            schedule.initialize_pair(replace(records[0], hidden_width=99), SourceConverter)
        with self.assertRaisesRegex(ValueError, "exact_five"):
            schedule.capture_pair_initializations({"sub-01": 6}, SourceConverter)


if __name__ == "__main__":
    unittest.main()
