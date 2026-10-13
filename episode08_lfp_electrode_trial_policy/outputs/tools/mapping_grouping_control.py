"""Only acquired-row utility groups are shuffled; true feature geometry stays fixed."""
import random

import numpy as np

from mapping_variance import MappingDesign

POLICY = 'mapping_design_grouping_shuffle3_v1'
MC_SEEDS = (0, 1, 2)
ORDERS = ('original', '0', '1', '2')


class GroupingDesign(MappingDesign):
    def __init__(self, initial_x, labels, statistics, alpha, trial_ids,
                 filename, order, electrodes, subset_seed, mc_seed):
        if (order not in ORDERS or mc_seed not in MC_SEEDS
                or electrodes not in (8, 16, 32, 96)):
            raise ValueError('Exact frozen order, electrode count and MC seed required')
        super().__init__(initial_x, labels, statistics, alpha)
        self.trial_ids = [str(trial) for trial in trial_ids]
        if len(self.trial_ids) != 32 or len(set(self.trial_ids)) != 32:
            raise ValueError('Exact unique original initializer IDs required')
        self.key = (str(filename), str(order), str(electrodes),
                    'all' if electrodes == 96 else str(int(subset_seed)), str(mc_seed))

    def choose(self, counts):
        counts = np.asarray(counts)
        if (counts.shape != (8,) or not np.issubdtype(counts.dtype, np.integer)
                or np.any(counts < 4) or np.any(counts > 12)
                or not np.array_equal(counts, np.bincount(self.labels, minlength=8))
                or len(self.trial_ids) != len(self.labels)):
            raise ValueError('Counts and original IDs must describe only this acquired history')
        legal = np.flatnonzero(counts < 12)
        if not len(legal):
            raise ValueError('Finite queue exhausted')
        denominator = 1 + self.leverage
        gains = np.sum(self.cross * self.cross, axis=0) / (32 * denominator)
        if np.any(denominator <= 0) or not np.isfinite(gains).all():
            raise ArithmeticError('Invalid acquired design index; no jitter or clipping')
        canonical = sorted(range(len(self.trial_ids)), key=lambda index: self.trial_ids[index])
        literal = ('ep08-mapping-grouping-control-v1|20261003|'
                   + '|'.join((*self.key, str(len(self.trial_ids)))))
        pseudo_labels = [direction for direction in range(8) for _ in range(int(counts[direction]))]
        rng = random.Random()
        rng.seed(literal, version=2)
        rng.shuffle(pseudo_labels)  # Chance unchanged assignments are valid, with no rejection.
        pseudo = np.asarray(pseudo_labels)
        ordered_gains = gains[canonical]
        utilities = np.array([ordered_gains[pseudo == direction].mean() for direction in range(8)])
        chosen = int(min(legal, key=lambda direction: (-utilities[direction], counts[direction], direction)))
        return chosen, {
            'counts_before': counts.tolist(), 'utility': utilities.tolist(),
            'design_index_before': float(np.trace(self.cross[:, :32]) / 32),
            'seed_literal': literal,
            'canonical_acquired_trial_ids': [self.trial_ids[index] for index in canonical],
            'pseudo_direction_labels': pseudo_labels,
            'acquired_row_gain_sum': float(gains.sum()),
        }

    def observe(self, row, direction, trial_id):
        token = str(trial_id)
        if token in self.trial_ids:
            raise ValueError('Newly acquired original trial ID must be unique')
        super().observe(row, direction)
        self.trial_ids.append(token)
