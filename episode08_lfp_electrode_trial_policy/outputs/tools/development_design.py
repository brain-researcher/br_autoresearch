"""Fixed outcome-blind EP08 resource cells, electrode seeds and row mappings."""
import random

import numpy as np

TRAINING_TOTALS = (32, 64, 96)
ELECTRODE_COUNTS = (8, 16, 32, 96)


def candidate_designs(role, channels, replicates):
    row_by_id = {trial: index for index, trial in enumerate(role["nested_training_sets"]["96"]["training_trials"])}
    channel_position = {channel: index for index, channel in enumerate(channels)}
    permutations = []
    for seed in range(replicates):
        shuffled = list(channels)
        rng = random.Random()
        rng.seed(f"ep08-random-electrodes-v1|20261003|{role['filename']}|{seed}", version=2)
        rng.shuffle(shuffled)
        permutations.append(shuffled)
    for total in TRAINING_TOTALS:
        training = role["nested_training_sets"][str(total)]
        rows = np.array([row_by_id[trial] for trial in training["training_trials"]])
        relative = {trial: index for index, trial in enumerate(training["training_trials"])}
        folds = [(np.array([relative[trial] for trial in fold["fitting_trials"]]),
                  np.array([relative[trial] for trial in fold["validation_trials"]]))
                 for fold in training["inner_folds"]]
        for count in ELECTRODE_COUNTS:
            for seed in range(1 if count == 96 else replicates):
                selected = permutations[seed][:count]
                columns = np.array([3 * channel_position[channel] + band for channel in selected for band in range(3)])
                yield {"training_trials": total, "electrodes": count, "seed": seed,
                       "channels": selected, "rows": rows, "columns": columns, "folds": folds}
