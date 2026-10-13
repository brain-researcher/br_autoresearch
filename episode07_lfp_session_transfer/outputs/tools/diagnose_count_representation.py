#!/usr/bin/env python3
"""Diagnose the observed EP07 native-count validity failure, without values.

These predicates do not authorize rounding, rescaling, neuron removal, or a
new response representation. No neural amplitudes, moments or scores are emitted.
"""
import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from prepare_role_views import field, identity, TrialUnavailable
from source_support import compute_only, write_json


def near_integer(values):
    tolerance = 8 * np.spacing(np.maximum(np.abs(values), 1.0))
    return bool(np.all(np.abs(values - np.rint(values)) <= tolerance))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    compute_only()
    report = {"session": "Mihili_CO_20140306.mat", "scope": "native_count_validity_only",
              "neural_values_or_moments_emitted": False,
              "rounding_rescaling_or_exclusion_authorized": False, "regions": {}}
    regions = {region: Counter() for region in ("M1", "PMd")}
    shapes = {region: set() for region in regions}
    with h5py.File(args.source / report["session"], "r") as handle:
        count = int(handle["trial_data"]["monkey"].size)
        report["source_trials"] = count
        for trial in range(count):
            try:
                _, windows, _, _ = identity(handle, trial, "Mihili", "20140306")
            except TrialUnavailable as error:
                for summary in regions.values():
                    summary[str(error)] += 1
                continue
            for region, summary in regions.items():
                node = field(handle, trial, f"{region}_spikes")
                shapes[region].add((str(node.dtype), node.shape[0]))
                left, right = windows[region]
                values = node[:, left:right]
                if right > node.shape[1] or values.shape[1] != right - left:
                    summary["required_window_absent"] += 1
                    continue
                summary["windows_examined"] += 1
                if not np.isfinite(values).all():
                    summary["nonfinite_windows"] += 1
                    continue
                summary["finite_windows"] += 1
                if np.all(values >= 0):
                    summary["nonnegative_windows"] += 1
                if np.all(values == np.floor(values)):
                    summary["exact_integer_windows"] += 1
                elif near_integer(values):
                    summary["noninteger_but_within_8_ulp_of_integer_windows"] += 1
                else:
                    summary["noninteger_beyond_8_ulp_windows"] += 1
                    if near_integer(values * 0.03):
                        summary["noninteger_becomes_near_integer_if_multiplied_by_bin_seconds"] += 1
        report["regions"] = {region: {"validity_counts": dict(summary),
                              "spike_dtype_and_column_counts": [list(item) for item in sorted(shapes[region])]}
                              for region, summary in regions.items()}
    write_json(args.out, report)
    print("Completed fixed-file native-count validity diagnosis; no values or scores emitted.")


if __name__ == "__main__":
    main()
