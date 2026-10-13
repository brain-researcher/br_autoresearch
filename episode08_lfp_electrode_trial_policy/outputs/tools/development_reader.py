"""Trusted selective development reader; never imported by the fixed fitter.

The two explicit entrypoints enforce disjoint training/evaluation slices.
This is a callable boundary within one trusted process, not a same-UID OS seal.
"""
import h5py
import numpy as np
from source_support import field, mclass
from prepare_development_trial_roles import DEVELOPMENT
from development_design import TRAINING_TOTALS

BANDS = ((0., 0.), (100., 200.), (200., 400.))


def selected_metadata(metadata, roles, filename):
    if filename not in DEVELOPMENT:
        raise ValueError("Only the fixed eight development sessions are permitted")
    sources = [item for item in metadata["sessions"] if item["filename"] == filename]
    selected = [item for item in roles["sessions"] if item["filename"] == filename]
    if len(sources) != 1 or len(selected) != 1:
        raise ValueError("Expected the single selected development session")
    source, role = sources[0], selected[0]
    if source["session_role"] != "development" or role["session_role"] != "development":
        raise ValueError("Held sessions cannot enter this development reader")
    evaluation = set(role["evaluation_trials"])
    if len(evaluation) != 40 or len(role["evaluation_trials"]) != 40:
        raise ValueError("Preserve the forty reserved evaluation trials")
    records = {item["trial_index_zero_based"]: item for item in source["metadata_eligible_trials"]}
    all_training = role["nested_training_sets"]["96"]["training_trials"]
    if len(all_training) != 96 or len(set(all_training)) != 96 or evaluation.intersection(all_training):
        raise ValueError("Training identities are duplicated or overlap evaluation")
    for total in TRAINING_TOTALS:
        item = role["nested_training_sets"][str(total)]
        expected = [trial for pool in role["fitting_pool_by_direction"] for trial in pool[:total // 8]]
        if item["training_trials"] != expected or not set(expected).issubset(all_training):
            raise ValueError("Training sets differ from the adopted nested prefixes")
        if evaluation.intersection(expected):
            raise ValueError("Evaluation trial requested by a fitting set")
        if len(item["inner_folds"]) != 4:
            raise ValueError("Expected four fixed training-only folds")
        for fold_index, fold in enumerate(item["inner_folds"]):
            valid = [trial for pool in role["fitting_pool_by_direction"]
                     for position, trial in enumerate(pool[:total // 8]) if position % 4 == fold_index]
            fit = [trial for trial in expected if trial not in set(valid)]
            if fold["validation_trials"] != valid or fold["fitting_trials"] != fit:
                raise ValueError("Inner folds differ from the adopted whole-trial rule")
    if any(trial not in records for trial in list(all_training) + list(evaluation)):
        raise ValueError("A selected trial lacks source metadata")
    return source, role, records


def _read_arrays(path, source, role, records, check_deadline, access, evaluation_mode, reference):
    if (path.name not in DEVELOPMENT or path.name != source["filename"]
            or path.name != role["filename"] or source["session_role"] != "development"
            or role["session_role"] != "development"):
        raise ValueError("Reader path and fixed development role disagree")
    channels = source["common_physical_channels"]
    if channels != sorted(set(channels)) or len(channels) != 96:
        raise ValueError("Expected all96 previously supported physical electrodes")
    training_ids = role["nested_training_sets"]["96"]["training_trials"]
    evaluation = set(role["evaluation_trials"])
    if evaluation.intersection(training_ids):
        raise ValueError("Training and evaluation identities overlap")
    selected_ids = role["evaluation_trials"] if evaluation_mode else training_ids
    x, y = [], []
    reference_guide, reference_units = reference if reference is not None else (None, None)
    if evaluation_mode and reference is None:
        raise ValueError("Evaluation requires the established training guide identities")
    with h5py.File(path, "r") as handle:
        groups = [handle[name] for name in handle if not name.startswith("#")
                  and isinstance(handle[name], h5py.Group) and mclass(handle[name]) == "struct"]
        if len(groups) != 1:
            raise ValueError("Expected one native trial struct")
        group = groups[0]
        for trial in selected_ids:
            check_deadline()
            if (evaluation_mode and trial not in evaluation) or (not evaluation_mode and trial in evaluation):
                raise ValueError("Trial read violates the declared reader role")
            guide = field(handle, group, "M1_lfp_guide", trial)[...].T
            units = field(handle, group, "M1_unit_guide", trial)[...].T
            if (guide.ndim != 2 or guide.shape[1] != 3 or units.ndim != 2
                    or units.shape[1] != 2 or units.shape[0] < 1):
                raise ValueError("Unexpected physical-electrode or population-unit guide")
            if not np.isfinite(units).all() or len(np.unique(units, axis=0)) != len(units):
                raise ValueError("Population-unit identities are nonfinite or duplicated; no unit removal")
            if reference_guide is None:
                reference_guide, reference_units = guide.copy(), units.copy()
            elif not np.array_equal(guide, reference_guide) or not np.array_equal(units, reference_units):
                raise ValueError("Trial electrode or population-unit identity changed")
            selected_rows = []
            for channel in channels:
                for band in BANDS:
                    matches = np.flatnonzero((guide[:, 0] == channel) & (guide[:, 1] == band[0]) & (guide[:, 2] == band[1]))
                    if len(matches) != 1:
                        raise ValueError("Required channel/band identity is absent or duplicated")
                    selected_rows.append(int(matches[0]))
            sorted_rows = np.sort(selected_rows)
            undo = np.searchsorted(sorted_rows, selected_rows)
            info = records[trial]
            start, end = info["window_start_zero_based"], info["window_end_exclusive"]
            if end - start != 10:
                raise ValueError("Expected the qualified ten-bin movement window")
            lfp_node = field(handle, group, "M1_lfp", trial)
            spike_node = field(handle, group, "M1_spikes", trial)
            if (list(reversed(lfp_node.shape)) != info["array_shapes"]["M1_lfp"]
                    or list(reversed(spike_node.shape)) != info["array_shapes"]["M1_spikes"]
                    or lfp_node.shape[0] != len(guide) or spike_node.shape[0] != len(units)):
                raise ValueError("Source array shape and guide/support metadata disagree")
            # Only these two expressions deserialize neural numeric elements.
            if evaluation_mode:
                access["evaluation_neural_elements_read"] = True
            lfp = lfp_node[sorted_rows, start:end].T[:, undo]
            access["logical_neural_bytes"] += int(lfp.nbytes)
            spikes = spike_node[:, start:end].T
            access["logical_neural_bytes"] += int(spikes.nbytes)
            access["evaluation_trials_read" if evaluation_mode else "fitting_trials_read"] += 1
            if not np.isfinite(lfp).all() or not np.isfinite(spikes).all():
                raise ValueError("Nonfinite representation; no trial/unit removal")
            x.append(np.asarray(lfp, dtype=np.float64).mean(axis=0))
            y.append(np.asarray(spikes, dtype=np.float64).mean(axis=0))
    return np.stack(x), np.stack(y), channels, (reference_guide, reference_units)



def read_training_arrays(path, source, role, records, check_deadline, access):
    """Read only the declared 96 fitting rows; no evaluation read is possible here."""
    return _read_arrays(path, source, role, records, check_deadline, access, False, None)


def read_evaluation_arrays(path, source, role, records, check_deadline, access, training_guides):
    """Trusted evaluator only: exactly the 40 reserved rows, matching training guides."""
    return _read_arrays(path, source, role, records, check_deadline, access, True, training_guides)
