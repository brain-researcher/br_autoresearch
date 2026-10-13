"""Only adopted LMP/low/high guides and selected primary-window LFP/XY values."""
import json

import numpy as np

from primary_multichannel import BLOCKS, guide_columns, lmp_scale_diagnostic
from primary_reference import RECIPE as SCALAR_RECIPE
from run_primary_reference import trial_records
from source_adapter import M1_SESSIONS, M1Session


def load_session(source, table, scalar_root, session_id, check):
    specification = next(item for item in M1_SESSIONS if f"{item[0]}_{item[2]}" == session_id)
    rows = trial_records(table, session_id)
    previous = json.loads((scalar_root/session_id/"score_bank.json").read_text())
    if (previous["recipe"] != SCALAR_RECIPE or previous["status"] != "complete_session_score_bank"
            or previous["session_id"] != session_id or previous["source"]["complete_trials"] != len(rows)):
        raise ValueError("Completed unchanged scalar reference source identity required")
    channel_ids = previous["source"]["common_LMP_electrodes"]
    if (not channel_ids or channel_ids != sorted(set(channel_ids))
            or any(type(c) is not int or c < 1 for c in channel_ids)):
        raise ValueError("Original stable positive LMP electrode set required")
    metadata, columns, common = [], [], None
    all_bands = tuple(b for bands in BLOCKS.values() for b in bands)
    arrays, targets, dtypes, logical_bytes = [], [], set(), 0
    with M1Session(source, *specification) as session:
        for row in rows:
            check()
            trial = int(row["source_trial_index_zero_based"])
            meta = session.metadata(trial)
            if (meta["trial_id"] != row["trial_id"] or meta["direction"] != int(row["direction"])
                    or meta["direction_circular_error"] > .01 or meta["result"] != "R"
                    or meta["idx_movement_on"] != int(row["idx_movement_on"])):
                raise ValueError("Frozen original trial/direction/event changed")
            guide = np.asarray(session.field(trial, "M1_lfp_guide")).T
            selected = guide_columns(guide, channel_ids, all_bands)
            lmp = guide[(guide[:, 1] == 0) & (guide[:, 2] == 0), 0]
            if (len(set(lmp)) != len(lmp) or np.any(lmp < 1) or np.any(lmp != np.floor(lmp))):
                raise ValueError("Unique positive integral LMP guide identities required")
            common = set(lmp) if common is None else common & set(lmp)
            columns.append(selected)
            metadata.append(meta)
        if sorted(common) != channel_ids:
            raise ValueError("Stable LMP guide intersection differs from preserved scalar comparison")
        # All trial/band guide identities are fixed before any new feature value.
        for row, meta, selected in zip(rows, metadata, columns):
            check()
            trial = int(row["source_trial_index_zero_based"])
            lfp, velocity = session.field(trial, "M1_lfp"), session.field(trial, "vel")
            start, end = meta["idx_movement_on"], meta["idx_movement_on"]+15
            if (lfp.ndim != 2 or velocity.shape != (2, lfp.shape[1]) or start < 0
                    or end > lfp.shape[1] or max(selected) >= lfp.shape[0]):
                raise ValueError("Selected fixed execution window unavailable")
            ordered = np.sort(selected)
            undo = np.searchsorted(ordered, selected)
            values, target = lfp[ordered, start:end], velocity[:, start:end]
            if not np.isfinite(values).all() or not np.isfinite(target).all():
                raise ValueError("Nonfinite complete trial; no row or channel exclusion")
            dtypes.add(str(values.dtype))
            logical_bytes += values.nbytes+target.nbytes
            arrays.append(values[undo].T.astype(np.float64))
            targets.append(target.T.astype(np.float64))
    combined = np.stack(arrays)
    blocks, offset = {}, 0
    for recipe, bands in BLOCKS.items():
        indices = [channel*len(all_bands)+offset+band for channel in range(len(channel_ids)) for band in range(len(bands))]
        blocks[recipe] = combined[:, :, indices]
        offset += len(bands)
    diagnostic = lmp_scale_diagnostic(blocks["lmp_full_channel_ridge_v1"], dtypes)
    return blocks, np.stack(targets), np.array([m["direction"] for m in metadata]), rows, {
        "animal": specification[0], "implant": specification[1], "session_id": session_id,
        "complete_trials": len(rows), "common_LMP_electrodes": channel_ids,
        "bands_hz": {name: [list(b) for b in bands] for name, bands in BLOCKS.items()},
        "feature_order": "physical electrode ascending, then listed bands",
        "native_offsets": [0, 14], "native_bin_seconds": .03,
        "selected_logical_source_bytes": int(logical_bytes), "LMP_scale_diagnostic": diagnostic,
        "source_fields_with_values_read": ["M1_lfp:only_selected_LMP_low_high", "vel:XY"],
        "source_spike_values_read": False,
        "HDF_chunk_boundary": "Storage codecs may process mixed chunks; only selected feature/velocity values decoded to Python"}
