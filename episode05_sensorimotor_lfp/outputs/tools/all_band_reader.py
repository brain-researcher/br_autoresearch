"""Qualify all nine fixed guide pairs on every original trial before values."""
import json

import numpy as np

from primary_all_band import ALL_BANDS, BLOCKS
from primary_multichannel import guide_columns
from primary_reference import RECIPE as SCALAR_RECIPE
from run_primary_reference import trial_records
from source_adapter import M1_SESSIONS, M1Session


def split_blocks(combined, electrodes):
    if combined.ndim != 3 or combined.shape[1:] != (15, 9*electrodes) or electrodes < 1:
        raise ValueError("Complete fifteen-bin nine-band physical layout required")
    blocks, offset = {}, 0
    for name, bands in BLOCKS.items():
        indices = [e*len(ALL_BANDS)+offset+b for e in range(electrodes) for b in range(len(bands))]
        # Match the preserved J reader's advanced column-selection layout.
        blocks[name] = combined[:, :, indices]
        offset += len(bands)
    return blocks


def load_session(source, table, scalar_root, session_id, check):
    specification = next(s for s in M1_SESSIONS if f"{s[0]}_{s[2]}" == session_id)
    rows = trial_records(table, session_id)
    old = json.loads((scalar_root/session_id/"score_bank.json").read_text())
    if (old["recipe"] != SCALAR_RECIPE or old["status"] != "complete_session_score_bank"
            or old["session_id"] != session_id or old["source"]["complete_trials"] != len(rows)):
        raise ValueError("Original complete scalar source identity required")
    channels = old["source"]["common_LMP_electrodes"]
    if not channels or channels != sorted(set(channels)) or any(type(c) is not int or c < 1 for c in channels):
        raise ValueError("Frozen positive physical LMP electrode identities required")
    del old
    metadata, columns, common = [], [], None
    arrays, targets, dtypes, logical_bytes = [], [], set(), 0
    with M1Session(source, *specification) as session:
        for row in rows:
            check()
            trial = int(row["source_trial_index_zero_based"])
            meta = session.metadata(trial)
            if (meta["trial_id"] != row["trial_id"] or meta["direction"] != int(row["direction"])
                    or meta["direction_circular_error"] > .01 or meta["result"] != "R"
                    or meta["idx_movement_on"] != int(row["idx_movement_on"])):
                raise ValueError("Original trial/direction/event identity changed")
            guide = np.asarray(session.field(trial, "M1_lfp_guide")).T
            selected = guide_columns(guide, channels, ALL_BANDS)
            lmp = guide[(guide[:, 1] == 0) & (guide[:, 2] == 0), 0]
            if len(set(lmp)) != len(lmp) or np.any(lmp < 1) or np.any(lmp != np.floor(lmp)):
                raise ValueError("Unique positive integral LMP guide identities required")
            common = set(lmp) if common is None else common & set(lmp)
            columns.append(selected)
            metadata.append(meta)
        if sorted(common) != channels:
            raise ValueError("All-band qualification may not change the frozen LMP intersection")
        # Every original trial has all nine bands before any new feature read.
        for row, meta, selected in zip(rows, metadata, columns):
            check()
            trial = int(row["source_trial_index_zero_based"])
            lfp, velocity = session.field(trial, "M1_lfp"), session.field(trial, "vel")
            start, end = meta["idx_movement_on"], meta["idx_movement_on"]+15
            if (lfp.ndim != 2 or velocity.shape != (2, lfp.shape[1]) or start < 0
                    or end > lfp.shape[1] or max(selected) >= lfp.shape[0]):
                raise ValueError("Complete fixed native LMP/power/XY window required")
            ordered = np.sort(selected)
            values, target = lfp[ordered, start:end], velocity[:, start:end]
            if not np.isfinite(values).all() or not np.isfinite(target).all():
                raise ValueError("Nonfinite complete trial; no channel/band/trial exclusions")
            dtypes.add(str(values.dtype))
            logical_bytes += values.nbytes+target.nbytes
            arrays.append(values[np.searchsorted(ordered, selected)].T.astype(np.float64))
            targets.append(target.T.astype(np.float64))
    blocks = split_blocks(np.stack(arrays), len(channels))
    return blocks, np.stack(targets), np.array([m["direction"] for m in metadata]), rows, {
        "session_id": session_id, "animal": specification[0], "implant": specification[1],
        "complete_trials": len(rows), "common_LMP_electrodes": channels,
        "bands_hz": {name: [list(b) for b in bands] for name, bands in BLOCKS.items()},
        "all_bands_hz": [list(b) for b in ALL_BANDS], "all_trial_guides_qualified_before_values": True,
        "feature_order": "raw combined: physical electrode then LMP, high2, added6; final blocks LMP/high/added",
        "native_offsets": [0, 14], "native_bin_seconds": .03,
        "stored_dtypes": sorted(dtypes), "selected_logical_source_bytes": int(logical_bytes),
        "source_fields_with_values_read": ["M1_lfp:fixed_LMP_and_all8_power_bands", "vel:XY"],
        "source_spike_values_read": False,
        "newly_requested_bands_hz": [[12, 25], [25, 50], [50, 100]],
        "HDF_chunk_boundary": "Codecs may process mixed chunks; only selected feature/velocity values reach Python"}
