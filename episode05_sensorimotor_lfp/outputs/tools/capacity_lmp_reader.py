"""Exact current-bin LMP/XY source view; no other bands or spike values."""
import json

import numpy as np

from primary_multichannel import guide_columns
from primary_reference import RECIPE as SCALAR_RECIPE
from run_primary_reference import trial_records
from source_adapter import M1_SESSIONS, M1Session


def load_lmp_session(source, table, scalar_root, session_id, check):
    specification = next(s for s in M1_SESSIONS if f"{s[0]}_{s[2]}" == session_id)
    rows = trial_records(table, session_id)
    # Trusted source identity reader only returns these structural fields;
    # no reference score is provided to the capacity fitter.
    old = json.loads((scalar_root/session_id/"score_bank.json").read_text())
    if (old["recipe"] != SCALAR_RECIPE or old["status"] != "complete_session_score_bank"
            or old["session_id"] != session_id or old["source"]["complete_trials"] != len(rows)):
        raise ValueError("Original complete scalar source identity required")
    channels = old["source"]["common_LMP_electrodes"]
    if not channels or channels != sorted(set(channels)) or any(type(c) is not int or c < 1 for c in channels):
        raise ValueError("Original positive physical LMP electrode identities required")
    del old
    metadata, columns, common = [], [], None
    x, y, logical_bytes, dtypes = [], [], 0, set()
    with M1Session(source, *specification) as session:
        for row in rows:
            check()
            trial = int(row["source_trial_index_zero_based"])
            meta = session.metadata(trial)
            if (meta["trial_id"] != row["trial_id"] or meta["direction"] != int(row["direction"])
                    or meta["direction_circular_error"] > .01 or meta["result"] != "R"
                    or meta["idx_movement_on"] != int(row["idx_movement_on"])):
                raise ValueError("Original trial identity/direction/event changed")
            guide = np.asarray(session.field(trial, "M1_lfp_guide")).T
            selected = guide_columns(guide, channels, ((0., 0.),))
            lmp = guide[(guide[:, 1] == 0) & (guide[:, 2] == 0), 0]
            if len(set(lmp)) != len(lmp) or np.any(lmp < 1) or np.any(lmp != np.floor(lmp)):
                raise ValueError("Unique positive integral LMP guide identities required")
            common = set(lmp) if common is None else common & set(lmp)
            metadata.append(meta)
            columns.append(selected)
        if sorted(common) != channels:
            raise ValueError("Original common LMP set changed")
        for row, meta, selected in zip(rows, metadata, columns):
            check()
            trial = int(row["source_trial_index_zero_based"])
            lfp, velocity = session.field(trial, "M1_lfp"), session.field(trial, "vel")
            start, end = meta["idx_movement_on"], meta["idx_movement_on"]+15
            if (lfp.ndim != 2 or velocity.shape != (2, lfp.shape[1]) or start < 0
                    or end > lfp.shape[1] or max(selected) >= lfp.shape[0]):
                raise ValueError("The full fixed primary window is unavailable")
            order = np.sort(selected)
            values, target = lfp[order, start:end], velocity[:, start:end]
            if not np.isfinite(values).all() or not np.isfinite(target).all():
                raise ValueError("Nonfinite complete trial; no exclusions")
            logical_bytes += values.nbytes+target.nbytes
            dtypes.add(str(values.dtype))
            x.append(values[np.searchsorted(order, selected)].T.astype(np.float64))
            y.append(target.T.astype(np.float64))
    return np.stack(x), np.stack(y), np.array([m["direction"] for m in metadata]), rows, {
        "session_id": session_id, "animal": specification[0], "implant": specification[1],
        "complete_trials": len(rows), "common_LMP_electrodes": channels, "native_offsets": [0, 14],
        "native_bin_seconds": .03, "stored_dtypes": sorted(dtypes), "selected_logical_source_bytes": int(logical_bytes),
        "source_fields_with_values_read": ["M1_lfp:LMP_only", "vel:XY"], "spike_values_read": False,
        "HDF_chunk_boundary": "Codecs may process mixed chunks; only selected LMP/XY values reach Python"}
