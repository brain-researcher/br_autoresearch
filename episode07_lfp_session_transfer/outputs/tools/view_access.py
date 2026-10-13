"""EP07 candidate-side reader for the two explicitly permitted role files.

This is runner-enforced separation, not operating-system isolation. Candidate
jobs must receive this candidate directory, never the source/scorer/sealed roots.
"""
from pathlib import Path

import h5py
import numpy as np

EARLIER = {
    "Mihili": ("20140217", "20140218", "20140303"),
    "Chewie-L": ("20160929", "20161005", "20161006"),
}
NEW = {
    # Joint October 10 amendment; retained dates keep their original identities.
    "Mihili": ("20140304", "20140307"),
    "Chewie-L": ("20161007", "20161014", "20161021"),
}
FEATURE_KEYS = ("lfp", "lfp_guide", "bin_offsets")
FIT_KEYS = FEATURE_KEYS + ("spikes", "unit_guide")


class CandidateViews:
    def __init__(self, candidate_directory):
        self._directory = Path(candidate_directory)

    @staticmethod
    def _check_scope(animal, target_day, region):
        if animal not in NEW or target_day not in NEW[animal] or region not in ("M1", "PMd"):
            raise ValueError("scope must name one contract animal, target day and region")

    def iter_records(self, role, *, animal, target_day, region, stream):
        if role not in ("fit", "development_features"):
            raise PermissionError("candidate reader cannot open scoring targets or held-out data")
        self._check_scope(animal, target_day, region)
        if stream not in ("history_assisted", "today_only"):
            raise ValueError("unknown model stream")
        path = self._directory / ("fit.h5" if role == "fit" else "development_features.h5")
        with h5py.File(path, "r") as handle:
            days = ((EARLIER[animal] + (target_day,))
                    if role == "fit" and stream == "history_assisted" else (target_day,))
            for day in days:
                group = handle[f"{animal}/{day}/{region}"]
                for trial in sorted(group):
                    record = group[trial]
                    expected = ("training" if day in EARLIER[animal] else "calibration") if role == "fit" else "development"
                    if record.attrs["role"] != expected:
                        raise ValueError("role file contains an ineligible fitting trial")
                    keys = FIT_KEYS if role == "fit" else FEATURE_KEYS
                    if set(record.keys()) != set(keys):
                        raise ValueError("unexpected data field in candidate view")
                    yield {"animal": animal, "day": day, "region": region,
                           "trial_id": trial, "role": expected,
                           "direction": int(record.attrs["direction"]),
                           **{key: np.asarray(record[key]) for key in keys}}
