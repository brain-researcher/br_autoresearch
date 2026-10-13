"""Map provider acquisition-ordered trial rows to session NIfTI volumes."""
from itertools import groupby


def session_volume_indices(records, session, volume_count):
    if len(records) != volume_count:
        raise ValueError("Trial count differs from session beta volume count")
    groups = [(run, list(rows)) for run, rows in groupby(records, key=lambda r: r["run"])]
    run_ids = [run for run, _ in groups]
    if len(set(run_ids)) != len(run_ids) or run_ids != sorted(run_ids, key=int):
        raise ValueError("Trial table does not preserve unique acquisition-ordered runs")
    for _, rows in groups:
        if any(row["session"] != session for row in rows):
            raise ValueError("Trial session identity differs from beta file")
        if [int(row["beta_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError("Run-local trial beta indices are not consecutive")
    # The provider sidecar declares one volume per acquisition-ordered trial.
    # beta_index is run-local; the TSV row ordinal is the session volume index.
    return range(volume_count)
