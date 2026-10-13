"""M0a: raw training-person mean in the fixed anatomical space.

Array-only LOPO callback, not a reader or empirical-access grant. No Task A,
functional alignment, target-specific gain or target record is consumed.
All four raw Task-B partitions are spatially centered on the complete fixed
selected location axis and averaged equally across the exact training people.
Geometry uses the two actual cross-partition products, not the mean-map Gram.
M0b remains the separately defined residual origin and source reference.
"""
import numpy as np

from development_lopo import DEVELOPMENT_IDS, M0aPrediction, _array, _support_size
from registered_endpoint_metrics import CONDITIONS, crossvalidated_geometries


class AnatomicalGroupBaseline:
    """Callable as (training_records, context), with no target-specific input."""

    def __init__(self, expected_voxel_ids, *, check, support_size=208):
        self.support_size = _support_size(support_size)
        ids = expected_voxel_ids
        if not isinstance(ids, tuple) or len(ids) != self.support_size:
            raise ValueError("the exact ordered selected anatomical axis is required")
        strings = all(isinstance(value, str) and bool(value) for value in ids)
        integers = all(isinstance(value, (int, np.integer))
                       and not isinstance(value, (bool, np.bool_)) for value in ids)
        if not (strings or integers) or len(set(ids)) != self.support_size:
            raise ValueError("anatomical IDs must be unique homogeneous strings or integers")
        if not callable(check):
            raise TypeError("the caller's allocation/deadline check is required")
        self.voxel_ids, self.check = ids, check

    def __call__(self, training_records, context):
        self.check()
        if (context.target_id not in DEVELOPMENT_IDS
                or context.deleted_id not in (None, *DEVELOPMENT_IDS)
                or context.deleted_id == context.target_id):
            raise ValueError("a valid development target/deletion context is required")
        expected = tuple(person for person in DEVELOPMENT_IDS
                         if person not in (context.target_id, context.deleted_id))
        records = tuple(training_records)
        # Finish every identity/axis check before converting any B arrays.
        if context.training_ids != expected or tuple(record.participant_id for record in records) != expected:
            raise ValueError("M0a training identities must exactly exclude target and deleted participant")
        for record in records:
            if record.voxel_ids != self.voxel_ids or record.task_b_condition_ids != CONDITIONS:
                raise ValueError("M0a requires the unchanged ordered anatomical and 18-condition axes")
        centered = []
        for record in records:
            partitions = _array(record.task_b_partitions, (4, 18, self.support_size), "raw M0a source partitions")
            with np.errstate(over="raise", invalid="raise"):
                centered.append(partitions-partitions.mean(axis=-1, keepdims=True))
        with np.errstate(over="raise", invalid="raise"):
            mean_partitions = np.mean(np.stack(centered), axis=0)
            maps = mean_partitions.mean(axis=0)
            geometry = crossvalidated_geometries(mean_partitions[None])[0].mean(axis=0)
        maps = _array(maps, (18, self.support_size), "M0a mean maps")
        geometry = _array(geometry, (18, 18), "M0a cross-partition geometry")
        self.check()
        return M0aPrediction(maps, geometry, expected)
