"""Prospective M1 parcel predictors selected on October 7, before any scores.

This array-only reference uses the existing fold-specific functional source,
fixed selected support and common Task-A gain. Anatomy defines 4/8/16
profile cells; target Task A defines soft or hard memberships. It provides
predictions, not spatial smoothness, boundary enrichment or class adequacy.
No reader, payload access, empirical release or additional training is implied.
"""
import numpy as np

from development_lopo import M1Prediction, M1_VARIANTS
from local_anatomical_basis import _partition
from local_functional_source import TASK_A_CONDITIONS, _array, _center, _frozen
from registered_endpoint_metrics import crossvalidated_geometries


class LocalParcelPrediction:
    """Four initial M1 variants sharing source maps and a pre-operator gain.

    Profiles are arithmetic voxel means in anatomy-only recursive median
    cells. The exact existing split/axis/native-ID rule defines depths 2/3/4.
    Each fold's temperature T is the median strictly positive weighted squared
    separation of its Task-A profiles. T must exceed numerical_atol times the
    source's mean-voxel Task-A energy; zero/roundoff-only separation refuses.
    Probabilities have equal parcel priors and softmax(-distance/T), where
    distance compares centered target-A/common_gain to the source profiles.

    Hard ties within numerical_atol*(1+abs(minimum distance/T)) use the first
    anatomy-ordered label and are reported as unidentified, including duplicate
    source profiles. No target B resolves a tie. Soft probabilities are not
    posterior uncertainty calibration or evidence of literal parcel boundaries.
    """

    def __init__(self, functional_source, anatomy):
        self.source_recipe = functional_source
        self.atol = functional_source.atol
        self.voxel_ids = functional_source.voxel_ids
        self.support_size = functional_source.support_size
        coordinates = _array(anatomy.coordinates, (self.support_size, 3), "anatomical coordinates")
        if (anatomy.voxel_ids != self.voxel_ids or not np.array_equal(anatomy.B, functional_source.B)
                or not np.array_equal(anatomy.cell_labels, functional_source.labels)):
            raise ValueError("M1 must retain the selected source's exact anatomical support")
        ids = np.asarray(self.voxel_ids)
        self.cells = {}
        for count, depth in ((4, 2), (8, 3), (16, 4)):
            cells, _ = _partition(coordinates, ids, depth)
            labels = np.empty(self.support_size, dtype=np.int64)
            for label, members in enumerate(cells):
                labels[members] = label
            sizes = tuple(len(members) for members in cells)
            if (len(cells) != count or min(sizes) < 1
                    or set(sizes) - {self.support_size//count, (self.support_size+count-1)//count}
                    or sorted(np.concatenate(cells).tolist()) != list(range(self.support_size))):
                raise ValueError("all fixed-support voxels must enter balanced anatomical profile cells")
            if count == 8 and not np.array_equal(labels, functional_source.labels):
                raise ValueError("reference M1 and M3 anatomical splits must agree")
            labels.flags.writeable = False
            self.cells[count] = labels

    def _profiles(self, source, count):
        labels = self.cells[count]
        A = _center(_array(source.task_a_maps, (29, self.support_size), "source Task A"))
        partitions = _center(_array(source.geometry_state.source_partitions, (4, 18, self.support_size), "source B partitions"))
        profiles_a = np.column_stack([A[:, labels == label].mean(axis=-1) for label in range(count)])
        profiles_b = np.stack([partitions[..., labels == label].mean(axis=-1) for label in range(count)], axis=-1)
        separation = np.sum(source.task_a_weights[:, None, None]
                            *(profiles_a[:, :, None]-profiles_a[:, None, :])**2, axis=0)
        pairs = separation[np.triu_indices(count, k=1)]
        positive = pairs[pairs > 0]
        energy = float(np.mean(np.sum(source.task_a_weights[:, None]*A**2, axis=0)))
        temperature = float(np.median(positive)) if len(positive) else 0.
        if (not np.isfinite(pairs).all() or not np.isfinite(energy) or energy <= 0
                or not np.isfinite(temperature) or temperature <= self.atol*energy):
            raise ValueError("source-A parcel profiles have no informative positive temperature")
        duplicate_pairs = int(np.sum(pairs <= self.atol*energy))
        return profiles_a, profiles_b, temperature, duplicate_pairs

    def __call__(self, source, target_task_a, common_gain, context):
        self.source_recipe._source(source, context)
        if source.task_a_condition_ids != TASK_A_CONDITIONS:
            raise ValueError("the complete exact Task-A roster is required")
        source_a = _center(_array(source.task_a_maps, (29, self.support_size), "source Task A"))
        target_a = _center(_array(target_task_a, (29, self.support_size), "target Task A"))
        gain = np.asarray(common_gain)
        if gain.shape != () or gain.dtype.kind not in "fiu" or not np.isfinite(gain) or gain <= 0:
            raise ValueError("one supplied positive finite common gain is required")
        gain = float(gain)
        energy = float(np.sum(source.task_a_weights[:, None]*source_a**2))
        expected_gain = float(np.sum(source.task_a_weights[:, None]*source_a*target_a)/energy) if energy > 0 else np.nan
        if (not np.isfinite(expected_gain) or expected_gain <= 0
                or abs(gain-expected_gain) > self.atol*max(1., abs(expected_gain))):
            raise ValueError("M1 must use the already fixed untransformed-source Task-A gain")
        calibrated = target_a/gain
        if not np.isfinite(calibrated).all():
            raise ArithmeticError("nonfinite common-gain calibration")
        fitted = {}
        for count in (4, 8, 16):
            profiles_a, profiles_b, temperature, duplicate_pairs = self._profiles(source, count)
            distance = np.sum(source.task_a_weights[:, None, None]
                              *(calibrated[:, :, None]-profiles_a[:, None, :])**2, axis=0)/temperature
            if not np.isfinite(distance).all():
                raise ArithmeticError("nonfinite Task-A assignment distance")
            minimum = distance.min(axis=1, keepdims=True)
            tied = distance <= minimum+self.atol*(1+np.abs(minimum))
            labels = np.argmax(tied, axis=1)
            probability = np.exp(-(distance-minimum))
            probability /= probability.sum(axis=1, keepdims=True)
            hard = np.eye(count)[labels]
            fitted[count] = (profiles_b, probability, hard, temperature, duplicate_pairs,
                             tuple(int(value) for value in np.flatnonzero(tied.sum(axis=1) > 1)))
        predictions = []
        for name, count, assignment in zip(M1_VARIANTS, (8, 8, 4, 16), ("probabilistic", "hard", "probabilistic", "probabilistic")):
            profiles_b, probability, hard, temperature, duplicate_pairs, ambiguous = fitted[count]
            memberships = hard if assignment == "hard" else probability
            # The identical target-A memberships and common gain act on every
            # B condition and every source partition, then common spatial
            # centering is applied. This is not a spatially regularized fit.
            partitions = _center(gain*(profiles_b@memberships.T))
            maps = partitions.mean(axis=0)
            geometry = crossvalidated_geometries(partitions[None])[0].mean(axis=0)
            diagnostics = {"parcel_count": count, "assignment": assignment,
                "memberships": _frozen(memberships), "temperature": temperature,
                "temperature_rule": "median_strictly_positive_weighted_source_A_profile_squared_separation",
                "source_profile_numerically_duplicate_pairs": duplicate_pairs,
                "hard_assignment_ambiguous_voxel_indices": ambiguous,
                "hard_assignment_identified": not ambiguous,
                "hard_tie_rule": "first_anatomy_ordered_label_within_fixed_numerical_tolerance",
                "common_gain": gain, "training_ids": context.training_ids,
                "spatial_smoothness_certified": False, "native_signature_certified": False,
                "class_adequacy_claimed": False, "empirical_execution_authorized": False}
            predictions.append(M1Prediction(name, _frozen(maps), _frozen(geometry), diagnostics))
        return tuple(predictions)
