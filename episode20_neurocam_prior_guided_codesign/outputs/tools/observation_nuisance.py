"""Adopted N1–N4 decoded-observation challenges using saved nuisance draws.

The October 9 recipe places these AFTER digital arithmetic/quantization and
BEFORE valid-pixel counters/publication callbacks. It is not analog pickup,
ADC saturation, changed circuitry or a inferred physical DAQ configuration.
Truth, aperture support, physical costs, bit costs, clipping and timestamps are
unchanged. An explicit binding declares a single physical site and acquisition
point; no site, midpoint, assigned-time rule or publication-time substitution
is inferred. Multi-site outputs require their separately declared model.
"""

from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

import numpy as np

from acquisition_readout import Observation, PublishedValue
from raw_world_time import RawWorldTime


def _time(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


@dataclass(frozen=True)
class ObservationNuisanceBinding:
    observation_id: str
    physical_site_index: int
    acquisition_time_s: float

    def __post_init__(self):
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("An explicit observation identity is required")
        if (isinstance(self.physical_site_index, (bool, np.bool_))
                or not isinstance(self.physical_site_index, Integral)
                or not 0 <= self.physical_site_index < 4096):
            raise ValueError("An explicit row-major physical site index 0..4095 is required")
        point = _time(self.acquisition_time_s, "acquisition query point")
        if not 0 <= point < 16.:
            raise ValueError("Nuisance query point must lie in the half-open record [0,16)")
        object.__setattr__(self, "physical_site_index", int(self.physical_site_index))
        object.__setattr__(self, "acquisition_time_s", point)


class SavedObservationNuisance:
    """Reuse one validated RawWorldTime snapshot; never read/regenerate worlds.

    ``validate_binding`` is purely metadata work for whole-plan preflight and
    replay. Only ``apply`` queries saved nuisance state. Bindings assert the
    producer's site-labelled acquisition point, and must use one declared
    convention across arms. This component cannot authenticate a physical
    channel map or infer the site of an arbitrary mixture.
    """

    def __init__(self, raw_time):
        if not isinstance(raw_time, RawWorldTime):
            raise ValueError("An existing validated RawWorldTime snapshot is required")
        metadata = raw_time.metadata
        condition = metadata["world_id"].split(":", 1)[0]
        if condition not in ("N1", "N2", "N3", "N4"):
            raise ValueError("This adapter applies only the retained N1–N4 nuisance definitions")
        self._raw_time = raw_time
        self._metadata = dict(world_id=metadata["world_id"],
            independence_group=metadata["independence_group"], condition_id=condition,
            placement="decoded_site_labelled_pixel_after_digital_arithmetic_and_quantization",
            query="explicit_acquisition_point_in_contributing_raw_aperture_support",
            nonpixel_outputs_unchanged=True, truth_and_evaluation_mask_unchanged=True,
            physical_state_and_costs_unchanged=True, hardware_clipping_flags_unchanged=True,
            actual_DAQ_configuration_qualified=False, physical_line_pickup_modeled=False,
            electrical_dead_row_modeled=False, new_nuisance_draws=False)

    @property
    def metadata(self):
        return dict(self._metadata)

    def validate_binding(self, binding, observation, contributing_apertures):
        """Validate metadata only; aperture endpoints are allowed, global16 is not.

        The point must be inside at least one contributing raw aperture, not
        merely inside the bounding interval around several disjoint apertures.
        Availability is an upper bound, never an automatically chosen query.
        No saved nuisance values, masks or line evaluations are accessed here.
        """
        if (not isinstance(binding, ObservationNuisanceBinding)
                or not isinstance(observation, (Observation, PublishedValue))
                or observation.role != "pixel" or binding.observation_id != observation.observation_id
                or not isinstance(observation.site_id, str) or not observation.site_id):
            raise ValueError("Each affected site-labelled pixel needs its own explicit binding")
        point = binding.acquisition_time_s
        if point > _time(observation.available_at, "observation availability"):
            raise ValueError("Nuisance acquisition query cannot occur after publication")
        supported = False
        for item in contributing_apertures:
            if len(item) != 3:
                raise ValueError("Each contributing raw aperture needs ID, start and end")
            _, start, end = item
            start, end = _time(start, "aperture start"), _time(end, "aperture end")
            if start >= end:
                raise ValueError("Contributing raw apertures must have positive duration")
            supported = supported or start <= point <= end
        if not supported:
            raise ValueError("Nuisance acquisition point must lie inside contributing raw aperture support")

    def apply(self, value, binding):
        """Return one observation; modify only decoded value and/or validity."""
        if not isinstance(value, PublishedValue):
            raise ValueError("An ordinary decoded PublishedValue is required")
        if value.role != "pixel":
            if binding is not None:
                raise ValueError("Non-pixel outputs must have no nuisance binding")
            return value
        self.validate_binding(binding, value, value.contributing_apertures)
        if not isinstance(value.valid, bool):
            raise ValueError("Decoded observation validity must be explicit")
        if not value.valid:
            if value.value_volts is not None:
                raise ValueError("An invalid decoded observation must remain absent")
            return value
        original = _time(value.value_volts, "decoded observation voltage")
        nuisance = self._raw_time.nuisance_at_pairs(
            np.array([binding.acquisition_time_s]), site_indices=np.array([binding.physical_site_index], dtype=np.int64))
        condition = self._metadata["condition_id"]
        if condition in ("N1", "N2"):
            adjusted = original + float(nuisance["spatially_common_line_uv"][0]) / 1e6
            if not math.isfinite(adjusted):
                raise ValueError("Nuisance-adjusted observation is not representable")
            return replace(value, value_volts=adjusted)
        if bool(nuisance["missing_sites"][0]):
            return replace(value, value_volts=None, valid=False)
        return value
