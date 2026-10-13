"""Fourteen selected EP07 adaptive paired opportunities after starting feedback.

These globally fixed choices follow the fifteen allowed completed starting
pairs. They explore the registered grammar; they are not retries of failed
starting08, a final shortlist, or a claim that the whole search is complete.
Each non-ensemble recipe is fitted independently in every setting and stream.
Ensemble membership/weights are global within stream, never chosen per day.
No feedback or payload is read by this declaration module.
"""
from dataclasses import dataclass

from lfp_features import FeatureRecipe
from primary_model import PrimaryRecipe
from regression import RegressionRecipe
from source_weighting import SourceWeightRecipe


@dataclass(frozen=True)
class AdaptiveStreamRecipe:
    kind: str
    primary: PrimaryRecipe | None
    source_weighting: SourceWeightRecipe | None = None
    member_ids: tuple[str, str] | None = None
    member_weights: tuple[float, float] | None = None


@dataclass(frozen=True)
class AdaptivePair:
    index: int
    candidate_id: str
    label: str
    streams: tuple[AdaptiveStreamRecipe, AdaptiveStreamRecipe]


def adaptive_pair(index):
    """Return history-assisted then today-only declarations for index0..13."""
    if isinstance(index, bool) or not isinstance(index, int) or index not in range(14):
        raise ValueError("adaptive index must be an integer in0..13")
    candidate_id = f"adaptive_{index:02d}"
    if index in (3, 7):
        second = "starting_00" if index == 3 else "adaptive_02"
        return AdaptivePair(index, candidate_id, "D" if index == 3 else "H", tuple(
            AdaptiveStreamRecipe("ensemble", None, member_ids=members,
                                 member_weights=(.5, .5))
            for members in (("starting_13", second), ("starting_09", second))))
    histogram = index in (0, 1, 5, 12)
    features = (FeatureRecipe(family="robust_histogram", histogram_bins=4,
                    histogram_quantiles=(.01, .99), scaling="robust") if histogram else
                FeatureRecipe(quantiles=(.25, .5, .75), scaling="robust"))
    if index in (0, 2, 4):
        regression = RegressionRecipe("ridge", .003 if index == 2 else .01)
    elif index == 8:
        regression = RegressionRecipe("reduced_rank_ridge", .001, rank=2)
    elif index == 9:
        regression = RegressionRecipe("ridge", .005)
    elif index in (10, 12, 13):
        regression = RegressionRecipe("ridge", .003 if index == 13 else .0003)
    else:
        regression = RegressionRecipe("shallow_network", {1: .03, 5: .1, 6: .01, 11: .003}[index],
            width=32 if index in (5, 11) else 16, activation="tanh" if index == 6 else "relu",
            seed=700712 if index == 6 else 700713, learning_rate=.01, epochs=300)
    weight_recipe = None
    if index in (2, 10):
        weight_recipe = SourceWeightRecipe("calibration_similarity", target_mass=1.,
            source_total_mass=1., weight_floor=.1, similarity_temperature=1., feature_scale_floor=.1)
    elif index == 4:
        weight_recipe = SourceWeightRecipe("reliability", target_mass=1.,
            source_total_mass=3., weight_floor=.1, reliability_power=1.)
    elif index == 9:
        weight_recipe = SourceWeightRecipe("hierarchical_shrinkage", target_mass=1.,
            source_total_mass=1., weight_floor=.1, reliability_power=1., shrinkage_strength=120.)
    streams = []
    for history in (True, False):
        primary = PrimaryRecipe(features, 8 if histogram or index == 13 else 4,
            "orthogonal" if history and (histogram or index in (10, 11, 13)) else "none",
            None, "explicit", regression, calibration_mass=1.,
            source_masses=((1 / 3,) * 3 if index in (11, 12, 13) else (1., 1., 1.)) if history else ())
        weighted = weight_recipe if history else None
        streams.append(AdaptiveStreamRecipe("weighted_primary" if weighted else "primary",
                                             primary, weighted))
    return AdaptivePair(index, candidate_id, "ABCDEFGHIJKLMN"[index], tuple(streams))
