"""Sixteen diverse starting pairs, not a closed EP07 model shortlist.

These configurations precede all development feedback. Each stream receives
sixteen distinct configurations with the same representation/rank/mapping
opportunities. Its own data fit every parameter. Later adaptive successors may
explore the full registered grammar, including learned source weighting, set
encoders and two-member ensembles. No candidate is chosen per day or region.
The common configuration is fitted separately in each authorized setting.
"""
from lfp_features import FeatureRecipe
from primary_model import PrimaryRecipe
from regression import RegressionRecipe


def starting_pair(index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 16:
        raise ValueError("starting pair index must be 0..15")
    if index < 8:
        family = "ridge" if index < 4 else "reduced_rank_ridge"
    elif index < 12:
        family = "elastic_net"
    else:
        family = "shallow_network"
    variant = index % 4
    rank = (4, 8, 12, 16)[variant]
    feature = (FeatureRecipe(quantiles=(.25, .5, .75), scaling="robust")
               if variant % 2 == 0 else
               FeatureRecipe(family="robust_histogram", histogram_bins=4 if variant == 1 else 8,
                             histogram_quantiles=(.01, .99), scaling="robust"))
    adjustment = ("explicit", "residualize", "both", "explicit")[(variant + index // 4) % 4]
    penalty = (.001, .01, .1, 1.)[variant]
    if family == "ridge":
        regression = RegressionRecipe(family, penalty)
    elif family == "reduced_rank_ridge":
        regression = RegressionRecipe(family, penalty, rank=(2, 4, 6, 8)[variant])
    elif family == "elastic_net":
        regression = RegressionRecipe(family, penalty, l1_ratio=(.1, .5, .9, 1.)[variant],
                                      max_iterations=1000, tolerance=1e-5)
    else:
        regression = RegressionRecipe(family, penalty, width=(16, 32, 48, 64)[variant],
            activation="tanh" if variant % 2 == 0 else "relu",
            seed=700700 + index, learning_rate=.01, epochs=300)
    alignment = ("none", "orthogonal", "regularized_correlation", "orthogonal")[variant]
    history = PrimaryRecipe(feature, rank, alignment,
        .1 if alignment == "regularized_correlation" else None, adjustment,
        regression, calibration_mass=1., source_masses=(1., 1., 1.))
    today = PrimaryRecipe(feature, rank, "none", None, adjustment, regression,
                          calibration_mass=1., source_masses=())
    return history, today
