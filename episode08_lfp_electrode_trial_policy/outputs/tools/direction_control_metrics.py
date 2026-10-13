"""Trusted fixed-prediction scores; never called by a candidate fitter."""
import numpy as np


def predict_residual(model, x, directions):
    s = model.statistics
    xx = (x - s.direction_input_mean[directions] - s.input_mean) / s.input_sd
    return s.direction_target_mean[directions] + s.target_sd * (
        s.target_residual_mean + xx @ model.coefficients)


def prediction_scores(prediction, y, directions, target_mean, target_sd):
    target, pred = (y-target_mean)/target_sd, (prediction-target_mean)/target_sd
    denominator = float(np.sum(target ** 2))
    if (prediction.shape != y.shape or not np.isfinite(target).all()
            or not np.isfinite(pred).all() or denominator <= 0 or not np.isfinite(denominator)
            or not np.array_equal(np.unique(directions), np.arange(8))):
        raise ValueError("Complete finite fixed-prediction panel and positive common denominator required")
    correct = float(np.sum((target-pred) ** 2))
    wrong = 0.
    for d in range(8):
        indices = np.flatnonzero(directions == d)
        if len(indices) != 5:
            raise ValueError("Preserve five evaluation trials in each direction")
        # Every off-diagonal (target,prediction) trial pair occurs equally often
        # among derangements. Whole multivariate population rows move together.
        pair_error = np.sum((target[indices, None, :] - pred[None, indices, :]) ** 2, axis=2)
        wrong += float(pair_error[~np.eye(5, dtype=bool)].sum() / 4)
    return {"Q": 1-correct/denominator, "Q_wrong": 1-wrong/denominator,
            "pairing_increment": (wrong-correct)/denominator}
