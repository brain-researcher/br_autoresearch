"""Trusted fixed-model prediction only; no target or fitting interface."""
import numpy as np
from scipy.spatial.distance import cdist


def predict_capacity(model, x, directions):
    s = model.statistics
    xx = (x-s.direction_input_mean[directions]-s.input_mean)/s.input_sd
    if model.method == "reduced_rank":
        prediction = xx @ model.coefficients
    else:
        state = model.kernel_state
        if state.sigma_squared == 0:
            prediction = np.zeros((len(xx), model.coefficients.shape[1]))
        else:
            raw = np.exp(-cdist(xx, state.anchors, metric="sqeuclidean")/(2*state.sigma_squared))
            cross = raw-state.column_mean[None, :]-raw.mean(axis=1)[:, None]+state.grand_mean
            prediction = cross @ model.coefficients
    return s.direction_target_mean[directions] + s.target_sd*(s.target_residual_mean+prediction)
