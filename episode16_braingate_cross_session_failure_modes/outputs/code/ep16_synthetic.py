"""Synthetic EP16 participants for the policy's pre-scoring checks (no real data).

Sessions follow a linear encoding x = mu + T y + noise on E channels, with two 64-trial whole-block
sides, radial-8 centre-out-back label directions with cursor jitter, and four clusters of sessions so that
near pairs fall within a cluster and long pairs cross clusters. Scenario changes grow with the cluster index.
"""
from __future__ import annotations

import numpy as np

from ep16_panel import Session

# Four clusters of six sessions: near pairs fall inside a cluster, long pairs cross clusters. With two
# eras the long indicator would equal source-era + target-era effects (rank-deficient design).
CLUSTERS = ((10, 14, 19, 24, 30, 36), (260, 264, 269, 274, 280, 286), (510, 514, 519, 524, 530, 536),
            (760, 764, 769, 774, 780, 786))
# Later clusters with 35-day spacing contribute no near pairs: a decline then reaches long pairs
# without also lowering the near-pair reference of the raw transported contrast.
SPARSE_LATE = (CLUSTERS[0], CLUSTERS[1], (510, 545, 580, 615, 650, 685), (900, 935, 970, 1005, 1040, 1075))
# As above, but each late cluster keeps one near pair so the adjusted design stays full rank.
MOSTLY_SPARSE_LATE = (CLUSTERS[0], CLUSTERS[1], (510, 514, 545, 580, 615, 650), (900, 904, 935, 970, 1005, 1040))
SCENARIOS = ("stationary", "output_rotation", "low_rank", "identity_dropout_noise", "random_dropout_noise",
             "signal_reduction", "coexisting")


def labels(rng, n=64):
    """Alternating outward/return radial-8 directions, peripheral order permuted per cycle, jittered."""
    angles = []
    while len(angles) < n:
        for d in rng.permutation(8):
            angles.extend([d * 45.0, d * 45.0 + 180.0])
    angles = np.radians(np.array(angles[:n]) + rng.normal(0, 6.0, n))
    return np.column_stack((np.cos(angles), np.sin(angles)))


def participant(scenario, seed=0, E=48, amplitude=None, trial_latent=0.6, schedule=None):
    rng = np.random.default_rng(seed)
    amplitude = amplitude or (0.7 if scenario == "coexisting" else 0.55)  # coexisting: stronger base signal
    T0 = rng.normal(0, amplitude, (E, 2))
    mu0 = rng.normal(0, 1, E)
    ranked = np.argsort(-np.linalg.norm(T0, axis=1))
    strong, next_strong = ranked[:12], ranked[12:16]
    U = np.linalg.qr(rng.normal(size=(E, 2)))[0] * 8.0 * amplitude
    sessions, truth = [], {"strong_channels": strong.tolist()}
    # Cluster states are non-monotone in time: a monotone drift is absorbed by the adjusted contrast's
    # source/target session effects (frozen estimand), so a long-gap step needs states that differ
    # between distant clusters without a time trend.
    angles = (0.0, 55.0, -50.0, 35.0)
    decline = (1.0, 0.75, 0.55, 0.4)  # later-session signal amplitude by cluster
    clusters = schedule or {"signal_reduction": SPARSE_LATE, "coexisting": MOSTLY_SPARSE_LATE}.get(scenario, CLUSTERS)
    Cs = [np.zeros((2, 2))] + [rng.normal(size=(2, 2)) for _ in range(3)]
    for c, days in enumerate(clusters):
        phi = np.radians(angles[c])
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        for day in days:
            T, sigma = T0.copy(), np.ones(E)
            if scenario in ("output_rotation", "coexisting"):
                T = T0 @ R.T
            if scenario == "low_rank":
                T = T0 + U @ Cs[c]
            dropped = np.zeros(E, bool)
            if scenario == "identity_dropout_noise" and c in (1, 3):
                dropped[strong] = True         # the same strongly tuned electrodes disappear
                sigma[next_strong] = 2.0       # and four more become noisy
            if scenario == "random_dropout_noise" and c in (1, 3):
                chosen = rng.choice(E, 16, replace=False)
                dropped[chosen[:12]] = True
                sigma[chosen[12:]] = 2.0
            if scenario == "signal_reduction":
                T = decline[c] * T0
            if scenario == "coexisting":
                T = decline[c] * T
            Y = np.stack([labels(rng), labels(rng)])
            latent = rng.normal(0, trial_latent, (2, 64, E))
            noise = lambda: rng.normal(size=(2, 64, E)) * sigma
            X = mu0 + Y @ T.T + latent + noise()
            halves = np.stack([mu0 + Y @ T.T + latent + np.sqrt(2) * noise() for _ in range(2)])
            X[..., dropped] = np.nan
            halves[..., dropped] = np.nan
            octant = np.floor(np.remainder(np.arctan2(Y[..., 1], Y[..., 0]), 2 * np.pi) / (np.pi / 4)).astype(int) % 8
            block = np.stack([np.full(64, 1), np.full(64, 2)])
            time = np.stack([np.arange(64), 1000 + np.arange(64)])
            sessions.append(Session(day, {"sbp": X, "sbp_odd_even": halves}, Y, octant, block, ~dropped, True, time))
    return sessions, truth
