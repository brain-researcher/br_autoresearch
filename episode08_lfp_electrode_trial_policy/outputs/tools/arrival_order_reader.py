"""Frozen metadata-only postinitializer permutations and canonical fitting rows."""
import random

import numpy as np

from allocation_reader import AllocationTrajectory

ORDER_REPETITIONS = (0, 1, 2)


def order_queues(role, repetition):
    if repetition not in ORDER_REPETITIONS:
        raise ValueError("Only the three prospectively selected order repetitions are allowed")
    queues = []
    for direction, original in enumerate(role["fitting_pool_by_direction"]):
        if len(original) < 12:
            raise ValueError("Twelve fitting rows per direction required")
        tail = list(original[4:12])
        rng = random.Random()
        rng.seed(f"ep08-postinitial-order-v1|20261003|{role['filename']}|{direction}|{repetition}", version=2)
        rng.shuffle(tail)
        queues.append(tuple(original[:4])+tuple(tail))
    if len(queues) != 8:
        raise ValueError("Eight fixed directions required")
    return tuple(queues)


class OrderTrajectory(AllocationTrajectory):
    def __init__(self, source, channels, initial_x, initial_y, queues):
        original = source.role["fitting_pool_by_direction"]
        if (len(queues) != 8 or any(len(q) != 12 or tuple(q[:4]) != tuple(p[:4])
                or len(set(q)) != 12 or set(q) != set(p[:12]) for q, p in zip(queues, original))):
            raise ValueError("Only postinitializer permutations of the original direction pools are permitted")
        super().__init__(source, channels, initial_x, initial_y)
        self.queues = tuple(tuple(q) for q in queues)
        self.canonical = tuple(t for pool in original for t in pool[:12])
        self.labels = {t: d for d, pool in enumerate(original) for t in pool[:12]}

    def observed(self):
        allowed = {t for d, q in enumerate(self.queues) for t in q[:self.counts[d]]}
        if allowed != set(self.x) or allowed != set(self.y):
            raise ValueError("Observed arrays must exactly match the chosen queue prefixes")
        ids = [t for t in self.canonical if t in allowed]
        return (np.stack([self.x[t] for t in ids]), np.stack([self.y[t] for t in ids]),
                np.asarray([self.labels[t] for t in ids]), ids)
