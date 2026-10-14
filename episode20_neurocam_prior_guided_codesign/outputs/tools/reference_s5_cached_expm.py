"""Isolated S5 sparse-action plan reuse; the live family is unchanged.

Only SciPy 1.12's state-independent scalar scaling/Taylor plan is cached.
The original augmented operator, exact duration, forcing, phase, hold integral,
rail certificate and per-interval product limit are retained. Norm estimates
retain SciPy's meaning; they are not asserted to be upper bounds. This uses
private SciPy APIs deliberately and refuses a different installed version.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import time

import numpy as np
import scipy
from scipy.sparse.linalg._expm_multiply import (
    LazyOperatorNormInfo, _expm_multiply_simple_core, _fragment_3_1,
    _ident_like, onenormest)

from reference_s5_family import AirFamily, OMEGA, _SineIntegralOperator


TOLERANCE = 2 ** -53
ALGORITHM = "scipy1.12_simple_t1_n0=1_float64"


@dataclass(frozen=True)
class ScalarPlan:
    mu: float
    m_star: int
    scaling: int


class CachedAirFamily(AirFamily):
    """One immutable physical family with an entry-bounded scalar-plan LRU.

    As with AirFamily's matrix cache, circuit parameters, dimensions, probe,
    capacitance/factorization, projection and observers are construction-time
    values: create a new family when any of them changes. No operator, norm
    estimator or state vector is retained in this additional cache.
    """
    def __init__(self, *args, plan_cache_entries=8192, **kwargs):
        if scipy.__version__ != "1.12.0":
            raise RuntimeError("cached S5 action requires the reviewed SciPy 1.12.0 implementation")
        if (not isinstance(plan_cache_entries, int) or isinstance(plan_cache_entries, bool)
                or plan_cache_entries < 1):
            raise ValueError("a positive finite plan-cache entry cap is required")
        super().__init__(*args, **kwargs)
        self.plan_cache_entries = plan_cache_entries
        self._action_plans = OrderedDict()
        self.plan_stats = dict(hits=0, misses=0, evictions=0, setup_seconds=0.,
                               core_seconds=0., setup_products=0, core_products=0)

    def plan_report(self):
        return dict(self.plan_stats, entries=len(self._action_plans),
                    maximum_entries=self.plan_cache_entries, algorithm=ALGORITHM)

    def _shift_and_plan(self, operator, row, slot, marker, duration, trace):
        # duration is the actual end-start float, never rounded to a nominal
        # aperture. Start phase belongs to the RHS, not this matrix identity.
        key = (self.row_key(row), int(slot), float(marker).hex(), duration.hex(),
               operator.shape, operator.dtype.str, ALGORITHM)
        plan = self._action_plans.pop(key, None)
        if plan is not None:
            self._action_plans[key] = plan
            self.plan_stats["hits"] += 1
            return operator - plan.mu * _ident_like(operator), plan

        self.plan_stats["misses"] += 1
        started, products = time.monotonic(), operator.products
        try:
            # This is _expm_multiply_simple's t=1, n0=1 setup, with the same
            # supplied trace shift and random-probe norm estimators. Keeping a
            # successful plan changes no physical matrix or Taylor stopping rule.
            mu = trace / float(operator.shape[0])
            shifted = operator - mu * _ident_like(operator)
            norm_estimate = onenormest(shifted)
            if norm_estimate == 0:
                m_star, scaling = 0, 1
            else:
                info = LazyOperatorNormInfo(1.0 * shifted, A_1_norm=norm_estimate, ell=2)
                m_star, scaling = _fragment_3_1(info, 1, TOLERANCE, ell=2)
            plan = ScalarPlan(float(mu), int(m_star), int(scaling))
            self._action_plans[key] = plan
            if len(self._action_plans) > self.plan_cache_entries:
                self._action_plans.popitem(last=False)
                self.plan_stats["evictions"] += 1
            return shifted, plan
        finally:
            self.plan_stats["setup_seconds"] += time.monotonic() - started
            self.plan_stats["setup_products"] += operator.products - products

    def sparse_step(self, state, row, slot, marker_volts, start, end, *, max_products=100000):
        self.check()
        duration = float(end) - float(start)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("positive finite sparse interval required")
        # The rail proof depends on current state/time, so it runs on cache hits.
        self.certify_controlled_rails(state, row, start)
        g, forcing = self.matrices(row, slot)
        operator = _SineIntegralOperator(self, g, forcing, marker_volts, duration, max_products)
        n = self.size
        initial = np.zeros(operator.shape[0])
        initial[:n] = state
        initial[n:n + 3] = (math.sin(OMEGA * start), math.cos(OMEGA * start), 1.)
        trace = -duration * float(np.sum(g.diagonal() / self.capacitance.diagonal()))
        shifted, plan = self._shift_and_plan(operator, row, slot, marker_volts, duration, trace)
        started, products = time.monotonic(), operator.products
        try:
            value = _expm_multiply_simple_core(shifted, initial, 1.0, plan.mu,
                plan.m_star, plan.scaling, TOLERANCE, False)
        finally:
            self.plan_stats["core_seconds"] += time.monotonic() - started
            self.plan_stats["core_products"] += operator.products - products
        if not np.all(np.isfinite(value)):
            raise ArithmeticError("nonfinite dry-air propagated state")
        self.stats["sparse_exponentials"] += 1
        return value[:n], value[n + 3:]
