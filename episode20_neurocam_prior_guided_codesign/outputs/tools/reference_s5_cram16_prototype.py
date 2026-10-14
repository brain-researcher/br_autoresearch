"""Isolated full-augmentation IPF CRAM16 action; not a production backend.

Coefficients and sequential IPF recurrence are from the primary OpenMC source:
https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html
No negative-real-axis scalar bound is asserted for this augmented circuit.
Every state, sine/cosine/constant coordinate and24 hold integrals is propagated.
The time interval and the original R3 rail guard remain unchanged.
"""
from collections import OrderedDict
import math
import time

import numpy as np

from reference_s5_family import AirFamily, OMEGA
from reference_s5_shifted_structured import StructuredShiftedInverse


ALPHA = np.array([
    5.464930576870210e3-3.797983575308356e4j,
    9.045112476907548e1-1.115537522430261e3j,
    2.344818070467641e2-4.228020157070496e2j,
    9.453304067358312e1-2.951294291446048e2j,
    7.283792954673409e2-1.205646080220011e5j,
    3.648229059594851e1-1.155509621409682e2j,
    2.547321630156819e1-2.639500283021502e1j,
    2.394538338734709e1-5.650522971778156e0j], dtype=complex)
THETA = np.array([
    3.509103608414918+8.436198985884374j,
    5.948152268951177+3.587457362018322j,
    -5.264971343442647+16.22022147316793j,
    1.419375897185666+10.92536348449672j,
    6.416177699099435+1.194122393370139j,
    4.993174737717997+5.996881713603942j,
    -1.413928462488886+13.49772569889275j,
    -10.84391707869699+19.27744616718165j], dtype=complex)
ALPHA0 = 2.124853710495224e-16


class Cram16PrototypeAirFamily(AirFamily):
    """An independent experimental family, using only an exact-duration LRU.

    The cached object is a full-state shifted inverse. It does not retain a
    trajectory or reuse rounded physical times. This class is intentionally
    absent from every production entrypoint.
    """
    def __init__(self, *args, inverse_cache_bytes=128*1024**2, **kwargs):
        super().__init__(*args, **kwargs)
        if self.structure not in ("R2", "R3"):
            raise ValueError("CRAM16 prototype requires full-state R2/R3")
        if not isinstance(inverse_cache_bytes, int) or inverse_cache_bytes < 1:
            raise ValueError("positive finite inverse cache required")
        self.inverse_cache_limit = inverse_cache_bytes
        self._inverses = OrderedDict()
        self._inverse_bytes = 0
        self._cram_stats = dict(hits=0, misses=0, evictions=0, constructor_seconds=0.,
            action_seconds=0., actions=0, peak_numeric_cache_bytes=0,
            maximum_phase_endpoint_error=0.)

    def cram_report(self):
        return dict(self._cram_stats, cache_entries=len(self._inverses),
            numeric_cache_bytes=self._inverse_bytes, numeric_cache_limit=self.inverse_cache_limit)

    def _inverse(self, row, slot, duration, index):
        key = (int(row), int(slot), float(duration).hex(), int(index))
        saved = self._inverses.pop(key, None)
        if saved is not None:
            self._inverses[key] = saved
            self._cram_stats["hits"] += 1
            return saved
        self._cram_stats["misses"] += 1
        before = time.perf_counter()
        saved = StructuredShiftedInverse(self, THETA[index], duration, row, slot)
        self._cram_stats["constructor_seconds"] += time.perf_counter()-before
        size = saved.retained_numeric_bytes()
        if size > self.inverse_cache_limit:
            raise MemoryError("one exact shifted inverse exceeds prototype cache limit")
        while self._inverse_bytes+size > self.inverse_cache_limit:
            _, removed = self._inverses.popitem(last=False)
            self._inverse_bytes -= removed.retained_numeric_bytes()
            self._cram_stats["evictions"] += 1
        self._inverses[key] = saved
        self._inverse_bytes += size
        self._cram_stats["peak_numeric_cache_bytes"] = max(
            self._cram_stats["peak_numeric_cache_bytes"], self._inverse_bytes)
        return saved

    def sparse_step(self, state, row, slot, marker_volts, start, end, *, max_products=100000):
        self.check()
        duration = float(end)-float(start)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("positive actual interval required")
        self.certify_controlled_rails(state, row, start)
        _, forcing = self.matrices(row, slot)
        # Same numerical forcing as the original augmented operator, with C
        # moved to the RHS of its exact shifted linear system.
        physical_forcing = np.asarray(self.capacitance @ forcing)
        physical_forcing[:, 2] *= marker_volts
        n = self.size
        value = np.zeros(n+3+self.adc_count)
        value[:n] = state
        value[n:n+3] = math.sin(OMEGA*start), math.cos(OMEGA*start), 1.
        a = duration*OMEGA
        started = time.perf_counter()
        for index, (alpha, theta) in enumerate(zip(ALPHA, THETA)):
            self.check()
            solver = self._inverse(row, slot, duration, index)
            # Positive resolvent: w=(theta*I-h*A)^-1 value. The published
            # (h*A-theta*I)^-1 recurrence consequently has a minus sign below.
            phase = np.array([(theta*value[n]+a*value[n+1])/(theta**2+a*a),
                (-a*value[n]+theta*value[n+1])/(theta**2+a*a), value[n+2]/theta])
            physical = solver.solve(self.capacitance @ value[:n]+duration*(physical_forcing @ phase))
            integral = (value[n+3:]+duration*(self.hold_observer @ physical))/theta
            value[:n] -= 2*np.real(alpha*physical)
            value[n:n+3] -= 2*np.real(alpha*phase)
            value[n+3:] -= 2*np.real(alpha*integral)
        value *= ALPHA0
        self._cram_stats["action_seconds"] += time.perf_counter()-started
        self._cram_stats["actions"] += 1
        phase_exact = np.array([math.sin(OMEGA*end), math.cos(OMEGA*end), 1.])
        self._cram_stats["maximum_phase_endpoint_error"] = max(
            self._cram_stats["maximum_phase_endpoint_error"], float(np.max(np.abs(value[n:n+3]-phase_exact))))
        if not np.all(np.isfinite(value)):
            raise ArithmeticError("nonfinite CRAM16 prototype action")
        self.stats["sparse_exponentials"] += 1
        return value[:n], value[n+3:]
