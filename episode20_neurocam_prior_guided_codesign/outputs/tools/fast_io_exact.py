"""Vectorized, bit-exact replicas of legacy scalar arithmetic (EP20 O1/O2).

IEEE binary64 addition, subtraction, multiplication and division round the
same way in NumPy element-wise kernels and in CPython. Each function below
therefore reproduces one named legacy scalar computation exactly, provided it
keeps the legacy operation order and special cases; nothing here chooses a
time, tolerance, threshold or rounding rule. NumPy element-wise kernels never
fuse a multiply and an add, so no FMA can enter these replicas.
"""
import numpy as np


# repeating_acquisition_schedule.CYCLE_SECONDS / CYCLE_COUNT (contract constants).
CYCLE_SECONDS = 0.004
CYCLE_COUNT = 4000


def shift(cycles, relative):
    """repeating_acquisition_schedule._shift, broadcast over cycles and times.

    Legacy: 0.0 -> cycle*CYCLE_SECONDS; CYCLE_SECONDS -> (cycle+1)*CYCLE_SECONDS;
    otherwise cycle*CYCLE_SECONDS + relative. Python int*float converts the
    integer exactly, as float64 conversion of int64 does below 2**53.
    """
    cycles = np.asarray(cycles, dtype=np.int64)
    relative = np.asarray(relative, dtype=np.float64)
    base = cycles.astype(np.float64) * CYCLE_SECONDS
    following = (cycles + 1).astype(np.float64) * CYCLE_SECONDS
    base, following, relative = np.broadcast_arrays(base, following, relative)
    result = base + relative
    result = np.where(relative == 0.0, base, result)
    return np.where(relative == CYCLE_SECONDS, following, result)


def window_ends():
    """(m+1)*CYCLE_SECONDS for every window m, as in the legacy _last_window."""
    return (np.arange(CYCLE_COUNT, dtype=np.int64) + 1).astype(np.float64) * CYCLE_SECONDS


_ENDS = window_ends()
_ENDS.setflags(write=False)


def last_window(times):
    """repeating_acquisition_schedule._last_window: first window end >= time, capped."""
    index = np.searchsorted(_ENDS, np.asarray(times, dtype=np.float64), side="left")
    return np.minimum(index, CYCLE_COUNT - 1).astype(np.int64)


def upper_result(values):
    """raw_world_time._upper_result(value) for nonnegative values, exact_zero=False."""
    result = np.nextafter(np.asarray(values, dtype=np.float64), np.inf)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("nonnegative bound arithmetic is not representable")
    return result


def upper_mul(left, right):
    """raw_world_time._upper_mul: exact zero if either factor is zero, else nextafter up."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    zero = (left == 0) | (right == 0)
    result = np.where(zero, 0.0, np.nextafter(left * right, np.inf))
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("nonnegative bound arithmetic is not representable")
    return result


def upper_div(left, right):
    """raw_world_time._upper_div: exact zero numerator stays zero, else nextafter up."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.any(right <= 0):
        raise ValueError("bound denominator must be positive")
    result = np.where(left == 0, 0.0, np.nextafter(left / right, np.inf))
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("nonnegative bound arithmetic is not representable")
    return result


def midpoint(start, end):
    """raw_input_hold._midpoint for arrays: (midpoint, outward radius).

    midpoint = start + (end - start) / 2 and radius = max of the two outward
    rounded half-widths. A non-interior midpoint refuses, as in the legacy code.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    middle = start + (end - start) / 2
    if not np.all((start < middle) & (middle < end)):
        raise ValueError("no representable interior midpoint; tolerance is not relaxed")
    radius = np.maximum(upper_result(middle - start), upper_result(end - middle))
    return middle, radius


def fsum_rows(values, counts=None):
    """CPython 3.12 math.fsum applied independently to each row, bit for bit.

    values[n, width] holds finite summands; row i uses its first counts[i]
    entries in column order (the legacy iteration order). This follows the
    C implementation exactly: Shewchuk msum with nonzero roundoff partials
    compacted to the front, a carried value appended only when nonzero, then a
    top-down exact summation that stops at the first inexact step and the
    half-even correction across partials. A zero total is therefore +0.0.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("fsum_rows needs a two-dimensional summand array")
    n, width = x.shape
    counts = np.full(n, width, dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
    if counts.shape != (n,) or np.any(counts < 0) or np.any(counts > width):
        raise ValueError("row term counts must lie in [0, width]")
    used_terms = np.arange(width)[None, :] < counts[:, None]
    if not np.all(np.isfinite(x[used_terms])):
        raise ValueError("fsum_rows requires finite summands")
    rows = np.arange(n)
    partials = np.zeros((n, max(width, 1)))
    used = np.zeros(n, dtype=np.int64)
    for column in range(width):
        active = column < counts
        carry = x[:, column].copy()
        rebuilt = np.zeros_like(partials)
        kept = np.zeros(n, dtype=np.int64)
        for j in range(column):
            present = active & (j < used)
            y = partials[:, j]
            swap = np.abs(carry) < np.abs(y)
            big = np.where(swap, y, carry)
            small = np.where(swap, carry, y)
            hi = big + small
            lo = small - (hi - big)
            emit = present & (lo != 0.0)
            rebuilt[rows[emit], kept[emit]] = lo[emit]
            kept = kept + emit
            carry = np.where(present, hi, carry)
        if not np.all(np.isfinite(carry[active])):
            raise OverflowError("intermediate overflow in fsum")
        append = active & (carry != 0.0)
        rebuilt[rows[append], kept[append]] = carry[append]
        kept = kept + append
        partials = np.where(active[:, None], rebuilt, partials)
        used = np.where(active, kept, used)
    has = used > 0
    remaining = np.where(has, used - 1, 0)
    hi = np.where(has, partials[rows, np.maximum(used - 1, 0)], 0.0)
    lo = np.zeros(n)
    looping = has & (remaining > 0)
    for _ in range(width):
        if not looping.any():
            break
        y = partials[rows, np.maximum(remaining - 1, 0)]
        total = hi + y
        low = y - (total - hi)
        hi = np.where(looping, total, hi)
        lo = np.where(looping, low, lo)
        remaining = np.where(looping, remaining - 1, remaining)
        looping = looping & (low == 0.0) & (remaining > 0)
    previous = partials[rows, np.maximum(remaining - 1, 0)]
    adjust = (remaining > 0) & (((lo < 0.0) & (previous < 0.0)) | ((lo > 0.0) & (previous > 0.0)))
    doubled = lo * 2.0
    candidate = hi + doubled
    exact = doubled == (candidate - hi)
    return np.where(adjust & exact, candidate, hi)


def float_bits(values):
    """Raw IEEE bit patterns, so comparisons distinguish -0.0 and NaN payloads."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(np.uint64)


def bitwise_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind == "f" or right.dtype.kind == "f":
        return bool(np.array_equal(float_bits(left), float_bits(right)))
    return bool(np.array_equal(left, right))
