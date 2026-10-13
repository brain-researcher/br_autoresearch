"""Independent exact replay, authored from EP13's frozen statement only.

No imports from the main theorem package, fixtures, historical evaluators,
expected labels, or precomputed certificates. Uses the full Kronecker system
for identifiability and constructs a rank-one invisible perturbation if needed.
"""

from fractions import Fraction
import re


def rational(value):
    if type(value) is int:
        return Fraction(value)
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?\d+(?:/[+-]?\d+)?", value):
        raise ValueError("expected an integer or exact rational string")
    parts = value.split("/")
    return Fraction(int(parts[0]), int(parts[1]) if len(parts) == 2 else 1)


def vector(value):
    if not isinstance(value, list) or not value:
        raise ValueError("vector must have positive length")
    return [rational(x) for x in value]


def matrix(value):
    if not isinstance(value, list) or not value:
        raise ValueError("matrix must have positive row count")
    rows = [vector(row) for row in value]
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be rectangular")
    return rows


def transpose(a):
    return [list(col) for col in zip(*a)]


def multiply(a, b):
    if len(a[0]) != len(b):
        raise ValueError("incompatible matrix product")
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("incompatible vector product")
    return sum(x * y for x, y in zip(a, b))


def matvec(a, b):
    return [dot(row, b) for row in a]


def colvec(a):
    return [a[i][j] for j in range(len(a[0])) for i in range(len(a))]


def eliminate(a, coefficient_columns):
    """Exact Gauss-Jordan elimination, returning pivot columns."""
    out = [row[:] for row in a]
    pivots = []
    for col in range(coefficient_columns):
        row = len(pivots)
        pivot = next((i for i in range(row, len(out)) if out[i][col]), None)
        if pivot is None:
            continue
        out[row], out[pivot] = out[pivot], out[row]
        scale = out[row][col]
        out[row] = [x / scale for x in out[row]]
        for i in range(len(out)):
            if i != row and out[i][col]:
                scale = out[i][col]
                out[i] = [x - scale * y for x, y in zip(out[i], out[row])]
        pivots.append(col)
        if len(pivots) == len(out):
            break
    return out, pivots


def solve(a, b):
    if len(a) != len(b):
        raise ValueError("incompatible linear system")
    n = len(a[0])
    reduced, pivots = eliminate([row + [v] for row, v in zip(a, b)], n)
    if any(not any(row[:n]) and row[n] for row in reduced):
        return None
    solution = [Fraction(0) for _ in range(n)]
    for i, col in enumerate(pivots):
        solution[col] = reduced[i][n]
    if matvec(a, solution) != b:
        raise ArithmeticError("linear solve replay failed")
    return solution


def null_direction(a, target):
    reduced, pivots = eliminate(a, len(a[0]))
    for free in range(len(a[0])):
        if free in pivots:
            continue
        z = [Fraction(0) for _ in range(len(a[0]))]
        z[free] = Fraction(1)
        for i, col in enumerate(pivots):
            z[col] = -reduced[i][free]
        if any(matvec(a, z)):
            raise ArithmeticError("kernel replay failed")
        effect = dot(target, z)
        if effect:
            return [x / effect for x in z]
    return None


def unit_readout(target):
    index = next(i for i, value in enumerate(target) if value)
    out = [Fraction(0) for _ in target]
    out[index] = 1 / target[index]
    return out


def strings(value):
    if isinstance(value, list):
        return [strings(x) for x in value]
    return str(value)


def check_case(case):
    case_id = case.get("id") if isinstance(case, dict) else None
    try:
        if not isinstance(case, dict) or not isinstance(case_id, str):
            raise ValueError("case requires a string id")
        a, p, x = (matrix(case[k]) for k in ("A", "P", "X"))
        c, s = (vector(case[k]) for k in ("c", "s"))
        t, v, q, k = len(a), len(a[0]), len(p), len(p[0])
        if len(x) != v or len(x[0]) != k or len(c) != k or len(s) != v:
            raise ValueError("incompatible declared dimensions")
        if "application" in case:
            app = case["application"]
            r, selection, h = (matrix(app[name]) for name in ("R", "S", "H"))
            if len(r) != t or len(r[0]) != t:
                raise ValueError("R must be T by T")
            if multiply(r, r) != r:
                raise ValueError("R must be idempotent")
            if len(selection) != t or len(h[0]) != v:
                raise ValueError("incompatible application dimensions")
            if multiply(multiply(r, selection), h) != a:
                raise ValueError("A must equal R S H")

        y = multiply(multiply(a, x), transpose(p))
        tau = dot(s, matvec(x, c))
        # Column-major Y ordering: spatial observation outside temporal row.
        m = [[p[j][l] * a[i][h] for l in range(k) for h in range(v)]
             for j in range(q) for i in range(t)]
        ell = [c[l] * s[h] for l in range(k) for h in range(v)]
        observation_ok = matvec(m, colvec(x)) == colvec(y)
        target_ok = dot(ell, colvec(x)) == tau
        if not observation_ok or not target_ok:
            raise ArithmeticError("vectorization identity failed")
        w = solve(transpose(m), ell)
        if w is not None:
            certificate = {"kind": "readout", "w": strings(w)}
            certificate_ok = matvec(transpose(m), w) == ell and dot(w, colvec(y)) == tau
        else:
            z = null_direction(a, s)
            if z is not None:
                d = unit_readout(c)
            else:
                d = null_direction(p, c)
                if d is None:
                    raise ArithmeticError("full-system and separable criteria disagree")
                z = unit_readout(s)
            delta = [[zi * dj for dj in d] for zi in z]
            invisible = multiply(multiply(a, delta), transpose(p))
            effect = dot(s, matvec(delta, c))
            certificate_ok = not any(colvec(invisible)) and effect == 1
            certificate = {"kind": "null_witness", "delta_X": strings(delta)}
        if not certificate_ok:
            raise ArithmeticError("certificate replay failed")
        return {"id": case_id, "valid": True, "identifiable": w is not None,
                "Y": strings(y), "tau": str(tau), "observation_identity": observation_ok,
                "target_identity": target_ok, "certificate": certificate,
                "certificate_valid": certificate_ok}
    except (KeyError, TypeError, ValueError, ZeroDivisionError, StopIteration) as exc:
        return {"id": case_id, "valid": False, "error": str(exc) or type(exc).__name__}
