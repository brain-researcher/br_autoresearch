"""Exact EP13 component certificates; rational arithmetic, no scientific packages."""

from fractions import Fraction as F
import re


class InvalidCase(ValueError):
    pass


def rational(value):
    if type(value) is int:
        return F(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:/[+-]?\d+)?", value):
        try:
            return F(value)
        except (ValueError, ZeroDivisionError):
            pass
    raise InvalidCase("expected exact integer or rational string")


def matrix(value):
    if not isinstance(value, list) or not value or any(not isinstance(r, list) or not r for r in value):
        raise InvalidCase("matrix dimensions must be positive")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise InvalidCase("ragged matrix")
    return [[rational(x) for x in row] for row in value]


def vector(value, length):
    if not isinstance(value, list) or len(value) != length:
        raise InvalidCase("incompatible vector shape")
    return [rational(x) for x in value]


def tr(a):
    return [list(row) for row in zip(*a)]


def dot(a, b):
    assert len(a) == len(b)
    return sum((x * y for x, y in zip(a, b)), F(0))


def mul(a, b):
    assert len(a[0]) == len(b)
    return [[dot(row, col) for col in tr(b)] for row in a]


def mv(a, b):
    return [dot(row, b) for row in a]


def vec(a):
    return [x for col in tr(a) for x in col]


def kron(a, b):
    return [[x * y for x in ar for y in br] for ar in a for br in b]


def tensor(a, b):
    return [x * y for x in a for y in b]


def rref(a, columns=None):
    out = [row[:] for row in a]
    width = len(a[0]) if columns is None else columns
    pivot_columns = []
    next_row = 0
    for col in range(width):
        pivot = next((r for r in range(next_row, len(out)) if out[r][col]), None)
        if pivot is None:
            continue
        out[next_row], out[pivot] = out[pivot], out[next_row]
        scale = out[next_row][col]
        out[next_row] = [x / scale for x in out[next_row]]
        for row in range(len(out)):
            if row != next_row and out[row][col]:
                factor = out[row][col]
                out[row] = [x - factor * y for x, y in zip(out[row], out[next_row])]
        pivot_columns.append(col)
        next_row += 1
        if next_row == len(out):
            break
    return out, pivot_columns


def rank(a):
    return len(rref(a)[1])


def solve(a, b):
    n = len(a[0])
    reduced, pivots = rref([row + [target] for row, target in zip(a, b)], n)
    if any(not any(row[:n]) and row[n] for row in reduced):
        return None
    answer = [F(0)] * n
    for row, col in enumerate(pivots):
        answer[col] = reduced[row][-1]
    assert mv(a, answer) == b
    return answer


def null(a):
    reduced, pivots = rref(a)
    n = len(a[0])
    result = []
    for free in sorted(set(range(n)) - set(pivots)):
        v = [F(0)] * n
        v[free] = F(1)
        for row, col in enumerate(pivots):
            v[col] = -reduced[row][free]
        assert not any(mv(a, v))
        result.append(v)
    return result


def parse(case):
    try:
        a, p, x = (matrix(case[k]) for k in ("A", "P", "X"))
        s, c = vector(case["s"], len(a[0])), vector(case["c"], len(p[0]))
    except KeyError as error:
        raise InvalidCase("missing required matrix or vector") from error
    if len(x) != len(s) or len(x[0]) != len(c):
        raise InvalidCase("incompatible latent matrix shape")
    if "application" in case:
        app = case["application"]
        try:
            r, b, h = (matrix(app[k]) for k in ("R", "S", "H"))
        except (KeyError, TypeError) as error:
            raise InvalidCase("incomplete application factorization") from error
        if len(r) != len(a) or len(r[0]) != len(a):
            raise InvalidCase("R must be T by T")
        if len(b) != len(a) or len(b[0]) != len(h) or len(h[0]) != len(s):
            raise InvalidCase("incompatible application factorization")
        if mul(r, r) != r:
            raise InvalidCase("application R must be idempotent")
        if mul(mul(r, b), h) != a:
            raise InvalidCase("A differs from R S H")
    return a, p, x, s, c


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    return value


def dual_coordinate(v):
    result = [F(0)] * len(v)
    index = next(i for i, x in enumerate(v) if x)
    result[index] = 1 / v[index]
    return result


def normalized_kernel(a, v):
    result = next(z for z in null(a) if dot(z, v))
    scale = dot(result, v)
    return [x / scale for x in result]


def replay_certificate(case, certificate):
    a, p, x, s, c = parse(case)
    y, tau = mul(mul(a, x), tr(p)), dot(s, mv(x, c))
    m, ell = kron(p, a), tensor(c, s)
    if certificate["kind"] == "readout":
        w = vector(certificate["w"], len(m))
        assert mv(tr(m), w) == ell
        assert dot(w, vec(y)) == tau
    elif certificate["kind"] == "null_witness":
        delta = matrix(certificate["delta_X"])
        assert len(delta) == len(x) and len(delta[0]) == len(x[0])
        assert rank(delta) == 1
        assert not any(vec(mul(mul(a, delta), tr(p))))
        assert dot(s, mv(delta, c)) == 1
        other = [[a0 + b0 for a0, b0 in zip(ar, br)] for ar, br in zip(x, delta)]
        assert mul(mul(a, other), tr(p)) == y
        assert dot(s, mv(other, c)) == tau + 1
    else:
        raise AssertionError("unknown certificate kind")
    return True


def evaluate(case):
    identifier = case.get("id", "unnamed")
    try:
        a, p, x, s, c = parse(case)
    except InvalidCase as error:
        return {"id": identifier, "valid": False, "error": str(error)}
    m, ell = kron(p, a), tensor(c, s)
    y, tau = mul(mul(a, x), tr(p)), dot(s, mv(x, c))
    observation_identity = mv(m, vec(x)) == vec(y)
    target_identity = dot(ell, vec(x)) == tau
    assert observation_identity and target_identity
    v, u = solve(tr(a), s), solve(tr(p), c)
    identifiable = not any(c) or not any(s) or (v is not None and u is not None)
    general_w = solve(tr(m), ell)
    criteria = [rank(m + [ell]) == rank(m), all(dot(z, ell) == 0 for z in null(m)), general_w is not None]
    assert all(x == identifiable for x in criteria)
    if identifiable:
        w = [F(0)] * len(m) if not any(ell) else tensor(u, v)
        certificate = {"kind": "readout", "w": w}
        if any(ell):
            assert dot(v, mv(y, u)) == tau
            certificate.update(u=u, v=v)
    else:
        if v is None:
            z, d = normalized_kernel(a, s), dual_coordinate(c)
        else:
            z, d = dual_coordinate(s), normalized_kernel(p, c)
        certificate = {"kind": "null_witness", "delta_X": [[zi * dj for dj in d] for zi in z]}
    encoded = encode(certificate)
    assert replay_certificate(case, encoded)
    return encode({"id": identifier, "valid": True, "identifiable": identifiable,
                   "Y": y, "tau": tau, "observation_identity": observation_identity,
                   "target_identity": target_identity, "certificate_valid": True,
                   "operator_rank": rank(m), "latent_dimension": len(ell),
                   "equivalent_criteria_agree": True, "certificate": certificate})


def fixed_suite_checks(cases, results):
    """Declared regression purposes; never passed to the independent checker."""
    by_id = {case["id"]: case for case in cases}
    by_result = {result["id"]: result for result in results}
    expected = [True, False, False, False, True, True, True, False, True, True, True, False]
    for index, label in enumerate(expected, 1):
        result = by_result[f"case_{index:02d}"]
        assert result["valid"] and result["identifiable"] is label
    for index in range(13, 20):
        assert not by_result[f"case_{index:02d}"]["valid"]
    assert by_result["case_01"]["operator_rank"] < by_result["case_01"]["latent_dimension"]
    assert rational(by_result["case_08"]["tau"]) == 0
    a, p, x, s, c = parse(by_id["case_09"])
    assert mv(kron(a, p), vec(x)) != vec(mul(mul(a, x), tr(p)))
    a, p, x, s, c = parse(by_id["case_10"])
    app = by_id["case_10"]["application"]
    r, b, h = (matrix(app[key]) for key in ("R", "S", "H"))
    v = solve(tr(a), s)
    assert r != tr(r) and mv(mul(mul(tr(h), tr(b)), tr(r)), v) == s
    assert mv(mul(mul(tr(h), tr(b)), r), v) != s
    a, p, x, s, c = parse(by_id["case_11"])
    m = kron(p, a)
    w = vector(by_result["case_11"]["certificate"]["w"], len(m))
    alternative = [a0 + b0 for a0, b0 in zip(w, null(tr(m))[0])]
    assert w != alternative
    assert replay_certificate(by_id["case_11"], encode({"kind": "readout", "w": alternative}))
    return {"valid_cases": 12, "rejected_inputs": 7, "wrong_kronecker_order_detected": True,
            "wrong_residualizer_transpose_detected": True, "nonunique_readout_replayed": True,
            "zero_realized_target_not_confused_with_identifiability": True,
            "all_required_exact_cases_passed": True}
