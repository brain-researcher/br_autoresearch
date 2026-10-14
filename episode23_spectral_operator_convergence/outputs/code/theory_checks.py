"""Seeded numerical refutation attempts; NumPy 1.26 plus standard library.

Run only in the orchestrator's authorized Slurm allocation:
    python theory_checks.py --out /existing/directory/theory_checks.json

Passing is not proof. Identity ratios use numerical tolerances.
The Haar check is a stochastic sample-mean diagnostic.
"""
import argparse
import json
import numpy as np

TOL = 2e-9


def op(a):
    return float(np.linalg.norm(a, 2))


def haar(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs


def symmetric_noise(rng, d, radius):
    g = rng.normal(size=(d, d))
    g = (g + g.T) / 2
    return g * (radius / op(g))


def leading(a, k):
    _, u = np.linalg.eigh(a)
    return u[:, -k:]


def tolerance(*arrays):
    return TOL * max(1.0, *(op(a) for a in arrays))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2305)
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    rng = np.random.default_rng(args.seed)
    results = {}

    def record(name, lhs, bound, instances=1, kind="inequality"):
        lhs, bound = float(lhs), float(bound)
        valid = np.isfinite(lhs) and np.isfinite(bound) and lhs >= 0 and bound >= 0
        ratio = lhs / bound if valid and bound > 0 else float("inf")
        if valid and lhs == bound == 0:
            ratio = 0.0
        passed = bool(valid and ratio <= 1 + 1e-9)
        row = results.setdefault(name, {
            "instances": 0, "max_ratio": 0.0, "passed": True,
            "kind": kind, "evaluations": 0,
        })
        row["instances"] += int(instances)
        row["evaluations"] += 1
        row["max_ratio"] = max(row["max_ratio"], ratio)
        row["passed"] = bool(row["passed"] and passed)

    for _ in range(args.trials):
        # Algebraically ordered eigenvalues with a prescribed reference gap.
        n = int(rng.integers(4, 11))
        q = int(rng.integers(1, n))
        gap = float(rng.uniform(0.2, 1.4))
        upper = np.sort(rng.uniform(gap, gap + 2, q))[::-1]
        lower = -np.sort(rng.uniform(0, 2, n - q))
        upper[-1], lower[0] = gap, 0.0
        values = np.r_[upper, lower]
        v = haar(rng, n)
        a = (v * values) @ v.T
        u0 = v[:, :q]
        p0 = u0 @ u0.T
        eig = np.linalg.eigvalsh(a)
        delta = float(eig[-q] - eig[-q - 1])

        # YWS includes perturbations above the cluster-preservation threshold.
        ey = symmetric_noise(rng, n, delta * rng.uniform(0.01, 1.5))
        uy = leading(a + ey, q)
        numerator = min(np.sqrt(q) * op(ey), np.linalg.norm(ey, "fro"))
        sine_f = np.linalg.norm((np.eye(n) - p0) @ uy, "fro")
        record("YWS_sine", sine_f, 2 * numerator / delta)
        left, _, right = np.linalg.svd(uy.T @ u0)
        alignment = np.linalg.norm(uy @ (left @ right) - u0, "fro")
        record("YWS_alignment", alignment, 2 ** 1.5 * numerator / delta)

        perturbations = [
            symmetric_noise(rng, n, delta * rng.uniform(0.02, 0.44))
            for _ in range(2)
        ]
        projectors = []
        for perturbation in perturbations:
            uh = leading(a + perturbation, q)
            ph = uh @ uh.T
            projectors.append(ph)
            eps = op(perturbation)
            record("mixed_gap", op(ph - p0), eps / (delta - eps))
            pert_eig = np.linalg.eigvalsh(a + perturbation)
            pert_gap = float(pert_eig[-q] - pert_eig[-q - 1])
            record("Weyl_gap", max(0.0, delta - pert_gap), 2 * eps)
        eps = max(op(x) for x in perturbations)
        record("two_learners", op(projectors[0] - projectors[1]),
               2 * eps / (delta - eps))

        # P1's general symmetric version: ensure gamma ||T|| < 1.
        scale = max(1.0, op(a))
        t, lam, gamma = a / scale, values / scale, 0.6
        square = (v * (lam ** 2)) @ v.T
        successor = (v * (1 / (1 - gamma * lam))) @ v.T
        inverse = np.linalg.solve(np.eye(n) - gamma * t, np.eye(n))
        record("P1_functions", max(op(t @ t - square), op(inverse - successor)),
               tolerance(square, inverse), kind="identity")

        # Random feasible competitors attempt to improve P2's claimed optimum.
        rank = int(rng.integers(1, n + 1))
        weights = np.maximum(values[:rank], 0)
        factor = v[:, :rank] * np.sqrt(weights)
        competitor = factor + rng.uniform(0.01, 1) * rng.normal(size=factor.shape)
        loss = np.linalg.norm(a - factor @ factor.T, "fro") ** 2
        other = np.linalg.norm(a - competitor @ competitor.T, "fro") ** 2
        record("P2_PSD_optimum", loss, other)
        exact_loss = float(np.sum(values ** 2) - np.sum(weights ** 2))
        record("P2_loss_formula", abs(loss - exact_loss),
               TOL * max(1.0, abs(exact_loss)), kind="identity")

        # Stochastic blocks: both strict block-local and full fine-space noise.
        blocks, m = int(rng.integers(2, 5)), int(rng.integers(3, 6))
        nb, p, qb = blocks * m, 0.8, 0.2
        z = m * (p + (blocks - 1) * qb)
        beta = (p - qb) / (p + (blocks - 1) * qb)
        c = np.repeat(np.eye(blocks), m, axis=0) / np.sqrt(m)
        pc = c @ c.T
        t0 = (qb * np.ones((nb, nb)) + (p - qb) * m * pc) / z
        w = np.linalg.qr(c, mode="complete")[0][:, blocks:]
        rho = 0.2 * min(qb / z, beta)
        local = np.zeros((nb, nb))
        hm = np.eye(m) - np.ones((m, m)) / m
        for b in range(blocks):
            g = hm @ symmetric_noise(rng, m, 1.0) @ hm
            local[b*m:(b+1)*m, b*m:(b+1)*m] = rho * g / op(g)
        full = w @ symmetric_noise(rng, nb - blocks, rho) @ w.T
        e = np.ones((nb, 1)) / np.sqrt(nb)
        complement = np.linalg.qr(e, mode="complete")[0][:, 1:]
        for perturbation in (local, full):
            tb = t0 + perturbation
            uc = complement @ leading(complement.T @ tb @ complement, blocks - 1)
            residual = max(op(tb @ c - t0 @ c), op(tb @ e - e),
                           op(uc @ uc.T - (pc - e @ e.T)),
                           max(0.0, -float(np.min(tb))))
            record("block_invariance", residual, TOL, kind="identity")

        # PSD perturbation; eta < margin / 4 guarantees the anchor's kNN set.
        nk = int(rng.integers(6, 15))
        x = rng.normal(size=(nk, nk - 1))
        x -= x.mean(axis=0)
        kernel = x @ x.T
        dist = np.diag(kernel)[:, None] + np.diag(kernel)[None, :] - 2 * kernel
        anchor, k = int(rng.integers(nk)), int(rng.integers(1, nk - 1))
        candidates = np.delete(np.arange(nk), anchor)
        order = candidates[np.argsort(dist[anchor, candidates])]
        margin = float(dist[anchor, order[k]] - dist[anchor, order[k - 1]])
        eta = margin / 5
        direction = rng.normal(size=nk)
        direction -= direction.mean()
        dk = eta * np.outer(direction, direction) / (direction @ direction)
        changed = kernel + dk
        newdist = np.diag(changed)[:, None] + np.diag(changed)[None, :] - 2 * changed
        record("distance_bound", np.max(np.abs(newdist - dist)), 2 * eta)
        neworder = candidates[np.argsort(newdist[anchor, candidates])]
        mismatch = float(set(order[:k]) != set(neworder[:k]))
        record("knn_margin", mismatch, TOL, kind="zero_residual")

        # OLS prediction invariance and correspondingly transformed ridge.
        x = rng.normal(size=(12, 4))
        y = rng.normal(size=(12, 3))
        xe = np.vstack((x, rng.normal(size=(7, 4))))
        r = (haar(rng, 4) * np.exp(rng.uniform(-0.8, 0.8, 4))) @ haar(rng, 4).T
        bh = np.linalg.lstsq(x, y, rcond=None)[0]
        ch = np.linalg.lstsq(x @ r, y, rcond=None)[0]
        original, transformed = xe @ bh, xe @ r @ ch
        record("OLS_invariance", op(original - transformed),
               tolerance(original, transformed), kind="identity")
        alpha = 0.7
        bh = np.linalg.solve(x.T @ x + alpha * np.eye(4), x.T @ y)
        ch = np.linalg.solve(r.T @ x.T @ x @ r + alpha * r.T @ r,
                             r.T @ x.T @ y)
        record("ridge_transformed_penalty", op(xe @ bh - xe @ r @ ch),
               tolerance(xe @ bh), kind="identity")

        # P5 identity with unrestricted, generally mixing measurement.
        u, vv = haar(rng, 8)[:, :4], haar(rng, 6)[:, :4]
        amplitudes = np.exp(rng.uniform(-1, 1, 4))
        measurement = rng.normal(size=(6, 5))
        neural = (u * amplitudes) @ vv.T
        observed = neural @ measurement
        middle = vv.T @ measurement @ measurement.T @ vv
        rhs = (u * amplitudes) @ middle @ (u * amplitudes).T
        record("P5_identity", op(observed @ observed.T - rhs),
               tolerance(rhs), kind="identity")

    # Conditional Haar expectation, with a simultaneous Hoeffding threshold.
    batches, alpha_mc = 10, 1e-6
    draws = max(400, 5 * args.trials)
    for _ in range(batches):
        dimension = int(rng.integers(3, 13))
        k = int(rng.integers(1, dimension))
        overlap = np.empty(draws)
        for j in range(draws):
            p_basis = haar(rng, dimension)[:, :k]
            q_basis = haar(rng, dimension)[:, :k]
            overlap[j] = np.linalg.norm(p_basis.T @ q_basis, "fro") ** 2 / k
        threshold = np.sqrt(np.log(2 * batches / alpha_mc) / (2 * draws))
        record("Haar_expected_overlap", abs(overlap.mean() - k / dimension),
               threshold, instances=draws, kind="Monte_Carlo_mean")

    # Explicit numerical counterexamples: residuals to the analytic predictions.
    x, y, r = np.eye(2), np.ones(2), np.diag([2.0, 1.0])
    before = x @ np.linalg.solve(x.T @ x + np.eye(2), x.T @ y)
    xr = x @ r
    after = xr @ np.linalg.solve(xr.T @ xr + np.eye(2), xr.T @ y)
    record("ridge_noninvariance", op((after - before) - np.array([0.3, 0.0])),
           TOL, kind="counterexample_residual")
    blur = np.array([[2.0, 1.0], [1.0, 2.0]]) / 3
    measured = np.diag([2.0, 1.0]) @ blur
    expected = np.array([[20.0, 8.0], [8.0, 5.0]]) / 9
    record("blur_mixing", op(measured @ measured.T - expected),
           TOL, kind="counterexample_residual")

    print(f"{'check':30s} {'instances':>10s} {'max lhs/bound':>15s} {'status':>8s}")
    for name, row in results.items():
        status = "PASS" if row["passed"] else "FAIL"
        print(f"{name:30s} {row['instances']:10d} {row['max_ratio']:15.6g} {status:>8s}")
    passed = all(row["passed"] for row in results.values())
    for row in results.values():
        if not np.isfinite(row["max_ratio"]):
            row["max_ratio"] = "infinity"
    payload = {
        "seed": args.seed, "trials": args.trials, "numpy_version": np.__version__,
        "passed": passed, "checks": results, "haar_nominal_family_error": alpha_mc,
        "interpretation": "Finite numerical refutation attempts; passing is not proof.",
        "identity_tolerance": TOL,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
