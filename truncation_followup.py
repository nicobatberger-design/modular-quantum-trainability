"""Follow-up tests of truncation_test.py. Predictions: PREDICTIONS_2026-10-31.md.

A  bulk observable: Z on a middle qubit, with windows centred on it
B  scaling variable: do gradient variance and entropy depend on x = L g^2 only?
C  trained parameters: train on 6 qubits, evaluate on the full 12-qubit circuit
D  third system size: N=16

Usage: python truncation_followup.py A results_followup_A.json
"""
import json
import sys
import time

import numpy as np

from truncation_test import circuits, cost_and_gradient, nrmse, z_on


def test_a():
    N, q, K = 14, 6, 100
    windows = {2: (6, 8), 4: (4, 8), 6: (4, 10), 10: (2, 12)}
    res = []
    for L in (40, 160):
        g = 1.2 / np.sqrt(L)
        rng = np.random.default_rng(21)
        th = rng.uniform(0, 2 * np.pi, (K, L, N))
        fi = rng.uniform(0, 2 * np.pi, (K, L, N))
        c_ref, g_ref = cost_and_gradient(th, fi, g, q)
        out = {"N": N, "q": q, "L": L, "g": g, "K": K,
               "var_gradient": float(np.var(g_ref)), "window": {}}
        for n, (a, b) in windows.items():
            c_t, g_t = cost_and_gradient(th[:, :, a:b], fi[:, :, a:b], g, q - a)
            out["window"][n] = {
                "qubits": [a, b - 1],
                "nrmse_cost": nrmse(c_ref, c_t, rng),
                "nrmse_gradient": nrmse(g_ref, g_t, rng),
                "variance_ratio": float(np.var(g_t) / np.var(g_ref)),
            }
        print(json.dumps(out), flush=True)
        res.append(out)
    return res


def entropy(s, n, cut):
    m = s.reshape(s.shape[0], 2**cut, 2 ** (n - cut))
    sv = np.linalg.svd(m, compute_uv=False)
    p = sv**2
    return (-np.sum(p * np.log(np.where(p > 1e-12, p, 1.0)), axis=1)).mean()


def test_b():
    N, K = 10, 300
    xs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.44, 2.0, 3.0, 4.0, 6.0, 9.0]
    res = []
    for L in (40, 80, 160):
        for x in xs + ["monolith"]:
            g = np.pi / 2 if x == "monolith" else float(np.sqrt(x / L))
            rng = np.random.default_rng(31 + L)
            th = rng.uniform(0, 2 * np.pi, (K, L, N))
            fi = rng.uniform(0, 2 * np.pi, (K, L, N))
            t0 = time.time()
            _, grad = cost_and_gradient(th, fi, g)
            s = circuits(th, fi, g)
            out = {"N": N, "L": L, "x": x, "g": g, "K": K,
                   "var_gradient": float(np.var(grad)),
                   "entropy_midblock_cut": float(entropy(s, N, 5)),
                   "entropy_boundary_cut": float(entropy(s, N, 4)),
                   "seconds": round(time.time() - t0, 1)}
            print(json.dumps(out), flush=True)
            res.append(out)
    return res


def train_truncated(g, L, n, steps, lr, seed):
    """Adam on <Z0> of the n-qubit circuit, exact parameter-shift gradient."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 2 * np.pi, (2, L, n))
    first = p.copy()
    m1, m2 = np.zeros_like(p), np.zeros_like(p)
    n_par = p.size
    for t in range(1, steps + 1):
        batch = np.repeat(p[None], 2 * n_par, axis=0)
        idx = np.arange(n_par)
        coord = np.unravel_index(idx, p.shape)
        batch[2 * idx, coord[0], coord[1], coord[2]] += np.pi / 2
        batch[2 * idx + 1, coord[0], coord[1], coord[2]] -= np.pi / 2
        v = z_on(circuits(batch[:, 0], batch[:, 1], g))
        grad = ((v[::2] - v[1::2]) / 2).reshape(p.shape)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad**2
        p = p - lr * (m1 / (1 - 0.9**t)) / (np.sqrt(m2 / (1 - 0.999**t)) + 1e-8)
    return first, p


def compare(p, g, L, n, N, n_fill, rng):
    th_n, fi_n = p
    small = float(z_on(circuits(th_n[None], fi_n[None], g))[0])
    th = np.concatenate([np.repeat(th_n[None], n_fill, axis=0),
                         rng.uniform(0, 2 * np.pi, (n_fill, L, N - n))], axis=2)
    fi = np.concatenate([np.repeat(fi_n[None], n_fill, axis=0),
                         rng.uniform(0, 2 * np.pi, (n_fill, L, N - n))], axis=2)
    full = z_on(circuits(th, fi, g))
    return small, full


def test_c():
    N, n, L, restarts, n_fill = 12, 6, 40, 8, 20
    g = 1.2 / np.sqrt(L)
    res = []
    rng = np.random.default_rng(41)
    for seed in range(restarts):
        t0 = time.time()
        first, last = train_truncated(g, L, n, steps=120, lr=0.1, seed=100 + seed)
        s0, f0 = compare(first, g, L, n, N, n_fill, rng)
        s1, f1 = compare(last, g, L, n, N, n_fill, rng)
        out = {"seed": seed, "N": N, "n": n, "L": L, "g": g, "fillings": n_fill,
               "before": {"truncated": s0, "full_mean": float(f0.mean()),
                          "mean_abs_gap": float(np.mean(np.abs(f0 - s0)))},
               "after": {"truncated": s1, "full_mean": float(f1.mean()),
                         "full_std": float(f1.std()),
                         "mean_abs_gap": float(np.mean(np.abs(f1 - s1)))},
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(out), flush=True)
        res.append(out)
    return res


def test_d():
    res = []
    N, K = 16, 30
    for L in (40, 80):
        g = 1.2 / np.sqrt(L)
        rng = np.random.default_rng(51)
        th = rng.uniform(0, 2 * np.pi, (K, L, N))
        fi = rng.uniform(0, 2 * np.pi, (K, L, N))
        c_ref, g_ref = cost_and_gradient(th, fi, g)
        out = {"N": N, "L": L, "g": g, "K": K, "truncated": {}}
        for n in (4, 6, 8):
            c_t, g_t = cost_and_gradient(th[:, :, :n], fi[:, :, :n], g)
            out["truncated"][n] = {"nrmse_cost": nrmse(c_ref, c_t, rng),
                                   "nrmse_gradient": nrmse(g_ref, g_t, rng)}
        print(json.dumps(out), flush=True)
        res.append(out)
    return res


if __name__ == "__main__":
    tests = {"A": test_a, "B": test_b, "C": test_c, "D": test_d}
    result = tests[sys.argv[1]]()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("DONE", flush=True)
