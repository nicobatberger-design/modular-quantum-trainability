"""Can a small truncated circuit reproduce the modular circuit at the weak-coupling window?

If <Z0> and its gradient only depend on the first blocks when g = 1.2/sqrt(L),
a truncated circuit of n qubits reproduces the full circuit of N qubits, and the
regime is not out of reach of classical simulation for this observable.

Protocol of the depth-law scripts: blocks m=2, intra-block ZZ angle pi/2, inter-block
ZZ angle g in a chain, L layers of RY+RZ, parameter-shift gradient of theta[0,0].
Full and truncated circuits use the same random parameters (qubits 0 to n-1).

Prediction written before the first run (31 Oct 2026):
  - at g*: n=6 reproduces the gradient with NRMSE <= 0.30 at every depth, and the
    result does not depend on N (12 or 14);
  - control, monolith g=pi/2: NRMSE >= 0.90 for every n < N.
Outcome: the first held (2-3 %). The second failed at n=10 (0.61), and held for n <= 8.

NRMSE = rms(full - truncated) / rms(full). A value of 1 is no better than answering 0.

Usage: python truncation_test.py results_truncation.json
"""
import json
import os
import sys
import time

import numpy as np

ALPHA = np.pi / 2


def phases(n, g):
    idx = np.arange(2**n)
    z = np.array([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)], dtype=float)
    ang = np.zeros(2**n)
    for b in range(0, n, 2):
        if b + 1 < n:
            ang += ALPHA * z[b] * z[b + 1]
    for b in range(0, n - 2, 2):
        ang += g * z[b + 1] * z[b + 2]
    return np.exp(-0.5j * ang)


def circuits(th, fi, g):
    """th, fi of shape (K, L, n). Returns the final states, shape (K, 2**n)."""
    K, L, n = th.shape
    ph = phases(n, g)
    s = np.zeros((K, 2**n), dtype=complex)
    s[:, 0] = 1.0
    c, d = np.cos(th / 2), np.sin(th / 2)
    em, ep = np.exp(-0.5j * fi), np.exp(0.5j * fi)
    for l in range(L):
        for q in range(n):
            t = s.reshape(K, 2**q, 2, 2 ** (n - q - 1))
            a, b = t[:, :, 0, :].copy(), t[:, :, 1, :].copy()
            cc = c[:, l, q].reshape(K, 1, 1)
            dd = d[:, l, q].reshape(K, 1, 1)
            mm = em[:, l, q].reshape(K, 1, 1)
            pp = ep[:, l, q].reshape(K, 1, 1)
            t[:, :, 0, :] = mm * (cc * a - dd * b)
            t[:, :, 1, :] = pp * (dd * a + cc * b)
        s = s * ph
    return s


def z_on(s, q=0):
    """<Z_q> for each state of the batch."""
    n = int(np.log2(s.shape[1]))
    z = 1 - 2 * ((np.arange(2**n) >> (n - 1 - q)) & 1)
    return (np.abs(s) ** 2 * z).sum(axis=1)


def cost_and_gradient(th, fi, g, q=0):
    """<Z_q> and its parameter-shift gradient with respect to theta[0, q]."""
    K = th.shape[0]
    tp, tm = th.copy(), th.copy()
    tp[:, 0, q] += np.pi / 2
    tm[:, 0, q] -= np.pi / 2
    all_t = np.concatenate([th, tp, tm])
    all_f = np.concatenate([fi, fi, fi])
    v = z_on(circuits(all_t, all_f, g), q)
    return v[:K], (v[K:2 * K] - v[2 * K:]) / 2


def nrmse(ref, approx, rng, n_boot=1000):
    val = np.sqrt(np.mean((ref - approx) ** 2) / np.mean(ref**2))
    b = []
    K = len(ref)
    for _ in range(n_boot):
        i = rng.integers(0, K, K)
        b.append(np.sqrt(np.mean((ref[i] - approx[i]) ** 2) / np.mean(ref[i] ** 2)))
    return float(val), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def condition(N, L, g, K, sizes, seed=11):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, (K, L, N))
    fi = rng.uniform(0, 2 * np.pi, (K, L, N))
    t0 = time.time()
    c_ref, g_ref = cost_and_gradient(th, fi, g)
    t_ref = time.time() - t0
    out = {
        "N": N, "L": L, "g": g, "K": K, "seconds_full": round(t_ref, 1),
        "rms_gradient": float(np.sqrt(np.mean(g_ref**2))),
        "var_gradient": float(np.var(g_ref)), "truncated": {},
    }
    for n in sizes:
        t0 = time.time()
        c_t, g_t = cost_and_gradient(th[:, :, :n], fi[:, :, :n], g)
        out["truncated"][n] = {
            "nrmse_cost": nrmse(c_ref, c_t, rng),
            "nrmse_gradient": nrmse(g_ref, g_t, rng),
            "variance_ratio": float(np.var(g_t) / np.var(g_ref)),
            "seconds": round(time.time() - t0, 2),
        }
    print(json.dumps(out), flush=True)
    return out


def check_against_original_code():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scaling_fenetre import cost_Z0, run, zz_phase_vector

    rng = np.random.default_rng(3)
    n, L, g = 6, 12, 0.3
    th, fi = rng.uniform(0, 6.28, (2, L, n))
    ref = cost_Z0(run(th, fi, n, zz_phase_vector(n, 2, g)), n)
    mine = z_on(circuits(th[None], fi[None], g))[0]
    print(f"original code: {ref:.10f} / this script: {mine:.10f}", flush=True)
    assert abs(ref - mine) < 1e-9


def main(path):
    check_against_original_code()
    res = []
    for L in (40, 80, 160):
        res.append(condition(12, L, 1.2 / np.sqrt(L), 200, (2, 4, 6, 8, 10)))
    res.append(condition(14, 80, 1.2 / np.sqrt(80), 100, (2, 4, 6, 8, 10, 12)))
    g_ref = 1.2 / np.sqrt(80)
    for g in (2 * g_ref, 4 * g_ref):
        res.append(condition(12, 80, g, 200, (2, 4, 6, 8, 10)))
    res.append(condition(12, 40, np.pi / 2, 200, (2, 4, 6, 8, 10)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
