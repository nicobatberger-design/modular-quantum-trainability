# -*- coding: utf-8 -*-
"""
Point de confirmation L=160 (N=10) — l'asymptote est-elle bien 1/sqrt(L) ?
==========================================================================
Le run depth_test.py (5 points L=6..80) montre un CROSSOVER :
  - peu profond (L=6-20)  : pente locale ~ -1.2
  - profond     (L=40-80) : pente locale -0.52, g*sqrt(L) plat a ~1.24
PREDICTIONS PRE-ENREGISTREES pour L=160 (avant ce run) :
  - si regime profond 1/sqrt(L) tient : g* = 1.24/sqrt(160) ~ 0.098
  - si la decroissance continue de s'accentuer (~1/L) : g* <= 0.075
  - si plateau naissant : g* >= 0.12
Grille g affinee aux faibles valeurs (g* attendu ~0.1) ; protocole sinon identique
(N=10, m=2, S=80, 5 seeds, score = frac. intrication x frac. variance gradient).
"""
import numpy as np, time, json, os
from scaling_fenetre import apply_1q, RY, RZ, zz_phase_vector, cost_Z0, ent_entropy, gstar_parabolic
from depth_test import run_L

HERE = os.path.dirname(os.path.abspath(__file__))
N, M, S, L = 10, 2, 80, 160
GS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.35, 0.50, 1.00]
SEEDS = [1, 2, 3, 4, 5]

def measure_L(g, seed):
    rng = np.random.default_rng(seed); ph = zz_phase_vector(N, M, g)
    grads, ents = [], []
    for _ in range(S):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0,0] += np.pi/2; thm = th.copy(); thm[0,0] -= np.pi/2
        grads.append((cost_Z0(run_L(thp, fi, N, ph, L), N) - cost_Z0(run_L(thm, fi, N, ph, L), N))/2)
        ents.append(ent_entropy(run_L(th, fi, N, ph, L), N))
    return float(np.var(grads)), float(np.mean(ents))

def main():
    t0 = time.time(); per_seed = []
    for seed in SEEDS:
        vars_, ents_ = [], []
        for g in GS:
            v, e = measure_L(g, seed)
            vars_.append(v); ents_.append(e)
        vars_ = np.array(vars_); ents_ = np.array(ents_)
        v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
        emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
        gst = gstar_parabolic(GS, (ents_/emax)*(vars_/v0))
        per_seed.append(gst)
        print(f"seed {seed} : g* = {gst:.3f}  ({time.time()-t0:.0f}s cumul)", flush=True)
    gm, gsd = float(np.mean(per_seed)), float(np.std(per_seed))
    print(f"\nL=160 : g* = {gm:.3f} +/- {gsd:.3f}   g*sqrt(L) = {gm*np.sqrt(L):.2f}")
    print("Rappel predictions : 1/sqrt(L) -> 0.098 | acceleration ~1/L -> <=0.075 | plateau -> >=0.12", flush=True)
    out = {"N": N, "L": L, "S": S, "gs": GS, "seeds": SEEDS,
           "gstar_mean": gm, "gstar_std": gsd, "gstar_per_seed": per_seed,
           "gstar_sqrtL": gm*float(np.sqrt(L)),
           "pred": {"sqrtL": 0.098, "accel_1surL": 0.075, "plateau": 0.12}}
    with open(os.path.join(HERE, "results_depth_160.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Termine en {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
