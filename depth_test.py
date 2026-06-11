# -*- coding: utf-8 -*-
"""
Test independant du mecanisme percolation-profondeur — g*(L) a N fixe
=====================================================================
Hypothese (formulee 2026-06-11 APRES la chute du plateau, AVANT ce run) :
la fuite inter-bloc s'integre sur la profondeur, transition quand L*g^2 ~ K
  => g*(L) = sqrt(K/L) a N fixe, soit g* proportionnel a 1/sqrt(L).

Cette meme loi, appliquee au protocole L=2N des runs precedents, donne
g*(N) ~ 1/sqrt(2N) — coherent avec la decroissance mesuree (exposant -0.59,
c/sqrt(N) : R2=0.65 a 1 parametre, 0.89 sans l'outlier N=8).

PREDICTIONS PRE-ENREGISTREES (calees sur le point baseline N=10, L=20,
g*=0.322 => K = g*^2 * L = 2.07) :
  L=6  -> g* ~ 0.59
  L=10 -> g* ~ 0.45
  L=20 -> g* ~ 0.32  (baseline, deja mesure)
  L=40 -> g* ~ 0.23
  L=80 -> g* ~ 0.16
Si g**sqrt(L) ~ constant sur 6-80 : mecanisme profondeur CONFIRME independamment.
Si g*(L) ~ constant : mecanisme refute, la decroissance g*(N) a une autre origine.

Protocole identique a scaling_fenetre.py (N=10, m=2, S=80, 5 seeds) sauf L libre.
"""
import numpy as np, time, json, os
from scaling_fenetre import apply_1q, RY, RZ, zz_phase_vector, cost_Z0, ent_entropy, gstar_parabolic

HERE = os.path.dirname(os.path.abspath(__file__))
N, M, S = 10, 2, 80
GS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.57]
LS = [6, 10, 20, 40, 80]
SEEDS = [1, 2, 3, 4, 5]
K_PRED = 0.322**2 * 20  # cale sur baseline N=10, L=20

def run_L(th, fi, N, phase, L):
    s = np.zeros(2**N, dtype=complex); s[0] = 1.0
    for l in range(L):
        for q in range(N): s = apply_1q(s, RY(th[l,q]), q, N)
        for q in range(N): s = apply_1q(s, RZ(fi[l,q]), q, N)
        s = s*phase
    return s

def measure_L(g, L, seed):
    rng = np.random.default_rng(seed); ph = zz_phase_vector(N, M, g)
    grads, ents = [], []
    for _ in range(S):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0,0] += np.pi/2; thm = th.copy(); thm[0,0] -= np.pi/2
        grads.append((cost_Z0(run_L(thp, fi, N, ph, L), N) - cost_Z0(run_L(thm, fi, N, ph, L), N))/2)
        ents.append(ent_entropy(run_L(th, fi, N, ph, L), N))
    return float(np.var(grads)), float(np.mean(ents))

def main():
    t0 = time.time()
    R = {"N": N, "m": M, "S": S, "gs": GS, "Ls": LS, "seeds": SEEDS,
         "K_pred": K_PRED, "pred_gstar": {str(L): float(np.sqrt(K_PRED/L)) for L in LS},
         "data": {}}
    for L in LS:
        tL = time.time(); per_seed = []
        for seed in SEEDS:
            vars_, ents_ = [], []
            for g in GS:
                v, e = measure_L(g, L, seed)
                vars_.append(v); ents_.append(e)
            vars_ = np.array(vars_); ents_ = np.array(ents_)
            v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
            emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
            per_seed.append(gstar_parabolic(GS, (ents_/emax)*(vars_/v0)))
        gm, gsd = float(np.mean(per_seed)), float(np.std(per_seed))
        R["data"][str(L)] = {"gstar_mean": gm, "gstar_std": gsd, "gstar_per_seed": per_seed,
                             "gstar_sqrtL": gm*float(np.sqrt(L))}
        print(f"L={L:3d} : g* = {gm:.3f} +/- {gsd:.3f}   (pred {np.sqrt(K_PRED/L):.3f})   g*sqrt(L)={gm*np.sqrt(L):.2f}   ({time.time()-tL:.0f}s)", flush=True)

    gs_meas = np.array([R["data"][str(L)]["gstar_mean"] for L in LS])
    Ls_a = np.array(LS, float)
    b, loga = np.polyfit(np.log(Ls_a), np.log(gs_meas), 1)
    R["powerlaw_L"] = {"a": float(np.exp(loga)), "b": float(b)}
    const = gs_meas * np.sqrt(Ls_a)
    R["gsqrtL_mean"] = float(const.mean()); R["gsqrtL_std"] = float(const.std())
    verdict = "CONFIRME (g* ~ 1/sqrt(L))" if -0.65 < b < -0.35 else ("plat -> REFUTE" if b > -0.15 else f"pente {b:.2f} hors fenetre")
    print(f"\nFit : g*(L) = {np.exp(loga):.2f} * L^{b:.2f} | g*sqrt(L) = {const.mean():.2f} +/- {const.std():.2f}")
    print(f"VERDICT mecanisme profondeur : {verdict}", flush=True)
    R["verdict"] = verdict
    with open(os.path.join(HERE, "results_depth_test.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(f"Termine en {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
