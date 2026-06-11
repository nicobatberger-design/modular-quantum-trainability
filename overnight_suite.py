# -*- coding: utf-8 -*-
"""
Suite nocturne 2026-06-11 -> 12 — robustesse de la loi de profondeur (these v4)
================================================================================
Question ouverte n°1 de la v4 : la constante c ~ 1.2 de g*(L) ~ c/sqrt(L)
depend-elle de N, de la taille de bloc m, de la topologie ?

PREDICTIONS PRE-ENREGISTREES (avant les runs, 11/06 ~23h30) :
  A. c(N) a L=20 fixe : si la fenetre est PILOTEE PAR LA PROFONDEUR seule,
     g* doit etre ~constant en N : g* ~ 1.24/sqrt(20) ~ 0.277 pour N=6..14.
     (L'ancienne variation apparente de g*(N) venait du protocole L=2N.)
  B. c(m) a N=12, L=24 : heuristique naive percolation -> K depend du nombre
     de liens de frontiere par profondeur ; si K ~ 1/m, alors g* croit avec m.
     Prediction faible (ordre) : g*(m=4) >= g*(m=3) >= g*(m=2). Valeurs libres.
  C. Topologie ring vs chain (N=12, L=24, m=2) : le ring ajoute 1 lien (fuite
     accrue ~ (B+1)/B) -> g*_ring legerement INFERIEUR a g*_chain (~ -10%).
  D. g*(L) a N=12 (L=12,24,48,96) : meme loi qu'a N=10 en regime profond,
     g* ~ 1.24/sqrt(L) -> 0.36 / 0.25 / 0.18 / 0.127. Si confirme a un 2e N,
     la loi de profondeur est generique.
  E. (bonus si la nuit suffit) N=16 a L=20, 3 seeds : prolonge le test A.

Tout est ecrit incrementalement dans overnight_suite_results.json.
Protocole : S=80 echantillons, 5 seeds (3 pour E), score = frac. intrication
x frac. variance de gradient, g* = argmax parabolique. Grille g standard.
"""
import numpy as np, time, json, os, sys
from scaling_fenetre import apply_1q, RY, RZ, cost_Z0, ent_entropy, gstar_parabolic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "overnight_suite_results.json")
GS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.57]
S = 80
SMOKE = "--smoke" in sys.argv

def zz_phase_general(N, m, g, alpha=np.pi/2, topology="chain"):
    """Phases ZZ : intra-bloc all-to-all (alpha), inter-bloc (g) chain ou ring."""
    idx = np.arange(2**N)
    z = np.zeros((N, 2**N))
    for q in range(N):
        z[q] = 1 - 2*((idx >> (N-1-q)) & 1)
    ang = np.zeros(2**N)
    for b in range(0, N, m):
        blk = list(range(b, min(b+m, N)))
        for a in range(len(blk)):
            for c in range(a+1, len(blk)):
                ang += alpha*z[blk[a]]*z[blk[c]]
    for b in range(0, N-m, m):
        ang += g*z[b+m-1]*z[b+m]
    if topology == "ring" and N > 2*m:
        ang += g*z[N-1]*z[0]
    return np.exp(-0.5j*ang)

def run_circ(th, fi, N, phase, L):
    s = np.zeros(2**N, dtype=complex); s[0] = 1.0
    for l in range(L):
        for q in range(N): s = apply_1q(s, RY(th[l, q]), q, N)
        for q in range(N): s = apply_1q(s, RZ(fi[l, q]), q, N)
        s = s*phase
    return s

def measure(N, m, g, L, seed, topology="chain", samples=S):
    rng = np.random.default_rng(seed)
    ph = zz_phase_general(N, m, g, topology=topology)
    grads, ents = [], []
    for _ in range(samples):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0, 0] += np.pi/2; thm = th.copy(); thm[0, 0] -= np.pi/2
        grads.append((cost_Z0(run_circ(thp, fi, N, ph, L), N) - cost_Z0(run_circ(thm, fi, N, ph, L), N))/2)
        ents.append(ent_entropy(run_circ(th, fi, N, ph, L), N))
    return float(np.var(grads)), float(np.mean(ents))

def gstar_config(N, m, L, seeds, topology="chain", label=""):
    t0 = time.time(); per_seed = []
    for seed in seeds:
        vars_, ents_ = [], []
        for g in GS:
            v, e = measure(N, m, g, L, seed, topology)
            vars_.append(v); ents_.append(e)
        vars_ = np.array(vars_); ents_ = np.array(ents_)
        v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
        emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
        per_seed.append(gstar_parabolic(GS, (ents_/emax)*(vars_/v0)))
    gm, gsd = float(np.mean(per_seed)), float(np.std(per_seed))
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {label:28s} N={N:2d} m={m} L={L:3d} {topology:5s} : "
          f"g* = {gm:.3f} +/- {gsd:.3f}  ({dt:.0f}s)", flush=True)
    return {"N": N, "m": m, "L": L, "topology": topology, "seeds": list(seeds),
            "gstar_mean": gm, "gstar_std": gsd, "gstar_per_seed": per_seed,
            "gstar_sqrtL": gm*float(np.sqrt(L)), "secs": dt}

def save(R):
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)

def main():
    t0 = time.time()
    R = {"_doc": "Suite nocturne 11->12/06 : robustesse loi de profondeur (these v4). "
                 "Predictions pre-enregistrees dans le docstring du script.",
         "gs": GS, "S": S,
         "pred": {"A_cN_L20": 0.277, "C_ring_vs_chain": "ring ~ -10%",
                  "D_gstar_L_N12": {"12": 0.36, "24": 0.25, "48": 0.18, "96": 0.127}},
         "A_cN_fixedL20": [], "B_m_N12_L24": [], "C_topo_N12_L24": [],
         "D_depth_N12": [], "E_N16_L20": []}

    seeds5 = [1, 2, 3, 4, 5]
    if SMOKE:
        print("=== SMOKE TEST (params reduits) ===", flush=True)
        gstar_config(6, 2, 8, [1], "chain", "smoke-chain")
        gstar_config(6, 3, 8, [1], "ring", "smoke-ring-m3")
        print("SMOKE OK", flush=True)
        return

    # --- A. c(N) a profondeur fixe L=20 (le test central) ---
    print("=== A. c(N) a L=20 fixe — prediction : g* ~ 0.277 constant ===", flush=True)
    for N in [6, 8, 10, 12, 14]:
        R["A_cN_fixedL20"].append(gstar_config(N, 2, 20, seeds5, "chain", "A:c(N)@L20"))
        save(R)

    # --- B. dependance en m (N=12, L=24) ---
    print("=== B. c(m) a N=12, L=24 — prediction : g* croissant avec m ===", flush=True)
    for m in [2, 3, 4]:
        R["B_m_N12_L24"].append(gstar_config(12, m, 24, seeds5, "chain", "B:c(m)@N12L24"))
        save(R)

    # --- C. topologie ring (N=12, L=24, m=2) ; chain = run m=2 de B ---
    print("=== C. ring vs chain — prediction : ring ~ -10% ===", flush=True)
    R["C_topo_N12_L24"].append(gstar_config(12, 2, 24, seeds5, "ring", "C:ring@N12L24"))
    save(R)

    # --- D. loi de profondeur a N=12 (L=24 deja couvert par B[m=2]) ---
    print("=== D. g*(L) a N=12 — prediction : 1.24/sqrt(L) ===", flush=True)
    for L in [12, 48, 96]:
        R["D_depth_N12"].append(gstar_config(12, 2, L, seeds5, "chain", "D:depth@N12"))
        save(R)

    # --- E. bonus : N=16 a L=20 (3 seeds) ---
    print("=== E. N=16 a L=20 (bonus, 3 seeds) ===", flush=True)
    R["E_N16_L20"].append(gstar_config(16, 2, 20, [1, 2, 3], "chain", "E:N16@L20"))
    save(R)

    print(f"\nSuite terminee en {(time.time()-t0)/3600:.1f} h. Resultats : {OUT}", flush=True)

if __name__ == "__main__":
    main()
