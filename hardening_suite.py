# -*- coding: utf-8 -*-
"""
Suite de durcissement 2026-06-11 nuit — « que rien ne soit revocable »
======================================================================
Attaque les angles par lesquels un referee pourrait faire tomber les claims,
au-dela de la generalite (N, m, topologie) testee par overnight_suite.py.

PREDICTIONS PRE-ENREGISTREES (11/06 ~23h45, AVANT les runs) :
  H1 (barren multi-seeds, 5 graines) : pente monolithe <= -0.55 ; pente
     modulaire dans [-0.05, +0.05] ; ratio a N=12 >= x500.
  H2 (raffinement profondeur, N=10) : avec L=28,56,120 ajoutes et 10 graines
     sur L=40/80/160, l'exposant du regime profond (fit L>=40) reste dans
     [-0.60, -0.45] et c dans [1.0, 1.4].
  H3 (sensibilite metrique) : sous score pondere ent^a * var^(1-a) avec
     a=0.3 et a=0.7, c change mais l'exposant profond reste dans [-0.65,-0.40].
  H4 (von Neumann vs Renyi-2, memes etats) : |delta g*| <= 15% a L=40 et 80 ;
     quantifie la part « entropie » vs « graines » dans la retractation du 10/06.
  H5 (position du gradient : couche 0 vs L/2 vs L-1, N=10, L=80) :
     g* varie de moins de +/-25% selon la position -> la fenetre est une
     propriete du circuit, pas du premier parametre.

Resultats incrementaux : hardening_results.json.
"""
import numpy as np, time, json, os, sys
from scaling_fenetre import apply_1q, RY, RZ, zz_phase_vector, cost_Z0, gstar_parabolic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hardening_results.json")
GS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.57]
S = 80
SMOKE = "--smoke" in sys.argv

def run_circ(th, fi, N, phase, L):
    s = np.zeros(2**N, dtype=complex); s[0] = 1.0
    for l in range(L):
        for q in range(N): s = apply_1q(s, RY(th[l, q]), q, N)
        for q in range(N): s = apply_1q(s, RZ(fi[l, q]), q, N)
        s = s*phase
    return s

def both_entropies(s, N):
    """(von Neumann, Renyi-2) sur la demi-coupe, memes etats."""
    a = N//2
    M = s.reshape(2**a, 2**(N-a))
    sv = np.linalg.svd(M, compute_uv=False)
    p = sv**2
    p = p[p > 1e-14]
    vn = float(-np.sum(p*np.log(p)))
    r2 = float(-np.log(np.sum(p**2)))
    return vn, r2

def sweep_curves(N, L, seed, grad_layer=0, samples=S):
    """Courbes var(g), entVN(g), entR2(g) pour une graine."""
    rng = np.random.default_rng(seed)
    vars_, vns, r2s = [], [], []
    for g in GS:
        ph = zz_phase_vector(N, 2, g)
        grads, evn, er2 = [], [], []
        for _ in range(samples):
            th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
            thp = th.copy(); thp[grad_layer, 0] += np.pi/2
            thm = th.copy(); thm[grad_layer, 0] -= np.pi/2
            grads.append((cost_Z0(run_circ(thp, fi, N, ph, L), N)
                          - cost_Z0(run_circ(thm, fi, N, ph, L), N))/2)
            vn, r2 = both_entropies(run_circ(th, fi, N, ph, L), N)
            evn.append(vn); er2.append(r2)
        vars_.append(float(np.var(grads)))
        vns.append(float(np.mean(evn)))
        r2s.append(float(np.mean(er2)))
    return vars_, vns, r2s

def gstar_from_curves(vars_, ents_, alpha=0.5):
    """g* du score pondere ent^alpha * var^(1-alpha) (alpha=0.5 ~ produit baseline)."""
    v = np.array(vars_); e = np.array(ents_)
    v0 = v[0] if v[0] > 1e-15 else 1e-15
    emax = e.max() if e.max() > 1e-15 else 1e-15
    score = (e/emax)**alpha * (v/v0)**(1-alpha)
    return gstar_parabolic(GS, score)

def save(R):
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------- H1 : barren plateau multi-graines ----------
def entangle_phases(N, m, modular):
    ph = np.ones(2**N, dtype=complex)
    for i in range(2**N):
        bits = [(i >> (N-1-q)) & 1 for q in range(N)]
        if not modular:
            k = sum(bits); ph[i] = (-1)**(k*(k-1)//2)
        else:
            p = 1
            for b in range(0, N, m):
                kk = sum(bits[b:b+m]); p *= (-1)**(kk*(kk-1)//2)
            ph[i] = p
    return ph

def grad_var_barren(N, m, modular, seed, samples=100):
    rng = np.random.default_rng(seed); L = 2*N
    ph = entangle_phases(N, m, modular); g = []
    for _ in range(samples):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0, 0] += np.pi/2; thm = th.copy(); thm[0, 0] -= np.pi/2
        g.append((cost_Z0(run_circ(thp, fi, N, ph, L), N)
                  - cost_Z0(run_circ(thm, fi, N, ph, L), N))/2)
    return float(np.var(g))

def main():
    t0 = time.time()
    seeds5 = [1, 2, 3, 4, 5]
    R = {"_doc": "Durcissement nuit 11->12/06. Predictions pre-enregistrees dans le docstring.",
         "gs": GS, "S": S,
         "H1_barren_seeds": {}, "H2_depth_refine": {}, "H2_extra_seeds": {},
         "H3_H4_metric_entropy": {}, "H5_gradpos": {}}

    if SMOKE:
        log("=== SMOKE ===")
        v, vn, r2 = sweep_curves(6, 8, 1, samples=5)[0][:3], None, None
        log(f"sweep ok ({len(v) if v else 0} pts)")
        gv = grad_var_barren(6, 2, True, 1, samples=5)
        log(f"barren ok ({gv:.3e})")
        g1 = gstar_from_curves([1]*12, list(np.linspace(0, 1, 12)), 0.5)
        log(f"gstar_from_curves ok ({g1:.2f})")
        log("SMOKE OK")
        return

    # --- H1 : barren multi-graines (rapide) ---
    log("=== H1 : barren plateau, 5 graines, N=4..12 ===")
    Ns = [4, 6, 8, 10, 12]
    for N in Ns:
        mono = [grad_var_barren(N, 2, False, s) for s in seeds5]
        modu = [grad_var_barren(N, 2, True, s) for s in seeds5]
        R["H1_barren_seeds"][f"N_{N}"] = {"monolithe": mono, "modulaire": modu}
        log(f"H1 N={N:2d} : mono {np.mean(mono):.2e}±{np.std(mono):.1e} | "
            f"modu {np.mean(modu):.2e}±{np.std(modu):.1e} | ratio {np.mean(modu)/np.mean(mono):.0f}x")
        save(R)
    slopes_m = [np.polyfit(Ns, np.log([np.mean(R['H1_barren_seeds'][f'N_{N}']['monolithe']) for N in Ns]), 1)[0]]
    log(f"H1 pente monolithe (moyenne graines) : {slopes_m[0]:+.3f}")

    # --- H3/H4 d'abord sur L=40 et 80 (reutilise sweep_curves, fournit aussi H2 partiel) ---
    log("=== H3/H4 : metrique + entropie, N=10, L=40/80 (5 graines, courbes completes) ===")
    for L in [40, 80]:
        per_seed = {"vars": [], "vns": [], "r2s": []}
        for s in seeds5:
            v, vn, r2 = sweep_curves(10, L, s)
            per_seed["vars"].append(v); per_seed["vns"].append(vn); per_seed["r2s"].append(r2)
        entry = {"curves": per_seed}
        for alpha, key in [(0.5, "a05"), (0.3, "a03"), (0.7, "a07")]:
            gst = [gstar_from_curves(per_seed["vars"][i], per_seed["vns"][i], alpha) for i in range(5)]
            entry[f"gstar_vn_{key}"] = {"mean": float(np.mean(gst)), "std": float(np.std(gst)), "per_seed": gst}
        gst_r2 = [gstar_from_curves(per_seed["vars"][i], per_seed["r2s"][i], 0.5) for i in range(5)]
        entry["gstar_r2_a05"] = {"mean": float(np.mean(gst_r2)), "std": float(np.std(gst_r2)), "per_seed": gst_r2}
        R["H3_H4_metric_entropy"][f"L_{L}"] = entry
        log(f"H3/H4 L={L} : vN a=.5 {entry['gstar_vn_a05']['mean']:.3f} | a=.3 {entry['gstar_vn_a03']['mean']:.3f} | "
            f"a=.7 {entry['gstar_vn_a07']['mean']:.3f} | R2 {entry['gstar_r2_a05']['mean']:.3f}")
        save(R)

    # --- H2 : points L intermediaires (5 graines) ---
    log("=== H2 : L=28, 56, 120 (N=10, 5 graines) ===")
    for L in [28, 56, 120]:
        gst = []
        for s in seeds5:
            v, vn, _ = sweep_curves(10, L, s)
            gst.append(gstar_from_curves(v, vn, 0.5))
        R["H2_depth_refine"][f"L_{L}"] = {"mean": float(np.mean(gst)), "std": float(np.std(gst)), "per_seed": gst}
        log(f"H2 L={L:3d} : g* = {np.mean(gst):.3f} +/- {np.std(gst):.3f}  (g*sqrtL={np.mean(gst)*np.sqrt(L):.2f})")
        save(R)

    # --- H2bis : graines 6-10 sur les ancres L=40/80/160 ---
    log("=== H2bis : graines 6-10 sur L=40/80/160 ===")
    for L in [40, 80, 160]:
        gst = []
        for s in [6, 7, 8, 9, 10]:
            v, vn, _ = sweep_curves(10, L, s)
            gst.append(gstar_from_curves(v, vn, 0.5))
        R["H2_extra_seeds"][f"L_{L}"] = {"per_seed_6_10": gst, "mean": float(np.mean(gst)), "std": float(np.std(gst))}
        log(f"H2bis L={L:3d} : graines 6-10 -> g* = {np.mean(gst):.3f} +/- {np.std(gst):.3f}")
        save(R)

    # --- H5 : position du gradient (N=10, L=80) ---
    log("=== H5 : gradient couche 0 / L/2 / L-1 (N=10, L=80, 5 graines) ===")
    for layer, key in [(0, "layer0"), (40, "layerMid"), (79, "layerLast")]:
        gst = []
        for s in seeds5:
            v, vn, _ = sweep_curves(10, 80, s, grad_layer=layer)
            gst.append(gstar_from_curves(v, vn, 0.5))
        R["H5_gradpos"][key] = {"mean": float(np.mean(gst)), "std": float(np.std(gst)), "per_seed": gst}
        log(f"H5 {key:9s} : g* = {np.mean(gst):.3f} +/- {np.std(gst):.3f}")
        save(R)

    log(f"Durcissement termine en {(time.time()-t0)/3600:.1f} h. -> {OUT}")

if __name__ == "__main__":
    main()
