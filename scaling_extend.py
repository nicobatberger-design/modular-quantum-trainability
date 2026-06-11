"""
Extension du scaling g*(N) vers N=14 et N=16 — trancher plateau vs decroissance
================================================================================
Contexte (reprise 2026-06-11) : a N<=12, g*(N) etait stable [0.32-0.48] mais le
fit loi-de-puissance (b=-0.94) etait domine par le point N=4 (g*~0.99), artefact
de petite taille. Objectif : ajouter N=14 (5 seeds) et N=16 (3 seeds, indicatif)
pour voir si g* continue de chuter (loi de puissance) ou se stabilise (plateau).

Reutilise EXACTEMENT le protocole de scaling_fenetre.py (memes fonctions,
S=80, gs identiques) pour comparabilite stricte. Fusionne avec le json existant.
Refait le fit power-law en EXCLUANT N=4 (note explicite), et trace les deux.
"""
import numpy as np, time, json, os
from scaling_fenetre import measure, gstar_parabolic

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results_scaling_fenetre.json")        # N=4..12 (5 seeds)
OUT = os.path.join(HERE, "results_scaling_extended.json")
FIG = os.path.join(HERE, "resultats_scaling_extended.png")

GS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.57]
NEW = [(14, [1, 2, 3, 4, 5])]   # N=16 reserve au GPU Kaggle (265s/point en CPU = 2h40)
S = 80
M = 2

def compute_point(N, seeds):
    t = time.time()
    per_seed = []
    var_acc = np.zeros(len(GS)); ent_acc = np.zeros(len(GS))
    for seed in seeds:
        vars_, ents_ = [], []
        for g in GS:
            v, e = measure(N, M, g, S=S, seed=seed)
            vars_.append(v); ents_.append(e)
        vars_ = np.array(vars_); ents_ = np.array(ents_)
        var_acc += vars_; ent_acc += ents_
        v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
        emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
        score = (ents_ / emax) * (vars_ / v0)
        per_seed.append(gstar_parabolic(GS, score))
    gm, gsd = float(np.mean(per_seed)), float(np.std(per_seed))
    print(f"N={N:2d} : g* = {gm:.3f} +/- {gsd:.3f}  ({len(seeds)} seeds, {time.time()-t:.0f}s)", flush=True)
    return {
        "gstar_per_seed": per_seed, "gstar_mean": gm, "gstar_std": gsd,
        "var_mean_curve": (var_acc/len(seeds)).tolist(),
        "ent_mean_curve": (ent_acc/len(seeds)).tolist(),
        "n_seeds": len(seeds),
    }

def main():
    t0 = time.time()
    with open(SRC) as f:
        base = json.load(f)

    data = dict(base["data"])
    for N, seeds in NEW:
        data[f"N_{N}"] = compute_point(N, seeds)

    Ns = sorted(int(k.split("_")[1]) for k in data)
    gstars = [data[f"N_{N}"]["gstar_mean"] for N in Ns]
    gstds  = [data[f"N_{N}"]["gstar_std"]  for N in Ns]

    # Fit power-law SUR N>=6 (exclut le point N=4 aberrant, petite taille)
    Ns_fit = [N for N in Ns if N >= 6]
    g_fit  = [data[f"N_{N}"]["gstar_mean"] for N in Ns_fit]
    b, loga = np.polyfit(np.log(Ns_fit), np.log(g_fit), 1)
    a = float(np.exp(loga))
    pred = a * np.array(Ns_fit, float)**b
    ss_res = float(np.sum((np.array(g_fit)-pred)**2))
    ss_tot = float(np.sum((np.array(g_fit)-np.mean(g_fit))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    # Modele plateau (constante) sur N>=6
    plateau = float(np.mean(g_fit))
    ss_res_flat = float(np.sum((np.array(g_fit)-plateau)**2))
    r2_flat = 1 - ss_res_flat/ss_tot if ss_tot > 0 else 0.0
    print(f"\nFit N>=6 : power-law g*={a:.3f}*N^{b:.3f} (R2={r2:.2f}) | plateau g*={plateau:.3f} (R2_flat={r2_flat:.2f})", flush=True)
    verdict = "plateau" if abs(b) < 0.15 or r2_flat >= r2 else "decroissance"
    print(f"VERDICT (N>=6) : {verdict}  (|b|={abs(b):.2f})", flush=True)

    out = {
        "m": M, "Ns": Ns, "gs": GS, "S": S,
        "data": data,
        "gstars_mean": gstars, "gstars_std": gstds,
        "fit_Nge6": {"a": a, "b": float(b), "r2": r2,
                     "plateau_mean": plateau, "r2_plateau": r2_flat,
                     "verdict": verdict},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON ecrit : {OUT}", flush=True)

    # Figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.errorbar(Ns, gstars, yerr=gstds, marker="s", capsize=4,
                    color="#2D5BFF", label="g*(N) mesure")
        xs = np.linspace(min(Ns_fit), max(Ns_fit), 100)
        ax.plot(xs, a*xs**b, "--", color="#E8764D",
                label=f"power-law N>=6 : {a:.2f} N^{b:.2f} (R2={r2:.2f})")
        ax.axhline(plateau, ls=":", color="#10B981",
                   label=f"plateau N>=6 : {plateau:.2f} (R2={r2_flat:.2f})")
        ax.axvspan(13.5, 16.5, color="grey", alpha=0.08)
        ax.text(14.6, ax.get_ylim()[1]*0.95, "nouveau\n(N=14,16)", fontsize=8,
                ha="center", va="top", color="grey")
        ax.set_xlabel("N (qubits)"); ax.set_ylabel("g* (fenetre optimale)")
        ax.set_title("Loi d'echelle de la fenetre de couplage optimale g*(N)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG, dpi=130)
        print(f"Figure ecrite : {FIG}", flush=True)
    except Exception as ex:
        print(f"(figure non generee : {ex})", flush=True)

    print(f"\nTermine en {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
