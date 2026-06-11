"""
Scaling de la fenetre de couplage optimale — g*(N)
===================================================
Question (approfondissement these, nuit 2026-06-10) :
La fenetre de couplage faible g~0.2-0.3 observee a N=10 est-elle FIXE,
ou depend-elle de la taille N du systeme ?

Si g*(N) suit une loi d'echelle (ex: g* ~ a*N^b), la these passe d'une
observation a une PREDICTION QUANTITATIVE extrapolable au regime
non-simulable — exactement ce qu'un labo peut tester sur hardware.

Critere objectif du sweet spot :
  score(g) = [Ent(g)/Ent_max(N)] * [Var(g)/Var(g=0)]
  = fraction d'intrication atteinte x fraction d'entrainabilite conservee
  g* = argmax du score (interpolation parabolique autour du max discret).

Protocole identique a sweet_spot.py (comparabilite) : blocs m=2,
intra fort (alpha=pi/2), inter = g en chaine, L=2N couches, S=80
echantillons par seed, 5 seeds par point -> barres d'erreur.
"""
import numpy as np, time, json, os

HERE = os.path.dirname(os.path.abspath(__file__))

def apply_1q(state, G, q, N):
    st = state.reshape([2]*N); st = np.tensordot(G, st, axes=([1],[q]))
    return np.moveaxis(st, 0, q).reshape(-1)

def RY(t):
    c, s = np.cos(t/2), np.sin(t/2); return np.array([[c,-s],[s,c]], dtype=complex)

def RZ(t):
    return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]], dtype=complex)

def zz_phase_vector(N, m, g, alpha=np.pi/2):
    idx = np.arange(2**N)
    z = np.zeros((N, 2**N))
    for q in range(N): z[q] = 1 - 2*((idx >> (N-1-q)) & 1)
    ang = np.zeros(2**N)
    for b in range(0, N, m):
        blk = list(range(b, min(b+m, N)))
        for a in range(len(blk)):
            for c in range(a+1, len(blk)): ang += alpha*z[blk[a]]*z[blk[c]]
    for b in range(0, N-m, m):
        ang += g*z[b+m-1]*z[b+m]
    return np.exp(-0.5j*ang)

def run(th, fi, N, phase):
    s = np.zeros(2**N, dtype=complex); s[0] = 1.0; L = th.shape[0]
    for l in range(L):
        for q in range(N): s = apply_1q(s, RY(th[l,q]), q, N)
        for q in range(N): s = apply_1q(s, RZ(fi[l,q]), q, N)
        s = s*phase
    return s

def cost_Z0(s, N):
    st = s.reshape([2]*N)
    return float(np.sum(np.abs(st[0])**2) - np.sum(np.abs(st[1])**2))

def ent_entropy(s, N):
    a = N//2; M = s.reshape(2**a, 2**(N-a))
    sv = np.linalg.svd(M, compute_uv=False); p = sv**2; p = p[p > 1e-12]
    return float(-np.sum(p*np.log(p)))

def measure(N, m, g, S=80, seed=1):
    rng = np.random.default_rng(seed); L = 2*N; ph = zz_phase_vector(N, m, g)
    grads = []; ents = []
    for _ in range(S):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0,0] += np.pi/2; thm = th.copy(); thm[0,0] -= np.pi/2
        grads.append((cost_Z0(run(thp, fi, N, ph), N) - cost_Z0(run(thm, fi, N, ph), N))/2)
        ents.append(ent_entropy(run(th, fi, N, ph), N))
    return float(np.var(grads)), float(np.mean(ents))

def gstar_parabolic(gs, scores):
    """Argmax avec raffinement parabolique autour du max discret."""
    i = int(np.argmax(scores))
    if i == 0 or i == len(gs)-1: return float(gs[i])
    x0, x1, x2 = gs[i-1], gs[i], gs[i+1]
    y0, y1, y2 = scores[i-1], scores[i], scores[i+1]
    denom = (x0-x1)*(x0-x2)*(x1-x2)
    if abs(denom) < 1e-15: return float(x1)
    A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
    B = (x2*x2*(y0-y1) + x1*x1*(y2-y0) + x0*x0*(y1-y2)) / denom
    if abs(A) < 1e-15: return float(x1)
    g = -B/(2*A)
    return float(min(max(g, x0), x2))

def main():
    t0 = time.time()
    m = 2
    Ns = [4, 6, 8, 10, 12]
    gs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00, 1.57]
    seeds = [1, 2, 3, 4, 5]
    R = {"m": m, "Ns": Ns, "gs": gs, "seeds": seeds, "data": {}}
    gstars_mean, gstars_std = [], []

    for N in Ns:
        tN = time.time()
        per_seed_gstar = []
        # stocke aussi les courbes moyennes pour la figure
        var_acc = np.zeros(len(gs)); ent_acc = np.zeros(len(gs))
        for seed in seeds:
            vars_, ents_ = [], []
            for g in gs:
                v, e = measure(N, m, g, S=80, seed=seed)
                vars_.append(v); ents_.append(e)
            vars_ = np.array(vars_); ents_ = np.array(ents_)
            var_acc += vars_; ent_acc += ents_
            v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
            emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
            score = (ents_/emax) * (vars_/v0)
            per_seed_gstar.append(gstar_parabolic(gs, score))
        gm, gsd = float(np.mean(per_seed_gstar)), float(np.std(per_seed_gstar))
        gstars_mean.append(gm); gstars_std.append(gsd)
        R["data"][f"N_{N}"] = {
            "gstar_per_seed": per_seed_gstar, "gstar_mean": gm, "gstar_std": gsd,
            "var_mean_curve": (var_acc/len(seeds)).tolist(),
            "ent_mean_curve": (ent_acc/len(seeds)).tolist(),
        }
        print(f"N={N:2d} : g* = {gm:.3f} +/- {gsd:.3f}   ({time.time()-tN:.0f}s)", flush=True)

    # Fit loi de puissance g* = a*N^b (si g* varie) — log-log
    Ns_a = np.array(Ns, dtype=float); gst = np.array(gstars_mean)
    fit = None
    if np.all(gst > 0):
        b, loga = np.polyfit(np.log(Ns_a), np.log(gst), 1)
        fit = {"a": float(np.exp(loga)), "b": float(b)}
        print(f"\nFit loi de puissance : g*(N) = {np.exp(loga):.3f} * N^{b:.3f}", flush=True)
    R["gstars_mean"] = gstars_mean; R["gstars_std"] = gstars_std; R["powerlaw_fit"] = fit

    with open(os.path.join(HERE, "results_scaling_fenetre.json"), "w") as f:
        json.dump(R, f, indent=2)

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        for N in Ns:
            d = R["data"][f"N_{N}"]
            v = np.array(d["var_mean_curve"]); e = np.array(d["ent_mean_curve"])
            score = (e/e.max()) * (v/(v[0] if v[0] > 1e-15 else 1e-15))
            ax.plot(gs, score, marker="o", label=f"N={N}")
        ax.set_xlabel("couplage inter-module g"); ax.set_ylabel("score = ent_frac x var_frac")
        ax.set_title("Score du compromis selon g, par taille N"); ax.legend(); ax.grid(alpha=0.3)
        ax2 = axes[1]
        ax2.errorbar(Ns, gstars_mean, yerr=gstars_std, marker="s", capsize=4, label="g*(N) mesure")
        if fit:
            xs = np.linspace(min(Ns), max(Ns), 100)
            ax2.plot(xs, fit["a"]*xs**fit["b"], "--", label=f"fit {fit['a']:.2f}*N^{fit['b']:.2f}")
        ax2.set_xlabel("N (qubits)"); ax2.set_ylabel("g* optimal")
        ax2.set_title("Loi d'echelle de la fenetre optimale"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "resultats_scaling_fenetre.png"), dpi=130)
        print("Figure sauvee : resultats_scaling_fenetre.png", flush=True)
    except Exception as ex:
        print(f"(figure non generee : {ex})", flush=True)

    print(f"\nTermine en {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
