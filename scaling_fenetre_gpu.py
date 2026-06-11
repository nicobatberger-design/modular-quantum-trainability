"""
Scaling de la fenetre g*(N) — version GPU (Kaggle/Colab) ou CPU fallback
========================================================================
Extension de scaling_fenetre.py vers N=16-24 pour TRANCHER la question
ouverte : g*(N) est-il un plateau (~0.4 constant) ou une decroissance ?

Le run CPU local (10/06) donnait N=6..12 : g* dans [0.32, 0.48], fit
g*=1.08*N^-0.47 mais R2=0.49 (indetermine). N=16-24 doit lever le doute.

PORTAGE : remplace numpy par cupy si dispo (GPU), sinon numpy (CPU lent).
Etat-vecteur complex128 : 2^N * 16 octets. N=24 = 256 MiB, N=26 = 1 GiB,
N=28 = 4 GiB -> tient sur un T4 16 GB. Au-dela : complex64 ou tensor-network.

Optimisations vs version CPU :
- Entropie via PURETE de Renyi-2 (tr(rho_A^2)) au lieu de SVD (entropie de
  von Neumann) : evite la SVD O(2^N * 2^(N/2)) couteuse, garde une mesure
  d'intrication monotone. Renyi-2 = -log(tr rho_A^2). Suffisant pour situer g*.
- Phase ZZ appliquee comme vecteur diagonal (deja le cas), portes 1q par
  reshape/tensordot (compatible cupy).

Usage Kaggle : !pip install cupy-cuda12x  (souvent preinstalle) ; mettre
ACCELERATOR=GPU T4. Puis copier-coller ce fichier dans une cellule + main().
"""
import time, json, os

try:
    import cupy as xp
    BACKEND = "cupy(GPU)"
except Exception:
    import numpy as xp
    BACKEND = "numpy(CPU)"

import numpy as np  # toujours dispo pour les petits calculs hote

def apply_1q(state, G, q, N):
    st = state.reshape([2]*N)
    st = xp.tensordot(G, st, axes=([1],[q]))
    return xp.moveaxis(st, 0, q).reshape(-1)

def RY(t):
    c, s = np.cos(t/2), np.sin(t/2)
    return xp.asarray([[c,-s],[s,c]], dtype=xp.complex128)

def RZ(t):
    return xp.asarray([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]], dtype=xp.complex128)

def zz_phase_vector(N, m, g, alpha=np.pi/2):
    idx = xp.arange(2**N)
    z = xp.zeros((N, 2**N))
    for q in range(N):
        z[q] = 1 - 2*((idx >> (N-1-q)) & 1)
    ang = xp.zeros(2**N)
    for b in range(0, N, m):
        blk = list(range(b, min(b+m, N)))
        for a in range(len(blk)):
            for c in range(a+1, len(blk)):
                ang += alpha*z[blk[a]]*z[blk[c]]
    for b in range(0, N-m, m):
        ang += g*z[b+m-1]*z[b+m]
    return xp.exp(-0.5j*ang)

def run(th, fi, N, phase):
    s = xp.zeros(2**N, dtype=xp.complex128); s[0] = 1.0; L = th.shape[0]
    for l in range(L):
        for q in range(N): s = apply_1q(s, RY(float(th[l,q])), q, N)
        for q in range(N): s = apply_1q(s, RZ(float(fi[l,q])), q, N)
        s = s*phase
    return s

def cost_Z0(s, N):
    st = s.reshape([2]*N)
    p0 = xp.sum(xp.abs(st[0])**2); p1 = xp.sum(xp.abs(st[1])**2)
    return float(p0 - p1)

def renyi2(s, N):
    """Intrication via purete de Renyi-2 : S2 = -log tr(rho_A^2), A = N//2 qubits.
    tr(rho_A^2) = || M M^dagger ||_F^2 ou M est la matrice (2^a x 2^(N-a))."""
    a = N//2
    M = s.reshape(2**a, 2**(N-a))
    rho = M @ M.conj().T            # 2^a x 2^a, a = N//2 -> gerable jusqu'a N~28
    purity = float(xp.real(xp.sum(rho * rho.conj())))
    purity = min(max(purity, 1e-12), 1.0)
    return float(-np.log(purity))

def measure(N, m, g, S=60, seed=1):
    rng = np.random.default_rng(seed); L = 2*N
    ph = zz_phase_vector(N, m, g)
    grads = []; ents = []
    for _ in range(S):
        th = rng.uniform(0, 2*np.pi, (L, N)); fi = rng.uniform(0, 2*np.pi, (L, N))
        thp = th.copy(); thp[0,0] += np.pi/2; thm = th.copy(); thm[0,0] -= np.pi/2
        grads.append((cost_Z0(run(thp, fi, N, ph), N) - cost_Z0(run(thm, fi, N, ph), N))/2)
        ents.append(renyi2(run(th, fi, N, ph), N))
    return float(np.var(grads)), float(np.mean(ents))

def gstar_parabolic(gs, scores):
    i = int(np.argmax(scores))
    if i == 0 or i == len(gs)-1: return float(gs[i])
    x0,x1,x2 = gs[i-1],gs[i],gs[i+1]; y0,y1,y2 = scores[i-1],scores[i],scores[i+1]
    d = (x0-x1)*(x0-x2)*(x1-x2)
    if abs(d) < 1e-15: return float(x1)
    A = (x2*(y1-y0)+x1*(y0-y2)+x0*(y2-y1))/d
    B = (x2*x2*(y0-y1)+x1*x1*(y2-y0)+x0*x0*(y1-y2))/d
    if abs(A) < 1e-15: return float(x1)
    return float(min(max(-B/(2*A), x0), x2))

def main(Ns=(16,18,20,22,24), seeds=(1,2,3), S=60):
    print(f"Backend : {BACKEND}", flush=True)
    m = 2
    gs = [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.70,1.00,1.57]
    R = {"backend": BACKEND, "m": m, "Ns": list(Ns), "gs": gs, "seeds": list(seeds), "S": S, "data": {}}
    gstars = []
    for N in Ns:
        tN = time.time(); per_seed = []
        for seed in seeds:
            vars_, ents_ = [], []
            for g in gs:
                v, e = measure(N, m, g, S=S, seed=seed)
                vars_.append(v); ents_.append(e)
            vars_ = np.array(vars_); ents_ = np.array(ents_)
            v0 = vars_[0] if vars_[0] > 1e-15 else 1e-15
            emax = ents_.max() if ents_.max() > 1e-15 else 1e-15
            score = (ents_/emax)*(vars_/v0)
            per_seed.append(gstar_parabolic(gs, score))
        gm, gsd = float(np.mean(per_seed)), float(np.std(per_seed))
        gstars.append(gm)
        R["data"][f"N_{N}"] = {"gstar_mean": gm, "gstar_std": gsd, "gstar_per_seed": per_seed}
        print(f"N={N:2d} : g* = {gm:.3f} +/- {gsd:.3f}   ({time.time()-tN:.0f}s)", flush=True)
    R["gstars_mean"] = gstars
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".", "results_scaling_gpu.json")
    with open(out, "w") as f: json.dump(R, f, indent=2)
    print(f"\nSauve : {out}", flush=True)
    # combine avec le run CPU N=6-12 pour le fit global si dispo
    print("Pour le verdict : combiner avec results_scaling_fenetre.json (N=6-12) et refit la loi de puissance.", flush=True)

if __name__ == "__main__":
    main()
