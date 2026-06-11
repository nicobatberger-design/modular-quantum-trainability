"""
LE test qui scelle tout — le sweet spot existe-t-il ?
=====================================================
On mesure, en fonction du couplage inter-module g, DEUX grandeurs :
  - ENTRAÎNABILITÉ : Var(gradient) (haute = pas de barren plateau)
  - INTRICATION    : entropie d'intrication (bipartition moitié/moitié)

Hypothèse (la thèse de Nico) : il existe une fenêtre de g FAIBLE où l'intrication
est déjà > 0 (ressource quantique présente) ET la variance encore haute (entraînable).
= le couplage faible optimise le compromis avantage-quantique / entraînabilité.

Système : N qubits en blocs de 2, intrication intra forte (fixe), inter = g (variable).
"""
import numpy as np, time, json, os

def apply_1q(state,G,q,N):
    st=state.reshape([2]*N); st=np.tensordot(G,st,axes=([1],[q]))
    return np.moveaxis(st,0,q).reshape(-1)
def RY(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],dtype=complex)
def RZ(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=complex)

def zz_phase_vector(N,m,g,alpha=np.pi/2):
    """Phase exp(-i/2 * sum_pairs angle * z_i z_j) : intra=alpha (fort), inter=g."""
    idx=np.arange(2**N)
    z=np.zeros((N,2**N))
    for q in range(N): z[q]=1-2*((idx>>(N-1-q))&1)   # +1/-1 selon le bit
    ang=np.zeros(2**N)
    # paires intra-bloc
    for b in range(0,N,m):
        blk=list(range(b,min(b+m,N)))
        for a in range(len(blk)):
            for c in range(a+1,len(blk)): ang+=alpha*z[blk[a]]*z[blk[c]]
    # paires inter-bloc (chaîne : dernier d'un bloc <-> premier du suivant)
    for b in range(0,N-m,m):
        ang+=g*z[b+m-1]*z[b+m]
    return np.exp(-0.5j*ang)

def run(th,fi,N,phase):
    s=np.zeros(2**N,dtype=complex); s[0]=1.0; L=th.shape[0]
    for l in range(L):
        for q in range(N): s=apply_1q(s,RY(th[l,q]),q,N)
        for q in range(N): s=apply_1q(s,RZ(fi[l,q]),q,N)
        s=s*phase
    return s

def cost_Z0(s,N):
    st=s.reshape([2]*N); return float(np.sum(np.abs(st[0])**2)-np.sum(np.abs(st[1])**2))

def ent_entropy(s,N):
    """Entropie d'intrication, bipartition [N//2 | reste]."""
    a=N//2; M=s.reshape(2**a,2**(N-a))
    sv=np.linalg.svd(M,compute_uv=False); p=sv**2; p=p[p>1e-12]
    return float(-np.sum(p*np.log(p)))

def measure(N,m,g,S=80,seed=1):
    rng=np.random.default_rng(seed); L=2*N; ph=zz_phase_vector(N,m,g)
    grads=[]; ents=[]
    for _ in range(S):
        th=rng.uniform(0,2*np.pi,(L,N)); fi=rng.uniform(0,2*np.pi,(L,N))
        thp=th.copy(); thp[0,0]+=np.pi/2; thm=th.copy(); thm[0,0]-=np.pi/2
        grads.append((cost_Z0(run(thp,fi,N,ph),N)-cost_Z0(run(thm,fi,N,ph),N))/2)
        ents.append(ent_entropy(run(th,fi,N,ph),N))
    return float(np.var(grads)), float(np.mean(ents))

def main():
    t0=time.time(); N,m=10,2; R={}
    gs=[0.0,0.05,0.1,0.2,0.4,0.7,1.0,1.57]
    print(f"N={N}, blocs de {m} — balayage du couplage inter-module g\n")
    print(f"{'g':>5} | {'Var(grad)':>11} | {'intrication':>11}")
    for g in gs:
        v,e=measure(N,m,g)
        R[f"g_{g}"]={"var":v,"ent":e}
        print(f"{g:>5.2f} | {v:11.2e} | {e:11.3f}")
    print(f"\n  Cherche la fenêtre : intrication > 0 ET Var encore haute (~1e-1).")
    print(f"  Durée: {time.time()-t0:.0f}s")
    R["N"]=N; R["gs"]=gs
    with open(os.path.join(os.path.dirname(__file__),"results_sweetspot.json"),"w") as f: json.dump(R,f,indent=2)

if __name__=="__main__":
    main()
