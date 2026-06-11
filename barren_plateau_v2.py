"""
TEST PÉPITE v2 — réparé pour INDUIRE un vrai barren plateau
===========================================================
v1 échouait : le monolithe ne s'effondrait pas (intrication/profondeur trop faibles).
Réparations :
  - intrication MONOLITHE = all-to-all par couche (CZ sur toutes les paires)
    -> propagation instantanée, régime barren plateau atteint
  - profondeur L = 2N, rotations RY+RZ (plus expressif)
  - modulaire : all-to-all INTRA-bloc seulement (qubit 0 confiné au bloc 0)
  - coût local O = Z_0, gradient parameter-shift sur le 1er RY
Le all-to-all est appliqué efficacement via un vecteur de phases (poids de Hamming).
"""
import numpy as np, time, json, os

def apply_1q(state, G, q, N):
    st=state.reshape([2]*N); st=np.tensordot(G,st,axes=([1],[q]))
    return np.moveaxis(st,0,q).reshape(-1)

def RY(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],dtype=complex)
def RZ(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=complex)

def entangle_phases(N, m, modular):
    """Vecteur de phases pour une couche d'intrication all-to-all (globale ou intra-bloc)."""
    ph=np.ones(2**N,dtype=complex)
    for i in range(2**N):
        bits=[(i>>(N-1-q))&1 for q in range(N)]
        if not modular:
            k=sum(bits); ph[i]=(-1)**(k*(k-1)//2)
        else:
            p=1
            for b in range(0,N,m):
                kk=sum(bits[b:b+m]); p*= (-1)**(kk*(kk-1)//2)
            ph[i]=p
    return ph

def run(thetas, phis, N, ph):
    state=np.zeros(2**N,dtype=complex); state[0]=1.0; L=thetas.shape[0]
    for layer in range(L):
        for q in range(N): state=apply_1q(state,RY(thetas[layer,q]),q,N)
        for q in range(N): state=apply_1q(state,RZ(phis[layer,q]),q,N)
        state=state*ph
    return state

def cost_Z0(state,N):
    st=state.reshape([2]*N); return float(np.sum(np.abs(st[0])**2)-np.sum(np.abs(st[1])**2))

def grad_var(N,m,modular,S=100,seed=1):
    rng=np.random.default_rng(seed); L=2*N; ph=entangle_phases(N,m,modular); g=[]
    for _ in range(S):
        th=rng.uniform(0,2*np.pi,(L,N)); fi=rng.uniform(0,2*np.pi,(L,N))
        thp=th.copy(); thp[0,0]+=np.pi/2; thm=th.copy(); thm[0,0]-=np.pi/2
        g.append((cost_Z0(run(thp,fi,N,ph),N)-cost_Z0(run(thm,fi,N,ph),N))/2)
    return float(np.var(g))

def main():
    t0=time.time(); Ns=[4,6,8,10,12]; m=2; R={"monolithe":{},"modulaire":{}}
    print("v2 (all-to-all, L=2N) — Var(gradient) vs N\n")
    print(f"{'N':>3} | {'monolithe':>12} | {'modulaire':>12} | ratio")
    for N in Ns:
        vm=grad_var(N,m,False); vd=grad_var(N,m,True)
        R['monolithe'][N]=vm; R['modulaire'][N]=vd
        print(f"{N:>3} | {vm:12.2e} | {vd:12.2e} | {vd/vm:5.1f}x")
    lm=np.polyfit(Ns,np.log([R['monolithe'][N] for N in Ns]),1)[0]
    ld=np.polyfit(Ns,np.log([R['modulaire'][N] for N in Ns]),1)[0]
    print(f"\n  pente log(Var)/N — monolithe: {lm:+.3f} | modulaire: {ld:+.3f}")
    print(f"  barren plateau monolithe si pente << 0 ; pépite si modulaire ~ 0")
    print(f"  Durée: {time.time()-t0:.0f}s")
    R['slopes']={'monolithe':float(lm),'modulaire':float(ld)}; R['Ns']=Ns
    with open(os.path.join(os.path.dirname(__file__),"results_barren_v2.json"),"w") as f: json.dump(R,f,indent=2)

if __name__=="__main__":
    main()
