"""
LA preuve finale — le sweet spot donne-t-il une meilleure PERFORMANCE ?
======================================================================
On ENTRAÎNE un modèle quantique variationnel (par descente de gradient,
parameter-shift) sur une tâche de régression, à BUDGET D'ENTRAÎNEMENT FIXE,
pour chaque couplage inter-module g.

Prédiction de la thèse de Nico :
  g=0  : entraînable mais peu expressif        -> perf moyenne
  g=1  : expressif mais barren plateau (n'apprend pas) -> perf mauvaise
  g faible : expressif ET entraînable          -> MEILLEURE perf
=> courbe de performance en cloche, sommet au couplage FAIBLE.

Tâche : régression d'une fonction cible à plusieurs fréquences (data re-uploading).
N=6, blocs de 2, intrication intra forte (fixe) + inter = g.
"""
import numpy as np, time, json, os

def apply_1q(state,G,q,N):
    st=state.reshape([2]*N); st=np.tensordot(G,st,axes=([1],[q]))
    return np.moveaxis(st,0,q).reshape(-1)
def RY(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],dtype=complex)
def RZ(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=complex)

def zz_phase(N,m,g,alpha=np.pi/2):
    idx=np.arange(2**N); z=np.array([1-2*((idx>>(N-1-q))&1) for q in range(N)])
    ang=np.zeros(2**N)
    for b in range(0,N,m):
        blk=list(range(b,min(b+m,N)))
        for a in range(len(blk)):
            for c in range(a+1,len(blk)): ang+=alpha*z[blk[a]]*z[blk[c]]
    for b in range(0,N-m,m): ang+=g*z[b+m-1]*z[b+m]
    return np.exp(-0.5j*ang)

def forward(x, th, fi, N, ph):
    """Modèle : data re-uploading + couches variationnelles. Sortie <Z_0> in [-1,1]."""
    s=np.zeros(2**N,dtype=complex); s[0]=1.0; L=th.shape[0]
    for l in range(L):
        for q in range(N): s=apply_1q(s,RY(x*np.pi),q,N)     # encodage de x
        for q in range(N): s=apply_1q(s,RY(th[l,q]),q,N)     # variationnel
        for q in range(N): s=apply_1q(s,RZ(fi[l,q]),q,N)
        s=s*ph
    st=s.reshape([2]*N)
    return float(np.sum(np.abs(st[0])**2)-np.sum(np.abs(st[1])**2))

def predict_all(X, th, fi, N, ph): return np.array([forward(x,th,fi,N,ph) for x in X])
def mse(th,fi,X,Y,N,ph): return float(np.mean((predict_all(X,th,fi,N,ph)-Y)**2))

def train(N,m,g,X,Y,iters=50,lr=0.25,seed=0):
    rng=np.random.default_rng(seed); L=3; ph=zz_phase(N,m,g)
    th=rng.uniform(0,2*np.pi,(L,N)); fi=rng.uniform(0,2*np.pi,(L,N))
    for it in range(iters):
        pred=predict_all(X,th,fi,N,ph); err=pred-Y
        # gradient parameter-shift sur chaque param (moyenné sur les points)
        gth=np.zeros_like(th); gfi=np.zeros_like(fi)
        for l in range(L):
            for q in range(N):
                tp=th.copy(); tp[l,q]+=np.pi/2; tm=th.copy(); tm[l,q]-=np.pi/2
                d=(predict_all(X,tp,fi,N,ph)-predict_all(X,tm,fi,N,ph))/2
                gth[l,q]=np.mean(2*err*d)
                fp=fi.copy(); fp[l,q]+=np.pi/2; fm=fi.copy(); fm[l,q]-=np.pi/2
                d2=(predict_all(X,th,fp,N,ph)-predict_all(X,th,fm,N,ph))/2
                gfi[l,q]=np.mean(2*err*d2)
        th-=lr*gth; fi-=lr*gfi
    return mse(th,fi,X,Y,N,ph)

def main():
    t0=time.time(); N,m=6,2
    X=np.linspace(0,1,24)
    Y=0.6*np.cos(3*np.pi*X)+0.4*np.sin(5*np.pi*X); Y=Y/np.max(np.abs(Y))   # cible multi-fréquences
    gs=[0.0,0.1,0.2,0.4,0.7,1.0]; R={}
    print(f"Entrainement variationnel (N={N}, 50 iters) — MSE finale vs couplage g")
    print(f"(cloche avec sommet au g faible = these confirmee)\n")
    print(f"{'g':>5} | {'MSE finale (moy/3 seeds)':>24}")
    for g in gs:
        ms=[train(N,m,g,X,Y,seed=s) for s in [0,1,2]]
        R[f"g_{g}"]={"mse":float(np.mean(ms)),"std":float(np.std(ms))}
        print(f"{g:>5.2f} | {np.mean(ms):.4f} ± {np.std(ms):.4f}")
    best=min(R,key=lambda k:R[k]['mse'])
    print(f"\n  MEILLEURE perf : {best}  (MSE={R[best]['mse']:.4f})")
    print(f"  g=0 : {R['g_0.0']['mse']:.4f}  |  g=1 : {R['g_1.0']['mse']:.4f}")
    print(f"  Durée: {time.time()-t0:.0f}s")
    R["gs"]=gs
    with open(os.path.join(os.path.dirname(__file__),"results_perf.json"),"w") as f: json.dump(R,f,indent=2)

if __name__=="__main__":
    main()
