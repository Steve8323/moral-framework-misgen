"""
Where (if anywhere) does CHANCE produce different generalization at GENUINE minima?
Scan the spurious-correlation strength rho across the crossover between "true cue wins"
and "spurious cue wins". At the knife-edge the two solutions are loss-equivalent, so
the seed may decide -> the regime where the thesis can hold. Weight decay => finite minima.
"""
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(8)
K_T,K_S,K_N=8,1,6; INPUT=K_T+K_S+K_N; HIDDEN=64; NOISE_T=1.6; NOISE_S=0.25; N=4000
def gen(n,seed,mode,rho):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n)
    if mode=="train":
        s=y.copy()
        if rho<1.0:
            f=rng.random(n)>rho; s[f]=rng.integers(0,2,f.sum())
    else: s=(y.copy() if mode=="indist" else 1-y)
    T=(2*y-1).astype(np.float32)[:,None]+NOISE_T*rng.standard_normal((n,K_T)).astype(np.float32)
    S=(2*s-1).astype(np.float32)[:,None]+NOISE_S*rng.standard_normal((n,K_S)).astype(np.float32)
    Z=rng.standard_normal((n,K_N)).astype(np.float32)
    return torch.tensor(np.concatenate([T,S,Z],1).astype(np.float32)), torch.tensor(y.astype(np.int64))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.fc1=nn.Linear(INPUT,HIDDEN); s.fc2=nn.Linear(HIDDEN,2)
    def forward(s,x): return s.fc2(torch.relu(s.fc1(x)))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def train(seed,wd,rho,steps=4000):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N,seed,"train",rho); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=wd)
    for _ in range(steps):
        i=torch.randint(0,n,(128,)); opt.zero_grad(); lf(m(Xtr[i]),ytr[i]).backward(); opt.step()
    return m
print("rho-scan at genuine minima (wd=1e-3, 30 seeds each). Looking for a regime with WIDE/BIMODAL spread.")
print(f"{'rho':5s} | {'misgen mean':11s} {'std':6s} {'min':5s} {'max':5s} | {'gen<0.3':7s} {'mis>0.7':7s} | hist[0..1]")
for rho in [0.92,0.94,0.96,0.97,0.98,0.99,1.0]:
    Xood,yood=gen(4000,90002,"ood",rho)
    mis=np.array([1-accm(train(3000+s,1e-3,rho),Xood,yood) for s in range(30)])
    h,_=np.histogram(mis,bins=10,range=(0,1))
    print(f"{rho:5.2f} | {mis.mean():11.3f} {mis.std():6.3f} {mis.min():5.2f} {mis.max():5.2f} | "
          f"{int((mis<0.3).sum()):7d} {int((mis>0.7).sum()):7d} | "+" ".join(f"{x:2d}" for x in h))
print("\nDONE.")
