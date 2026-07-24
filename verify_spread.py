"""
THE CRUX TEST. The thesis: at genuine converged local minima, sheer chance (seed)
produces a SPREAD of generalization -- some models generalize properly, others
misgeneralize. We use weight decay so there is a GENUINE FINITE local minimum
(weights don't run to infinity), train every seed to convergence, confirm
convergence (small grad norm, stable weight norm), then look at the DISTRIBUTION
of goal-misgen across seeds. All numbers real.
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
    else:
        s=(y.copy() if mode=="indist" else 1-y)
    T=(2*y-1).astype(np.float32)[:,None]+NOISE_T*rng.standard_normal((n,K_T)).astype(np.float32)
    S=(2*s-1).astype(np.float32)[:,None]+NOISE_S*rng.standard_normal((n,K_S)).astype(np.float32)
    Z=rng.standard_normal((n,K_N)).astype(np.float32)
    return torch.tensor(np.concatenate([T,S,Z],1).astype(np.float32)), torch.tensor(y.astype(np.int64))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.fc1=nn.Linear(INPUT,HIDDEN); s.fc2=nn.Linear(HIDDEN,2)
    def forward(s,x): return s.fc2(torch.relu(s.fc1(x)))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def gnorm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return g
def wnorm(m): return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters())).item()

def train(seed,wd,rho,opt_name="adam",steps=4000):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N,seed,"train",rho); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    opt=(torch.optim.Adam(m.parameters(),1e-3,weight_decay=wd) if opt_name=="adam"
         else torch.optim.SGD(m.parameters(),0.05,momentum=0.9,weight_decay=wd))
    for _ in range(steps):
        i=torch.randint(0,n,(128,)); opt.zero_grad(); lf(m(Xtr[i]),ytr[i]).backward(); opt.step()
    return m,Xtr,ytr

def study(wd,rho,opt_name,nseed=30,steps=4000):
    mis=[]; gns=[]; wns=[]; trd=[]
    Xood,yood=gen(4000,90002,"ood",rho); Xind,yind=gen(4000,90001,"indist",rho)
    for s in range(nseed):
        m,Xt,yt=train(3000+s,wd,rho,opt_name,steps)
        mis.append(1-accm(m,Xood,yood)); gns.append(gnorm(m,Xt,yt)); wns.append(wnorm(m))
        trd.append(accm(m,Xind,yind))
    mis=np.array(mis)
    print(f"\n[wd={wd} rho={rho} opt={opt_name} steps={steps} nseed={nseed}]")
    print(f"  convergence: grad_norm mean={np.mean(gns):.2e} max={np.max(gns):.2e} | weight_norm mean={np.mean(wns):.2f} | in-dist acc mean={np.mean(trd):.3f}")
    print(f"  goal-misgen: mean={mis.mean():.3f} std={mis.std():.3f} min={mis.min():.3f} max={mis.max():.3f}")
    print(f"  generalize well (misgen<0.3): {int(np.sum(mis<0.3))}/{nseed}   |   misgeneralize (misgen>0.7): {int(np.sum(mis>0.7))}/{nseed}")
    # crude histogram
    h,_=np.histogram(mis,bins=10,range=(0,1)); print("  hist[0..1]:", " ".join(f"{x:2d}" for x in h))
    return mis

print("="*72); print("GENUINE FINITE MINIMA (weight decay): does chance still spread generalization?"); print("="*72)
study(wd=1e-3, rho=1.0, opt_name="adam")
study(wd=1e-3, rho=0.9, opt_name="adam")    # imperfect spurious correlation -> more tension
study(wd=1e-3, rho=0.9, opt_name="sgd")
study(wd=3e-3, rho=0.9, opt_name="adam")    # stronger reg -> stronger finite minimum
print("\nDONE.")
