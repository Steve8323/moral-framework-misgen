"""
Rigorous local-minimum verification for the goal-misgen + framework setups.
For representative models we measure, at the converged solution:
  - gradient norm (proximity to a critical point)
  - top & MIN Hessian eigenvalue via Hessian-vector products + power iteration
    (min eigenvalue >~ 0  => local MIN;  clearly negative => SADDLE)
  - weight norm + its growth under continued training (CE-on-separable plateau check)
  - stability of the behavioral metric under 10x more training
  - a weight-decayed variant that HAS a genuine finite local minimum, as a control.
All numbers from real computation.
"""
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(8)

# ---------- goal-misgen data (mirrors experiment3) ----------
K_T,K_S,K_N=8,1,6; INPUT=K_T+K_S+K_N; HIDDEN=64; NOISE_T=1.6; NOISE_S=0.25; N=4000
def gen(n,seed,mode):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n)
    s={"train":y.copy(),"indist":y.copy(),"ood":1-y}[mode]
    T=(2*y-1).astype(np.float32)[:,None]+NOISE_T*rng.standard_normal((n,K_T)).astype(np.float32)
    S=(2*s-1).astype(np.float32)[:,None]+NOISE_S*rng.standard_normal((n,K_S)).astype(np.float32)
    Z=rng.standard_normal((n,K_N)).astype(np.float32)
    return torch.tensor(np.concatenate([T,S,Z],1).astype(np.float32)), torch.tensor(y.astype(np.int64))
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.fc1=nn.Linear(INPUT,HIDDEN); s.fc2=nn.Linear(HIDDEN,2)
    def forward(s,x): return s.fc2(torch.relu(s.fc1(x)))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()

def train(seed,opt_name,batch,lr,wd,steps=800,rho_sam=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N,seed,"train"); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    opt=(torch.optim.Adam(m.parameters(),lr,weight_decay=wd) if opt_name=="adam"
         else torch.optim.SGD(m.parameters(),lr,momentum=0.9,weight_decay=wd))
    for _ in range(steps):
        i=torch.randint(0,n,(batch,)); xb,yb=Xtr[i],ytr[i]
        if rho_sam>0:
            opt.zero_grad(); lf(m(xb),yb).backward()
            with torch.no_grad():
                g=[p.grad.clone() for p in m.parameters()]; gn=torch.sqrt(sum((q**2).sum() for q in g))+1e-12
                ew=[rho_sam*q/gn for q in g]
                for p,e in zip(m.parameters(),ew): p.add_(e)
            opt.zero_grad(); lf(m(xb),yb).backward()
            with torch.no_grad():
                for p,e in zip(m.parameters(),ew): p.sub_(e)
            opt.step()
        else:
            opt.zero_grad(); lf(m(xb),yb).backward(); opt.step()
    return m,Xtr,ytr

def grad_norm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return g

def weight_norm(m): return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters())).item()

def hvp(m,X,y,vec):
    lf=nn.CrossEntropyLoss(); params=list(m.parameters())
    loss=lf(m(X),y); g=torch.autograd.grad(loss,params,create_graph=True)
    flat=torch.cat([gi.reshape(-1) for gi in g]); gv=(flat*vec).sum()
    hv=torch.autograd.grad(gv,params,retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hv]).detach()

def eig_extremes(m,X,y,iters=60):
    nparam=sum(p.numel() for p in m.parameters())
    # subsample for speed/stability
    idx=torch.randperm(len(X))[:1000]; Xs,ys=X[idx],y[idx]
    v=torch.randn(nparam); v/=v.norm()
    for _ in range(iters):
        hv=hvp(m,Xs,ys,v); v=hv/(hv.norm()+1e-12)
    lam_top=(v*hvp(m,Xs,ys,v)).sum().item()
    # min eig via power iteration on (lam_top*I - H)
    shift=abs(lam_top)+1.0
    w=torch.randn(nparam); w/=w.norm()
    for _ in range(iters):
        hw=shift*w-hvp(m,Xs,ys,w); w=hw/(hw.norm()+1e-12)
    mu=(w*(shift*w-hvp(m,Xs,ys,w))).sum().item()
    lam_min=shift-mu
    return lam_top, lam_min

def report(tag, m, Xtr, ytr):
    Xood,yood=gen(4000,90002,"ood"); Xind,yind=gen(4000,90001,"indist")
    gn=grad_norm(m,Xtr,ytr); wn=weight_norm(m); lt,lmin=eig_extremes(m,Xtr,ytr)
    print(f"\n[{tag}]")
    print(f"  train_acc={accm(m,Xtr,ytr):.3f}  indist_acc={accm(m,Xind,yind):.3f}  ood_acc={accm(m,Xood,yood):.3f}  (misgen={1-accm(m,Xood,yood):.3f})")
    print(f"  grad_norm={gn:.4e}  weight_norm={wn:.3f}")
    print(f"  Hessian top eig={lt:+.4e}  min eig={lmin:+.4e}   -> {'LOCAL MIN (min eig >~ 0)' if lmin>-1e-2 else 'SADDLE (negative curvature)'}")
    return gn,wn,lt,lmin

print("="*70); print("GOAL-MISGEN models (experiment 3)"); print("="*70)
# low-misgen (Adam bs128) vs high-misgen (SGD high-lr)
mA,XA,yA=train(seed=2001,opt_name="adam",batch=128,lr=1e-3,wd=0.0,steps=800)
report("Adam bs128 (low misgen) @800 steps", mA,XA,yA)
mH,XH,yH=train(seed=2002,opt_name="sgd",batch=128,lr=0.20,wd=0.0,steps=800)
report("SGD high-lr (high misgen) @800 steps", mH,XH,yH)

print("\n--- CE-on-separable plateau check: weight-norm growth with longer training ---")
for nm,(opt,bs,lr) in [("Adam bs128",("adam",128,1e-3)),("SGD high-lr",("sgd",128,0.20))]:
    wns=[]; oods=[]
    for steps in [400,800,2000,8000]:
        m,Xt,yt=train(seed=2003,opt_name=opt,batch=bs,lr=lr,wd=0.0,steps=steps)
        Xood,yood=gen(4000,90002,"ood"); wns.append(weight_norm(m)); oods.append(1-accm(m,Xood,yood))
    print(f"  {nm:12s} steps[400,800,2000,8000] weight_norm={['%.2f'%w for w in wns]}  misgen={['%.3f'%o for o in oods]}")

print("\n" + "="*70); print("WEIGHT-DECAYED control: genuine finite local minima"); print("="*70)
mAd,XAd,yAd=train(seed=2001,opt_name="adam",batch=128,lr=1e-3,wd=1e-3,steps=3000)
report("Adam bs128 + wd1e-3 (low misgen?)", mAd,XAd,yAd)
mHd,XHd,yHd=train(seed=2002,opt_name="sgd",batch=128,lr=0.20,wd=1e-3,steps=3000)
report("SGD high-lr + wd1e-3 (high misgen?)", mHd,XHd,yHd)

print("\nDONE.")
