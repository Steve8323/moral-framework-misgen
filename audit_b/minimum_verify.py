"""Local-minimum verification for low-misgen (Adam bs128) vs high-misgen (SGD high-lr).
Measures: full-train grad norm, weight-norm trajectory under 5-10x longer training,
margin/weight-norm growth (separable-data plateau), and a weight-decay retrain to get
genuine finite minima. Writes ONLY into audit_b/."""
import numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
torch.set_num_threads(8)
K_T,K_S,K_N=8,1,6; INPUT=K_T+K_S+K_N; HIDDEN=64
NOISE_T,NOISE_S=1.6,0.25; N_TRAIN=4000
def gen(n,seed,mode):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n)
    if mode=='train': s=y.copy()
    elif mode=='indist': s=y.copy()
    elif mode=='ood': s=1-y
    T=(2*y-1).astype(np.float32)[:,None]+NOISE_T*rng.standard_normal((n,K_T)).astype(np.float32)
    S=(2*s-1).astype(np.float32)[:,None]+NOISE_S*rng.standard_normal((n,K_S)).astype(np.float32)
    Nz=rng.standard_normal((n,K_N)).astype(np.float32)
    X=np.concatenate([T,S,Nz],1).astype(np.float32)
    return torch.tensor(X),torch.tensor(y.astype(np.int64))
class MLP(nn.Module):
    def __init__(self): super().__init__(); self.fc1=nn.Linear(INPUT,HIDDEN); self.fc2=nn.Linear(HIDDEN,2)
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x)))
def acc(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def full_gradnorm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    gn=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return gn
def wnorm(m):
    with torch.no_grad(): return torch.sqrt(sum((p**2).sum() for p in m.parameters())).item()
def train_track(seed,opt_name,batch,lr,wd,steps,rho_sam=0.0,track_at=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N_TRAIN,seed,'train'); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    Xood,yood=gen(4000,90002,'ood')
    if opt_name=='adam': opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    else: opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
    track={} if track_at else None
    for t in range(steps):
        i=torch.randint(0,n,(batch,)); xb,yb=Xtr[i],ytr[i]
        if rho_sam>0:
            opt.zero_grad(); lf(m(xb),yb).backward()
            with torch.no_grad():
                g=[p.grad.clone() for p in m.parameters()]; gnn=torch.sqrt(sum((gg**2).sum() for gg in g))+1e-12
                ew=[rho_sam*gg/gnn for gg in g]
                for p,e in zip(m.parameters(),ew): p.add_(e)
            opt.zero_grad(); lf(m(xb),yb).backward()
            with torch.no_grad():
                for p,e in zip(m.parameters(),ew): p.sub_(e)
            opt.step()
        else:
            opt.zero_grad(); lf(m(xb),yb).backward(); opt.step()
        if track_at and (t+1) in track_at:
            with torch.no_grad(): ce=lf(m(Xtr),ytr).item()
            track[t+1]=dict(wnorm=wnorm(m),gradnorm=full_gradnorm(m,Xtr,ytr),
                            tracc=acc(m,Xtr,ytr),ood=acc(m,Xood,yood),ce=ce)
    return m,Xtr,ytr,track

cfgs={'Adam bs128 (low-misgen)':dict(opt_name='adam',batch=128,lr=1e-3,wd=0.0),
      'SGD bs128 hl (high-misgen)':dict(opt_name='sgd',batch=128,lr=0.20,wd=0.0)}
track_at=[800,1600,4000,8000]
print('=== LOCAL-MINIMUM VERIFICATION: grad-norm + long-train stability ===\n')
for name,c in cfgs.items():
    m,Xtr,ytr,tr=train_track(seed=1234,steps=8000,track_at=track_at,**c)
    print(name)
    print('  step | train_acc  OOD_acc   CE_loss    wnorm    gradnorm')
    for t in track_at:
        d=tr[t]; print('  %5d| %8.4f %8.4f %9.2e %8.3f %10.2e'%(t,d['tracc'],d['ood'],d['ce'],d['wnorm'],d['gradnorm']))
    print('  -> wnorm 800->8000: %.2f -> %.2f (x%.2f) | gradnorm 800->8000: %.1e -> %.1e | OOD 800->8000: %.3f -> %.3f'%(
        tr[800]['wnorm'],tr[8000]['wnorm'],tr[8000]['wnorm']/tr[800]['wnorm'],
        tr[800]['gradnorm'],tr[8000]['gradnorm'],tr[800]['ood'],tr[8000]['ood']))
    print()
