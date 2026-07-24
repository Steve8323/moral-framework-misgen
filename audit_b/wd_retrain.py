"""Retrain all 8 kinds WITH weight_decay=1e-3 to get genuine finite minima.
Confirm grad norm -> ~0, then re-test whether the kind-of-minimum->misgen finding
and the sharpness sign survive. Writes ONLY into audit_b/."""
import json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
torch.set_num_threads(8)
K_T,K_S,K_N=8,1,6; INPUT=K_T+K_S+K_N; HIDDEN=64
NOISE_T,NOISE_S=1.6,0.25; N_TRAIN=4000
WD=1e-3
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
def full_gradnorm(m,X,y,wd):
    # include weight-decay term so this is the gradnorm of the ACTUAL objective being minimized
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    gn=torch.sqrt(sum(((p.grad+wd*p)**2).sum() for p in m.parameters())).item(); m.zero_grad(); return gn
def sharp_acc(m,Xtr,ytr,scales=(0.5,1.0,1.5,2.0),ndir=12,seed=0):
    g=torch.Generator().manual_seed(seed)
    with torch.no_grad(): base=(m(Xtr).argmax(1)==ytr).float().mean().item()
    drops=[]; params=list(m.parameters()); orig=[p.detach().clone() for p in params]
    for scale in scales:
        for _ in range(ndir):
            with torch.no_grad():
                for p,o in zip(params,orig):
                    d=torch.randn(p.shape,generator=g); d=d*(o.norm()/(d.norm()+1e-12))*scale
                    p.copy_(o+d)
                drops.append(base-(m(Xtr).argmax(1)==ytr).float().mean().item())
                for p,o in zip(params,orig): p.copy_(o)
    return float(np.mean(drops))
def sharp_ce_small(m,Xtr,ytr,scales=(0.05,0.1,0.15,0.2),ndir=12,seed=0):
    g=torch.Generator().manual_seed(seed); lf=nn.CrossEntropyLoss()
    with torch.no_grad(): base=lf(m(Xtr),ytr).item()
    incs=[]; params=list(m.parameters()); orig=[p.detach().clone() for p in params]
    for scale in scales:
        for _ in range(ndir):
            with torch.no_grad():
                for p,o in zip(params,orig):
                    d=torch.randn(p.shape,generator=g); d=d*(o.norm()/(d.norm()+1e-12))*scale
                    p.copy_(o+d)
                incs.append(lf(m(Xtr),ytr).item()-base)
                for p,o in zip(params,orig): p.copy_(o)
    return float(np.mean(incs))
def train(seed,opt_name,batch,lr,wd,steps=800,rho_sam=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N_TRAIN,seed,'train'); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    if opt_name=='adam': opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    else: opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
    for _ in range(steps):
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
    return m,Xtr,ytr
# Use MORE steps (3000) so wd-regularized objective actually converges to finite min
configs=[]
for seed in range(6):
    configs+=[('SGD bs512',dict(opt_name='sgd',batch=512,lr=0.05)),
        ('SGD bs32',dict(opt_name='sgd',batch=32,lr=0.05)),
        ('SGD bs128 hl',dict(opt_name='sgd',batch=128,lr=0.20)),
        ('Adam bs128',dict(opt_name='adam',batch=128,lr=1e-3)),
        ('Adam bs512',dict(opt_name='adam',batch=512,lr=2e-3)),
        ('SAM0.05',dict(opt_name='sgd',batch=128,lr=0.05,rho_sam=0.05)),
        ('SAM0.15',dict(opt_name='sgd',batch=128,lr=0.05,rho_sam=0.15))]
Xood,yood=gen(4000,90002,'ood'); Xind,yind=gen(4000,90001,'indist')
rows=[]; t0=time.time()
for k,(kind,c) in enumerate(configs):
    m,Xtr,ytr=train(seed=1000+k,wd=WD,steps=3000,**c)
    rows.append(dict(kind=kind,misgen=1.0-acc(m,Xood,yood),tracc=acc(m,Xtr,ytr),
        indist=acc(m,Xind,yind),gradnorm=full_gradnorm(m,Xtr,ytr,WD),
        sh_large=sharp_acc(m,Xtr,ytr,seed=1000+k),sh_ce_small=sharp_ce_small(m,Xtr,ytr,seed=1000+k)))
print(f'trained {len(rows)} wd=1e-3 models, 3000 steps, in {time.time()-t0:.0f}s\n')
def col(k): return np.array([r[k] for r in rows])
mg=col('misgen'); tr=col('tracc'); ind=col('indist'); gn=col('gradnorm')
print('=== WD=1e-3 GENUINE-MINIMUM RETRAIN ===')
print('capability: train_acc %.4f+/-%.4f | indist %.4f+/-%.4f'%(tr.mean(),tr.std(),ind.mean(),ind.std()))
print('full-obj grad norm: mean %.2e range [%.2e,%.2e]  (was ~1e-4..1e-2 w/o wd)'%(gn.mean(),gn.min(),gn.max()))
print('misgen: mean %.3f range [%.3f,%.3f]  (2x spread? max/min=%.2f)'%(mg.mean(),mg.min(),mg.max(),mg.max()/mg.min()))
print()
print('sharpness sign survival:')
for key in ['sh_large','sh_ce_small','gradnorm']:
    v=col(key); pr,pp=pearsonr(v,mg); sr,sp=spearmanr(v,mg)
    print('  %-12s vs misgen: Pearson r=%+.3f (p=%.3f) | Spearman=%+.3f'%(key,pr,pp,sr))
print()
kinds={}
for r in rows: kinds.setdefault(r['kind'],[]).append(r)
print('%-14s %8s %8s %10s %10s'%('kind','misgen','tracc','gradnorm','sh_large'))
for kk in sorted(kinds,key=lambda z:np.mean([r['misgen'] for r in kinds[z]])):
    rs=kinds[kk]
    print('%-14s %8.3f %8.4f %10.2e %10.4f'%(kk,np.mean([r['misgen'] for r in rs]),
        np.mean([r['tracc'] for r in rs]),np.mean([r['gradnorm'] for r in rs]),np.mean([r['sh_large'] for r in rs])))
json.dump(rows,open('wd_retrain_rows.json','w'),indent=1)
