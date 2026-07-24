"""Re-train the 48 models, but evaluate sharpness at MANY scale sets and also
record small-scale CE-loss-based sharpness, gradient norm, weight norm.
Writes ONLY into audit_b/. """
import json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
torch.set_num_threads(8)
K_T,K_S,K_N=8,1,6; INPUT=K_T+K_S+K_N; HIDDEN=64
NOISE_T,NOISE_S=1.6,0.25; N_TRAIN=4000; RHO=1.0
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
                g=[p.grad.clone() for p in m.parameters()]
                gn=torch.sqrt(sum((gg**2).sum() for gg in g))+1e-12
                ew=[rho_sam*gg/gn for gg in g]
                for p,e in zip(m.parameters(),ew): p.add_(e)
            opt.zero_grad(); lf(m(xb),yb).backward()
            with torch.no_grad():
                for p,e in zip(m.parameters(),ew): p.sub_(e)
            opt.step()
        else:
            opt.zero_grad(); lf(m(xb),yb).backward(); opt.step()
    return m,Xtr,ytr
def sharp_acc(m,Xtr,ytr,scales,ndir=12,seed=0):
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
def sharp_ce(m,Xtr,ytr,scales,ndir=12,seed=0):
    """CE-LOSS based filter-normalized sharpness (the classic Li et al. metric)."""
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
def gradnorm(m,Xtr,ytr):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(Xtr),ytr).backward()
    gn=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return gn
def wnorm(m): 
    with torch.no_grad(): return torch.sqrt(sum((p**2).sum() for p in m.parameters())).item()

configs=[]
for seed in range(6):
    configs+=[dict(opt='sgd',batch=512,lr=0.05,wd=0.0),dict(opt='sgd',batch=32,lr=0.05,wd=0.0),
        dict(opt='sgd',batch=128,lr=0.20,wd=0.0),dict(opt='sgd',batch=128,lr=0.02,wd=5e-4),
        dict(opt='adam',batch=128,lr=1e-3,wd=0.0),dict(opt='adam',batch=512,lr=2e-3,wd=0.0),
        dict(opt='sgd',batch=128,lr=0.05,wd=0.0,rho_sam=0.05),dict(opt='sgd',batch=128,lr=0.05,wd=0.0,rho_sam=0.15)]
Xind,yind=gen(4000,90001,'indist'); Xood,yood=gen(4000,90002,'ood')
scale_sets={'orig(0.5-2.0)':(0.5,1.0,1.5,2.0),'small(0.05-0.2)':(0.05,0.1,0.15,0.2),
            'mid(0.2-0.5)':(0.2,0.3,0.4,0.5),'large(2-4)':(2.0,3.0,4.0)}
rows=[]
t0=time.time()
for k,c in enumerate(configs):
    m,Xtr,ytr=train(seed=1000+k,opt_name=c['opt'],batch=c['batch'],lr=c['lr'],wd=c['wd'],rho_sam=c.get('rho_sam',0.0))
    kind=('SAM'+str(c['rho_sam']) if c.get('rho_sam',0)>0 else
          (f"Adam bs{c['batch']}" if c['opt']=='adam' else
           f"SGD bs{c['batch']}"+(' wd' if c['wd']>0 else '')+(' hl' if c['lr']>=0.2 else '')))
    r=dict(kind=kind,misgen=1.0-acc(m,Xood,yood),gradnorm=gradnorm(m,Xtr,ytr),wnorm=wnorm(m),
           sharp_ce_small=sharp_ce(m,Xtr,ytr,(0.05,0.1,0.15,0.2),seed=1000+k))
    for nm,sc in scale_sets.items(): r['sh_'+nm]=sharp_acc(m,Xtr,ytr,sc,seed=1000+k)
    rows.append(r)
print(f'trained+measured 48 in {time.time()-t0:.0f}s\n')
misgen=np.array([r['misgen'] for r in rows])
print('=== Sensitivity of correlation to sharpness-metric/scale choice (n=48) ===')
for key in ['sh_orig(0.5-2.0)','sh_small(0.05-0.2)','sh_mid(0.2-0.5)','sh_large(2-4)','sharp_ce_small','gradnorm','wnorm']:
    v=np.array([r[key] for r in rows])
    pr,pp=pearsonr(v,misgen); sr,sp=spearmanr(v,misgen)
    print(f'  {key:18s} vs misgen: Pearson r={pr:+.3f} (p={pp:.3f}) | Spearman={sr:+.3f}  [mean={v.mean():.3g}]')
json.dump(rows,open('sharp_scale_rows.json','w'),indent=1)
