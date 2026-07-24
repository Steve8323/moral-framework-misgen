"""Focused search: small init + narrow width + SGD+momentum to maximize spontaneous
framework commitment (gradient-starvation / winner-take-all regime)."""
import numpy as np, torch, torch.nn as nn, time
torch.set_num_threads(8)

def make_features(latent, k, noise, rng):
    base = (2*latent-1).astype(np.float32)[:,None]
    return base + noise*rng.standard_normal((len(latent),k)).astype(np.float32)

def gen(n, seed, dim, noise, conflict=False):
    rng = np.random.default_rng(seed)
    c = rng.integers(0,2,n); d = (rng.integers(0,2,n) if conflict else c.copy())
    if conflict:
        keep=c!=d; c,d=c[keep],d[keep]
    Xc=make_features(c,dim,noise,rng); Xd=make_features(d,dim,noise,rng)
    Xn=noise*rng.standard_normal((len(c),4)).astype(np.float32)
    X=np.concatenate([Xc,Xd,Xn],1).astype(np.float32)
    return torch.tensor(X), torch.tensor(c.astype(np.int64)), torch.tensor(d.astype(np.int64))

def run_cfg(dim, noise, lr, mom, steps, hidden, init_scale, n_models=40):
    INPUT=2*dim+4
    Xc,c_e,d_e = gen(3000, 77777, dim, noise, conflict=True)
    idxs=[]; losses=[]
    for s in range(n_models):
        torch.manual_seed(s); np.random.seed(s)
        Xtr,ytr,_ = gen(3000, s, dim, noise)
        net=nn.Sequential(nn.Linear(INPUT,hidden), nn.ReLU(), nn.Linear(hidden,2))
        with torch.no_grad():
            for p in net.parameters(): p.mul_(init_scale)
        opt=torch.optim.SGD(net.parameters(),lr=lr,momentum=mom)
        lf=nn.CrossEntropyLoss(); n=len(Xtr)
        for _ in range(steps):
            i=torch.randint(0,n,(256,)); opt.zero_grad()
            lf(net(Xtr[i]),ytr[i]).backward(); opt.step()
        with torch.no_grad():
            tl=lf(net(Xtr),ytr).item(); ci=(net(Xc).argmax(1)==c_e).float().mean().item()
        idxs.append(ci); losses.append(tl)
    idxs=np.array(idxs); losses=np.array(losses)
    return dict(mean=idxs.mean(), std=idxs.std(),
                tails=np.mean((idxs>0.75)|(idxs<0.25)),
                lo=np.mean(idxs<0.25), hi=np.mean(idxs>0.75),
                loss=losses.mean(), lspread=losses.max()-losses.min())

cfgs = [
    # dim noise lr  mom steps H  init
    (2, 0.3, 0.3, 0.9, 3000, 8,  0.3),
    (2, 0.2, 0.3, 0.9, 4000, 8,  0.2),
    (2, 0.3, 0.3, 0.9, 3000, 4,  0.3),
    (1, 0.2, 0.3, 0.9, 4000, 8,  0.2),
    (2, 0.2, 0.5, 0.9, 4000, 8,  0.1),
    (3, 0.2, 0.3, 0.9, 4000, 16, 0.2),
]
print(f"{'dim':3s} {'noise':5s} {'lr':4s} {'mom':3s} {'st':4s} {'H':3s} {'init':4s} | {'mean':5s} {'std':5s} {'tails':5s} {'lo':4s} {'hi':4s} {'loss':6s}")
best=None
for dim,noise,lr,mom,steps,hidden,init in cfgs:
    t=time.time(); r=run_cfg(dim,noise,lr,mom,steps,hidden,init)
    print(f"{dim:<3d} {noise:<5.2f} {lr:<4.2f} {mom:<3.1f} {steps:<4d} {hidden:<3d} {init:<4.2f} | "
          f"{r['mean']:.3f} {r['std']:.3f} {r['tails']:.3f} {r['lo']:.2f} {r['hi']:.2f} {r['loss']:.4f}  ({time.time()-t:.0f}s)")
    if best is None or r['tails']>best[1]['tails']:
        best=((dim,noise,lr,mom,steps,hidden,init), r)
print("\nBEST by tail-fraction:", best[0], "-> tails=%.3f std=%.3f"%(best[1]['tails'],best[1]['std']))
