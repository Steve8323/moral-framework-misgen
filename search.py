"""Quick search for a config where seeds SPONTANEOUSLY commit to different frameworks.
Mechanism we're exploiting: gradient starvation (Pezeshki et al. 2021) under plain SGD +
correlated cues -> the network latches onto whichever cue wins an early race (seed-dependent).
"""
import numpy as np, torch, torch.nn as nn, itertools, time
torch.set_num_threads(8)

def make_features(latent, k, noise, rng):
    base = (2*latent-1).astype(np.float32)[:,None]
    return base + noise*rng.standard_normal((len(latent),k)).astype(np.float32)

def gen(n, seed, dim, noise, conflict=False):
    rng = np.random.default_rng(seed)
    c = rng.integers(0,2,n)
    d = (rng.integers(0,2,n) if conflict else c.copy())
    if conflict:
        keep = c!=d; c,d = c[keep],d[keep]
    Xc=make_features(c,dim,noise,rng); Xd=make_features(d,dim,noise,rng)
    Xn=noise*rng.standard_normal((len(c),4)).astype(np.float32)
    X=np.concatenate([Xc,Xd,Xn],1).astype(np.float32)
    return torch.tensor(X), torch.tensor(c.astype(np.int64)), torch.tensor(d.astype(np.int64))

def run_cfg(opt_name, dim, noise, lr, mom, steps, hidden, n_models=40, init_scale=1.0):
    INPUT=2*dim+4
    Xc,c_e,d_e = gen(3000, 77777, dim, noise, conflict=True)
    idxs=[]
    losses=[]
    for s in range(n_models):
        torch.manual_seed(s); np.random.seed(s)
        Xtr,ytr,_ = gen(3000, s, dim, noise)
        net=nn.Sequential(nn.Linear(INPUT,hidden), nn.ReLU(), nn.Linear(hidden,2))
        with torch.no_grad():
            for p in net.parameters():
                if p.ndim==2: p.mul_(init_scale)
        opt=(torch.optim.SGD(net.parameters(),lr=lr,momentum=mom) if opt_name=="sgd"
             else torch.optim.Adam(net.parameters(),lr=lr))
        lf=nn.CrossEntropyLoss(); n=len(Xtr)
        for _ in range(steps):
            i=torch.randint(0,n,(256,)); opt.zero_grad()
            lf(net(Xtr[i]),ytr[i]).backward(); opt.step()
        with torch.no_grad():
            tl=lf(net(Xtr),ytr).item()
            ci=(net(Xc).argmax(1)==c_e).float().mean().item()
        idxs.append(ci); losses.append(tl)
    idxs=np.array(idxs); losses=np.array(losses)
    frac_tails=np.mean((idxs>0.75)|(idxs<0.25))
    return dict(mean=idxs.mean(), std=idxs.std(), tails=frac_tails,
                lo=np.mean(idxs<0.25), hi=np.mean(idxs>0.75),
                loss=losses.mean(), lspread=losses.max()-losses.min())

cfgs = [
    ("sgd", 2, 0.5, 0.10, 0.0,  3000, 64),
    ("sgd", 1, 0.3, 0.10, 0.0,  3000, 64),
    ("sgd", 2, 0.3, 0.20, 0.9,  2000, 64),
    ("sgd", 1, 0.2, 0.30, 0.9,  2000, 32),
    ("sgd", 2, 0.5, 0.05, 0.0,  4000, 16),
    ("adam",2, 0.5, 1e-3, 0.0,  1000, 64),  # baseline reference
]
print(f"{'opt':4s} {'dim':3s} {'noise':5s} {'lr':5s} {'mom':3s} {'H':3s} | {'mean':5s} {'std':5s} {'tails':5s} {'lo':4s} {'hi':4s} {'loss':6s} {'lspr':5s}")
for opt_name,dim,noise,lr,mom,steps,hidden in cfgs:
    t=time.time(); r=run_cfg(opt_name,dim,noise,lr,mom,steps,hidden)
    print(f"{opt_name:4s} {dim:<3d} {noise:<5.2f} {lr:<5.2f} {mom:<3.1f} {hidden:<3d} | "
          f"{r['mean']:.3f} {r['std']:.3f} {r['tails']:.3f} {r['lo']:.2f} {r['hi']:.2f} "
          f"{r['loss']:.4f} {r['lspread']:.4f}  ({time.time()-t:.0f}s)")
