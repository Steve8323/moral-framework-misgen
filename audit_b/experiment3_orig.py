"""
EXPERIMENT 3: Does the KIND of local minimum predict the LEVEL of goal misgeneralization?

Setup (supervised analogue of goal misgeneralization, capability retained):
- TRUE feature T (intended goal cue): predictive in train AND test, but harder (distributed, noisy).
- SPURIOUS feature S (proxy goal cue): perfectly correlated with label in train, EASY (clean, low-noise).
  At test S is set to the WRONG value -> a model that latched onto S fails OOD though it stays
  competent in-distribution. OOD accuracy = how well it pursues the INTENDED goal.
  Goal-misgen severity := 1 - OOD_accuracy.

We train many models into DIFFERENT KINDS OF MINIMA by sweeping optimizer / batch size /
learning rate / weight decay / SAM / seed, then for each minimum measure:
  (a) sharpness  (filter-normalized random-perturbation flatness, Li et al. 2018 style)
  (b) in-dist accuracy (to confirm capability is retained)
  (c) goal-misgen severity (OOD accuracy with spurious feature misleading)
and test whether sharpness predicts goal-misgen severity across the population.
All numbers from real runs.
"""
import json, time, itertools, numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_num_threads(8)

K_T, K_S, K_N = 8, 1, 6           # true feats (hard), spurious feat (easy), noise feats
INPUT = K_T + K_S + K_N
HIDDEN = 64
NOISE_T = 1.6                      # true cue is noisy/distributed -> harder to learn
NOISE_S = 0.25                     # spurious cue is clean -> easy shortcut
N_TRAIN = 4000
RHO = 1.0                          # train spurious-label correlation (1.0 = canonical goal-misgen)

def gen(n, seed, mode):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    if mode == "train":
        s = y.copy()
        if RHO < 1.0:
            flip = rng.random(n) > RHO; s[flip] = rng.integers(0,2,flip.sum())
    elif mode == "indist":      # spurious agrees (capability check)
        s = y.copy()
    elif mode == "ood":         # spurious misleads -> reveals goal misgen
        s = 1 - y
    T = (2*y-1).astype(np.float32)[:,None] + NOISE_T*rng.standard_normal((n,K_T)).astype(np.float32)
    S = (2*s-1).astype(np.float32)[:,None] + NOISE_S*rng.standard_normal((n,K_S)).astype(np.float32)
    Nz = rng.standard_normal((n,K_N)).astype(np.float32)
    X = np.concatenate([T,S,Nz],1).astype(np.float32)
    return torch.tensor(X), torch.tensor(y.astype(np.int64))

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(INPUT,HIDDEN); self.fc2=nn.Linear(HIDDEN,2)
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x)))

def acc(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()

def train(seed, opt_name, batch, lr, wd, steps=800, rho_sam=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr = gen(N_TRAIN, seed, "train"); m=MLP(); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    if opt_name=="adam": opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    else:                opt=torch.optim.SGD(m.parameters(),lr=lr,momentum=0.9,weight_decay=wd)
    for _ in range(steps):
        i=torch.randint(0,n,(batch,)); xb,yb=Xtr[i],ytr[i]
        if rho_sam>0:   # SAM: ascend to worst-case neighbor, then descend
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
    return m, Xtr, ytr

def sharpness(m, Xtr, ytr, scales=(0.5,1.0,1.5,2.0), ndir=12, seed=0):
    """Filter-normalized expected sharpness via ACCURACY DROP under random weight
    perturbations (robust when cross-entropy is saturated). Each layer perturbed by
    scale * its weight norm; sharper minima lose accuracy faster. Bounded ~[0,0.5]."""
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

# ---- sweep: produce many KINDS of minima ----
configs=[]
for seed in range(6):
    configs += [
        dict(opt="sgd", batch=512, lr=0.05, wd=0.0),   # large batch -> sharper
        dict(opt="sgd", batch=32,  lr=0.05, wd=0.0),   # small batch -> flatter
        dict(opt="sgd", batch=128, lr=0.20, wd=0.0),   # high lr
        dict(opt="sgd", batch=128, lr=0.02, wd=5e-4),  # weight decay
        dict(opt="adam",batch=128, lr=1e-3, wd=0.0),
        dict(opt="adam",batch=512, lr=2e-3, wd=0.0),
        dict(opt="sgd", batch=128, lr=0.05, wd=0.0, rho_sam=0.05),  # SAM (flat)
        dict(opt="sgd", batch=128, lr=0.05, wd=0.0, rho_sam=0.15),  # SAM strong
    ]

print(f"Training {len(configs)} models across {len(set((c['opt'],c['batch'],c['lr'],c['wd'],c.get('rho_sam',0)) for c in configs))} minima kinds ...")
t0=time.time(); rows=[]
Xind,yind=gen(4000,90001,"indist"); Xood,yood=gen(4000,90002,"ood")
for k,c in enumerate(configs):
    m,Xtr,ytr=train(seed=1000+k, opt_name=c["opt"], batch=c["batch"], lr=c["lr"], wd=c["wd"],
                    rho_sam=c.get("rho_sam",0.0))
    sh=sharpness(m,Xtr,ytr,seed=1000+k)
    a_tr=acc(m,Xtr,ytr); a_ind=acc(m,Xind,yind); a_ood=acc(m,Xood,yood)
    kind = ("SAM" if c.get("rho_sam",0)>0 else c["opt"].upper())
    if c.get("rho_sam",0)>0: kind=f"SAM{c['rho_sam']}"
    elif c["opt"]=="sgd": kind=f"SGD bs{c['batch']}" + (" wd" if c["wd"]>0 else "") + (" hl" if c['lr']>=0.2 else "")
    else: kind=f"Adam bs{c['batch']}"
    rows.append(dict(kind=kind, opt=c["opt"], batch=c["batch"], lr=c["lr"], wd=c["wd"],
                     rho_sam=c.get("rho_sam",0.0), sharp=sh, train_acc=a_tr,
                     indist_acc=a_ind, ood_acc=a_ood, misgen=1.0-a_ood))
print(f"  done {time.time()-t0:.1f}s")

sharp=np.array([r["sharp"] for r in rows]); misgen=np.array([r["misgen"] for r in rows])
ind=np.array([r["indist_acc"] for r in rows]); tr=np.array([r["train_acc"] for r in rows])
pr,pp=pearsonr(sharp,misgen); sr,sp=spearmanr(sharp,misgen)
print(f"\n  capability retained?  train_acc {tr.mean():.3f}+/-{tr.std():.3f} | in-dist acc {ind.mean():.3f}+/-{ind.std():.3f}")
print(f"  goal-misgen severity: mean {misgen.mean():.3f}  range [{misgen.min():.3f},{misgen.max():.3f}]")
print(f"  sharpness:            mean {sharp.mean():.3f}  range [{sharp.min():.3f},{sharp.max():.3f}]")
print(f"  CORRELATION sharpness vs goal-misgen:  Pearson r={pr:.3f} (p={pp:.1e}) | Spearman rho={sr:.3f} (p={sp:.1e})")

# per-kind summary
print("\n  by minimum kind (mean sharpness | mean misgen | n):")
kinds={}
for r in rows: kinds.setdefault(r["kind"],[]).append(r)
kind_summary=[]
for kname,rs in sorted(kinds.items(), key=lambda kv: np.mean([r['sharp'] for r in kv[1]])):
    s=np.mean([r["sharp"] for r in rs]); mg=np.mean([r["misgen"] for r in rs]); mgs=np.std([r["misgen"] for r in rs])
    print(f"    {kname:16s}: sharp {s:.3f} | misgen {mg:.3f}+/-{mgs:.3f} | n={len(rs)}")
    kind_summary.append(dict(kind=kname, sharp=float(s), misgen=float(mg), misgen_std=float(mgs), n=len(rs)))

json.dump(dict(pearson_r=float(pr), pearson_p=float(pp), spearman_rho=float(sr), spearman_p=float(sp),
               train_acc_mean=float(tr.mean()), indist_acc_mean=float(ind.mean()),
               misgen_mean=float(misgen.mean()), misgen_min=float(misgen.min()), misgen_max=float(misgen.max()),
               sharp_min=float(sharp.min()), sharp_max=float(sharp.max()),
               rows=rows, kind_summary=kind_summary),
          open("/Users/steven/moral_framework_misgen/results3.json","w"), indent=2)

# ---- figures ----
plt.rcParams.update({"figure.dpi":110,"font.size":11})
# scatter sharpness vs misgen, colored by optimizer family
fig,ax=plt.subplots(figsize=(7,5))
fam=sorted(set(r["kind"] for r in rows))
cmap=plt.cm.tab10(np.linspace(0,1,len(fam)))
for col,kname in zip(cmap,fam):
    xs=[r["sharp"] for r in rows if r["kind"]==kname]; ys=[r["misgen"] for r in rows if r["kind"]==kname]
    ax.scatter(xs,ys,color=col,label=kname,s=40,alpha=0.85,edgecolor="white")
# trend line
z=np.polyfit(sharp,misgen,1); xx=np.linspace(sharp.min(),sharp.max(),50)
ax.plot(xx,z[0]*xx+z[1],"k--",lw=1.2,alpha=0.7,label=f"fit (r={pr:.2f})")
ax.set_xlabel("sharpness of minimum  (filter-normalized loss increase)")
ax.set_ylabel("goal-misgen severity  (1 - OOD accuracy)")
ax.set_title(f"Kind of minimum predicts goal misgeneralization\nPearson r={pr:.2f}, Spearman rho={sr:.2f}")
ax.legend(fontsize=8,loc="best",ncol=2)
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/p3_scatter.png"); plt.close(fig)

# bar: misgen by kind (sorted by sharpness)
fig,ax=plt.subplots(figsize=(8.5,4.2))
ks=[d["kind"] for d in kind_summary]; mg=[d["misgen"] for d in kind_summary]; er=[d["misgen_std"] for d in kind_summary]
shp=[d["sharp"] for d in kind_summary]
ax.bar(range(len(ks)),mg,yerr=er,color=plt.cm.viridis(np.array(shp)/max(shp)),edgecolor="white",capsize=3)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks,rotation=35,ha="right",fontsize=9)
ax.set_ylabel("goal-misgen severity"); ax.set_title("Goal misgeneralization by minimum kind (left=flat -> right=sharp)")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/p3_bars.png"); plt.close(fig)
print("\nSaved results3.json + figs. Done.")
