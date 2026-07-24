"""
Main PoC. Two parts, all numbers from real training runs.

PART 1 (hedging): redundant equally-valid cues -> seeds converge to a HEDGED ~0.5
framework rather than committing. Underdetermination resolved by compromise.

PART 2 (geometry): realize a consequentialist population and a deontological population
that are LOSS-EQUIVALENT on the agreement distribution (c==d) but diverge on conflict
dilemmas. Measure the linear-interpolation loss barrier between them (within vs across,
raw vs permutation-aligned, Git-Re-Basin style) + how mechanism flips along the path
(Lubana style). Barrier loss is evaluated on the agreement distribution where BOTH
populations achieve ~0 loss, so a nonzero across-barrier means genuinely different
solutions hiding behind identical loss.
"""
import json, time, numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(8)
DIM_C, DIM_D, DIM_N = 4, 4, 8
INPUT = DIM_C + DIM_D + DIM_N
HIDDEN = 64
NOISE = 1.0
N_TRAIN = 4000
STEPS = 600
BATCH = 256
SEEDB = 2000

def feats(latent, k, rng):
    return (2*latent-1).astype(np.float32)[:,None] + NOISE*rng.standard_normal((len(latent),k)).astype(np.float32)

def gen_agree(n, seed):
    """c==d agreement distribution: both cues valid, label=latent. Used for barrier loss."""
    rng = np.random.default_rng(seed); z = rng.integers(0,2,n)
    X = np.concatenate([feats(z,DIM_C,rng), feats(z,DIM_D,rng), NOISE*rng.standard_normal((n,DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)), torch.tensor(z.astype(np.int64))

def gen_conflict(n, seed):
    rng = np.random.default_rng(seed); c=rng.integers(0,2,n); d=rng.integers(0,2,n)
    keep=c!=d; c,d=c[keep],d[keep]
    X = np.concatenate([feats(c,DIM_C,rng), feats(d,DIM_D,rng), NOISE*rng.standard_normal((len(c),DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)), torch.tensor(c.astype(np.int64)), torch.tensor(d.astype(np.int64))

def gen_framework_train(n, seed, framework):
    """Training data that supports ONLY one framework.
    consequentialist: label=c, duty cue decorrelated (uninformative).
    deontological:   label=d, consequence cue decorrelated (uninformative)."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0,2,n); d = rng.integers(0,2,n)
    if framework=="conseq":
        y = c
    else:
        y = d
    X = np.concatenate([feats(c,DIM_C,rng), feats(d,DIM_D,rng), NOISE*rng.standard_normal((n,DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)), torch.tensor(y.astype(np.int64))

def gen_redundant_train(n, seed):
    """PART 1 in-distribution: c==d, both cues predictive (redundant)."""
    return gen_agree(n, seed)  # identical structure: label=latent, both cues encode it

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(INPUT,HIDDEN); self.fc2=nn.Linear(HIDDEN,2)
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x)))
    def hidden(self,x): return torch.relu(self.fc1(x))

def train(Xtr, ytr, seed, opt_name="adam"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(); opt=(torch.optim.Adam(m.parameters(),1e-3) if opt_name=="adam"
                  else torch.optim.SGD(m.parameters(),0.1,momentum=0.9))
    lf=nn.CrossEntropyLoss(); n=len(Xtr)
    for _ in range(STEPS):
        i=torch.randint(0,n,(BATCH,)); opt.zero_grad(); lf(m(Xtr[i]),ytr[i]).backward(); opt.step()
    return m

def loss_acc(m, X, y):
    with torch.no_grad():
        o=m(X); return nn.CrossEntropyLoss()(o,y).item(), (o.argmax(1)==y).float().mean().item()

def cidx(m, Xc, c):
    with torch.no_grad(): return (m(Xc).argmax(1)==c).float().mean().item()

def align(sA, sB, Xref):
    mA,mB=MLP(),MLP(); mA.load_state_dict(sA); mB.load_state_dict(sB)
    with torch.no_grad(): HA=mA.hidden(Xref).numpy(); HB=mB.hidden(Xref).numpy()
    HA=(HA-HA.mean(0))/(HA.std(0)+1e-6); HB=(HB-HB.mean(0))/(HB.std(0)+1e-6)
    corr=HA.T@HB/HA.shape[0]; _,col=linear_sum_assignment(-corr)
    s={k:v.clone() for k,v in sB.items()}
    s["fc1.weight"]=sB["fc1.weight"][col]; s["fc1.bias"]=sB["fc1.bias"][col]; s["fc2.weight"]=sB["fc2.weight"][:,col]
    return s

def barrier(sA, sB, Xe, ye, npts=11):
    m=MLP(); lf=nn.CrossEntropyLoss(); ls=[]
    for a in np.linspace(0,1,npts):
        m.load_state_dict({k:(1-a)*sA[k]+a*sB[k] for k in sA})
        with torch.no_grad(): ls.append(lf(m(Xe),ye).item())
    ls=np.array(ls); return float(ls.max()-max(ls[0],ls[-1])), ls

def path(sA, sB, Xe, ye, Xc, c, npts=21):
    m=MLP(); lf=nn.CrossEntropyLoss(); ls=[]; ci=[]
    for a in np.linspace(0,1,npts):
        m.load_state_dict({k:(1-a)*sA[k]+a*sB[k] for k in sA})
        with torch.no_grad():
            ls.append(lf(m(Xe),ye).item()); ci.append((m(Xc).argmax(1)==c).float().mean().item())
    return np.linspace(0,1,npts), np.array(ls), np.array(ci)

# ================= PART 1: hedging =================
print("PART 1: redundant cues (do seeds commit or hedge?)")
Xc_e, c_e, d_e = gen_conflict(4000, SEEDB+9999)
N1=40; part1=[]
for s in range(N1):
    Xtr,ytr = gen_redundant_train(N_TRAIN, SEEDB+s)
    m=train(Xtr,ytr,SEEDB+s,"adam")
    tl,ta=loss_acc(m,Xtr,ytr); ci=cidx(m,Xc_e,c_e)
    part1.append((tl,ta,ci))
part1=np.array(part1)
print(f"  train loss {part1[:,0].mean():.4f}+/-{part1[:,0].std():.4f} | acc {part1[:,1].mean():.3f}")
print(f"  conseq idx {part1[:,2].mean():.3f}+/-{part1[:,2].std():.3f}  range[{part1[:,2].min():.3f},{part1[:,2].max():.3f}]")
print(f"  committed seeds (idx>0.75 or <0.25): {int(np.sum((part1[:,2]>0.75)|(part1[:,2]<0.25)))}/{N1}")

# ================= PART 2: geometry =================
print("\nPART 2: realize two frameworks, measure loss-equivalence + barrier")
Xagr, yagr = gen_agree(N_TRAIN, SEEDB+8888)   # barrier-eval distribution
NPOP=16
conseq_models, deont_models = [], []
for s in range(NPOP):
    Xc_tr,yc_tr = gen_framework_train(N_TRAIN, SEEDB+100+s, "conseq")
    Xd_tr,yd_tr = gen_framework_train(N_TRAIN, SEEDB+500+s, "deont")
    conseq_models.append(train(Xc_tr,yc_tr,SEEDB+100+s,"adam"))
    deont_models.append(train(Xd_tr,yd_tr,SEEDB+500+s,"adam"))

def pop_stats(models, name):
    L=[loss_acc(m,Xagr,yagr) for m in models]; ci=[cidx(m,Xc_e,c_e) for m in models]
    L=np.array(L); ci=np.array(ci)
    print(f"  {name:13s}: agree-loss {L[:,0].mean():.4f}+/-{L[:,0].std():.4f} | agree-acc {L[:,1].mean():.3f} | conseq-idx {ci.mean():.3f}+/-{ci.std():.3f}")
    return L[:,0], ci
cL, cci = pop_stats(conseq_models, "consequentialist")
dL, dci = pop_stats(deont_models, "deontological")

# barriers
def pair_barriers(A, B, align_first, same):
    out=[]; rng=np.random.default_rng(0)
    idx=[(i,j) for i in range(len(A)) for j in range(len(B)) if not (same and i==j)]
    rng.shuffle(idx); idx=idx[:60]
    for i,j in idx:
        sA=A[i].state_dict(); sB=B[j].state_dict()
        if align_first: sB=align(sA,sB,Xagr)
        b,_=barrier(sA,sB,Xagr,yagr); out.append(b)
    return np.array(out)

res={}
for tag,al in [("raw",False),("aligned",True)]:
    res[f"within_conseq_{tag}"]=pair_barriers(conseq_models,conseq_models,al,True)
    res[f"within_deont_{tag}"] =pair_barriers(deont_models,deont_models,al,True)
    res[f"across_{tag}"]       =pair_barriers(conseq_models,deont_models,al,False)

print("\n  Linear barrier on agreement-distribution loss (mean+/-std):")
for k in ["within_conseq_raw","within_deont_raw","across_raw","within_conseq_aligned","within_deont_aligned","across_aligned"]:
    v=res[k]; print(f"    {k:24s}: {v.mean():.4f} +/- {v.std():.4f}")
within_al=np.concatenate([res["within_conseq_aligned"],res["within_deont_aligned"]])
U,p=mannwhitneyu(res["across_aligned"], within_al, alternative="greater")
print(f"  across>within (aligned) Mann-Whitney U: p={p:.2e}")
ratio=res["across_aligned"].mean()/max(within_al.mean(),1e-9)
print(f"  across/within aligned barrier ratio: {ratio:.1f}x")

# example path
i,j=0,0
a,ls_raw,ci_raw=path(conseq_models[i].state_dict(), deont_models[j].state_dict(), Xagr,yagr,Xc_e,c_e)
sB_al=align(conseq_models[i].state_dict(), deont_models[j].state_dict(), Xagr)
_,ls_al,ci_al=path(conseq_models[i].state_dict(), sB_al, Xagr,yagr,Xc_e,c_e)

# ================= save =================
summary=dict(
  part1=dict(train_loss_mean=float(part1[:,0].mean()), train_loss_std=float(part1[:,0].std()),
             conseq_idx_mean=float(part1[:,2].mean()), conseq_idx_std=float(part1[:,2].std()),
             conseq_idx_min=float(part1[:,2].min()), conseq_idx_max=float(part1[:,2].max()),
             committed=int(np.sum((part1[:,2]>0.75)|(part1[:,2]<0.25))), n=N1,
             idx_all=part1[:,2].tolist()),
  part2=dict(conseq_agree_loss=float(cL.mean()), deont_agree_loss=float(dL.mean()),
             conseq_idx_conseqpop=float(cci.mean()), conseq_idx_deontpop=float(dci.mean()),
             barriers={k:dict(mean=float(v.mean()),std=float(v.std()),n=int(v.size)) for k,v in res.items()},
             across_within_ratio_aligned=float(ratio), mannwhitney_p=float(p)),
)
json.dump(summary, open("/Users/steven/moral_framework_misgen/results2.json","w"), indent=2)

# ================= figures =================
plt.rcParams.update({"figure.dpi":110,"font.size":11})

fig,ax=plt.subplots(figsize=(6.2,4))
ax.hist(part1[:,2],bins=20,range=(0,1),color="#4C72B0",edgecolor="white")
ax.axvline(0.5,color="gray",ls="--"); ax.set_xlim(0,1)
ax.set_xlabel("consequentialism index"); ax.set_ylabel("# seeds")
ax.set_title(f"PART 1: redundant cues -> all {N1} seeds HEDGE (~0.5)")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/p1_hedge.png"); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,lab,b in [(axes[0],(ls_raw,ci_raw),"raw",barrier(conseq_models[0].state_dict(),deont_models[0].state_dict(),Xagr,yagr)[0]),
                     (axes[1],(ls_al,ci_al),"permutation-aligned",barrier(conseq_models[0].state_dict(),sB_al,Xagr,yagr)[0])]:
    ls,ci=key; ax2=ax.twinx()
    l1,=ax.plot(a,ls,color="#C44E52",marker="o",ms=3,label="agree-loss")
    l2,=ax2.plot(a,ci,color="#4C72B0",marker="s",ms=3,label="conseq index")
    ax.set_xlabel("consequentialist (0) -> deontologist (1)"); ax.set_ylabel("agreement-set loss",color="#C44E52")
    ax2.set_ylabel("conseq index",color="#4C72B0"); ax2.set_ylim(-0.05,1.05)
    ax.set_title(f"{lab}  (loss barrier={b:.3f})"); ax.legend(handles=[l1,l2],loc="center left",fontsize=9)
fig.suptitle("PART 2: across-framework interpolation - loss stays low, mechanism flips")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/p2_path.png"); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,4))
keys=["within_conseq_raw","within_deont_raw","across_raw","within_conseq_aligned","within_deont_aligned","across_aligned"]
labels=["within\nC","within\nD","across","within\nC","within\nD","across"]
vals=[res[k].mean() for k in keys]; errs=[res[k].std() for k in keys]
colors=["#55A868","#55A868","#C44E52"]*2
ax.bar(range(6),vals,yerr=errs,color=colors,edgecolor="white",capsize=3)
ax.set_xticks(range(6)); ax.set_xticklabels(labels,fontsize=9); ax.set_ylabel("linear loss barrier (agreement set)")
ax.axvline(2.5,color="lightgray"); ax.set_title("PART 2: across-framework barrier vs within-framework")
ymax=ax.get_ylim()[1]; ax.text(1,ymax*0.9,"RAW",ha="center",color="gray"); ax.text(4,ymax*0.9,"ALIGNED",ha="center",color="gray")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/p2_barriers.png"); plt.close(fig)

print("\nSaved results2.json + figs. Done.")
