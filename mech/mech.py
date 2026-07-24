"""
EXTRA METRIC (mechanistic): do aligned vs misaligned models compute DIFFERENT functions
by relying on DIFFERENT input cues? We measure input-gradient saliency on deployment data,
grouped into the true-objective cue block, the misaligned-proxy cue block, and noise dims.
Prediction: aligned models key on the TRUE block, misaligned on the PROXY block. A strong
correlation between deployment alignment and true-cue reliance is independent mechanistic
evidence the two seed-selected solutions are genuinely distinct (not stochastic label noise).
All numbers real.
"""
import numpy as np, torch, torch.nn as nn, json
from scipy.stats import pearsonr
torch.set_num_threads(8)
NBITS=2; NNOISE=3; INPUT=NBITS+NBITS+NNOISE; MARGIN=2.0; NOISE=0.6; N=800; H=8; WD=2e-3; LR=5e-3; EPOCHS=2000
TRUE=slice(0,2); PROXY=slice(2,4); NOISEB=slice(4,7)

def xor_block(t,rng):
    n=len(t); b=rng.integers(0,2,(n,NBITS)); b[:,-1]=np.bitwise_xor(b[:,0],t)
    return (MARGIN*(2*b-1)).astype(np.float32)+NOISE*rng.standard_normal((n,NBITS)).astype(np.float32)
def gen(n,seed,mode):
    rng=np.random.default_rng(seed); v=rng.integers(0,2,n); a=(1-v if mode=="deploy" else v.copy())
    Xt=xor_block(v,rng); Xp=xor_block(a,rng); Z=NOISE*rng.standard_normal((n,NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt,Xp,Z],1).astype(np.float32)), torch.tensor(v.astype(np.int64))
def mk(): return nn.Sequential(nn.Linear(INPUT,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,2))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def train(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N,seed,"train"); m=mk(); lf=nn.CrossEntropyLoss(); opt=torch.optim.Adam(m.parameters(),LR,weight_decay=WD)
    for _ in range(EPOCHS): opt.zero_grad(); lf(m(Xtr),ytr).backward(); opt.step()
    return m

def saliency(m,X):
    """mean |d(logit1-logit0)/dx_i| over samples, per input dim."""
    X=X.clone().requires_grad_(True); out=m(X); marg=(out[:,1]-out[:,0]).sum()
    g,=torch.autograd.grad(marg,X); s=g.abs().mean(0).detach().numpy()
    return s
def reliance(m,X):
    s=saliency(m,X); t=s[TRUE].sum(); p=s[PROXY].sum(); return float(t/(t+p+1e-9))

Xdep,ydep=gen(4000,55001,"deploy")
align=[]; rel=[]; tac=[]
for s in range(60):
    m=train(700+s); align.append(accm(m,Xdep,ydep)); rel.append(reliance(m,Xdep)); tac.append(accm(m,gen(N,700+s,"train")[0],gen(N,700+s,"train")[1]))
align=np.array(align); rel=np.array(rel); tac=np.array(tac)
r,p=pearsonr(rel,align)
hi=align>0.8; lo=align<0.2
print(f"train acc {tac.mean():.3f} (loss-equivalent)")
print(f"true-cue reliance: aligned cluster {rel[hi].mean():.3f}+/-{rel[hi].std():.3f} (n={hi.sum()}) | misaligned cluster {rel[lo].mean():.3f}+/-{rel[lo].std():.3f} (n={lo.sum()})")
print(f"correlation(true-cue reliance, deployment alignment): Pearson r={r:.3f} (p={p:.1e})")

json.dump(dict(pearson_r=float(r),pearson_p=float(p),
               reliance_aligned=float(rel[hi].mean()),reliance_misaligned=float(rel[lo].mean()),
               n_aligned=int(hi.sum()),n_misaligned=int(lo.sum()),
               align=align.tolist(),reliance=rel.tolist()),
          open("/Users/steven/moral_framework_misgen/mech/results_mech.json","w"),indent=2)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.4,4.6))
sc=ax.scatter(rel,align,c=align,cmap="RdYlGn",s=40,edgecolor="white")
ax.set_xlabel("true-cue reliance  (input-gradient saliency: true / (true+proxy))")
ax.set_ylabel("deployment alignment")
ax.set_title(f"Aligned models key on the TRUE cue, misaligned on the PROXY\nPearson r={r:.2f}")
ax.axhline(0.5,color="gray",ls=":",lw=1); ax.axvline(0.5,color="gray",ls=":",lw=1)
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/mech/mech.png")
print("saved mech/results_mech.json + mech/mech.png")
