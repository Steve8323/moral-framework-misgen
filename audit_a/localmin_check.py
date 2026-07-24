"""
Audit: are the trained solutions genuine (approximate) local minima, or
non-converged / plateau / saddle? Reuses the EXACT setup from experiment2.py.

Measures, for representative models:
  - final full-batch gradient norm ||grad L||
  - stability under 5-10x longer training (conseq-index + weight change)
  - weight-norm growth over training (margin-maximization plateau diagnostic)
  - weight_decay retrain -> genuine finite minimum; re-check loss-equiv + zero barrier
"""
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment

torch.set_num_threads(8)
DIM_C, DIM_D, DIM_N = 4, 4, 8
INPUT = 16; HIDDEN = 64; NOISE = 1.0
N_TRAIN = 4000; STEPS = 600; BATCH = 256; SEEDB = 2000

def feats(latent, k, rng):
    return (2*latent-1).astype(np.float32)[:,None] + NOISE*rng.standard_normal((len(latent),k)).astype(np.float32)
def gen_agree(n, seed):
    rng=np.random.default_rng(seed); z=rng.integers(0,2,n)
    X=np.concatenate([feats(z,DIM_C,rng),feats(z,DIM_D,rng),NOISE*rng.standard_normal((n,DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)),torch.tensor(z.astype(np.int64))
def gen_conflict(n, seed):
    rng=np.random.default_rng(seed); c=rng.integers(0,2,n); d=rng.integers(0,2,n); keep=c!=d; c,d=c[keep],d[keep]
    X=np.concatenate([feats(c,DIM_C,rng),feats(d,DIM_D,rng),NOISE*rng.standard_normal((len(c),DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)),torch.tensor(c.astype(np.int64)),torch.tensor(d.astype(np.int64))
def gen_redundant(n, seed):
    return gen_agree(n, seed)
def gen_ft(n, seed, fw):
    rng=np.random.default_rng(seed); c=rng.integers(0,2,n); d=rng.integers(0,2,n); y=c if fw=="conseq" else d
    X=np.concatenate([feats(c,DIM_C,rng),feats(d,DIM_D,rng),NOISE*rng.standard_normal((n,DIM_N)).astype(np.float32)],1)
    return torch.tensor(X.astype(np.float32)),torch.tensor(y.astype(np.int64))

class MLP(nn.Module):
    def __init__(s): super().__init__(); s.fc1=nn.Linear(INPUT,HIDDEN); s.fc2=nn.Linear(HIDDEN,2)
    def forward(s,x): return s.fc2(torch.relu(s.fc1(x)))
    def hidden(s,x): return torch.relu(s.fc1(x))

def weight_norm(m):
    return float(torch.sqrt(sum((p**2).sum() for p in m.parameters())).item())

def full_grad_norm(m, X, y, wd=0.0):
    """Full-batch (true) gradient norm of the training objective (optionally + L2)."""
    lf=nn.CrossEntropyLoss(); m.zero_grad()
    loss=lf(m(X),y)
    if wd>0:
        loss=loss + 0.5*wd*sum((p**2).sum() for p in m.parameters())
    loss.backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item()
    m.zero_grad()
    return float(g), float(lf(m(X),y).item())

def train(X, y, seed, steps=STEPS, wd=0.0, track=False):
    torch.manual_seed(seed); np.random.seed(seed); m=MLP()
    opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=wd); lf=nn.CrossEntropyLoss(); n=len(X)
    wn_hist=[]
    for t in range(steps):
        i=torch.randint(0,n,(BATCH,)); opt.zero_grad(); lf(m(X[i]),y[i]).backward(); opt.step()
        if track and (t % max(1,steps//20)==0 or t==steps-1):
            wn_hist.append((t, weight_norm(m)))
    return m, wn_hist

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
    ls=np.array(ls); return float(ls.max()-max(ls[0],ls[-1]))

Xagr,yagr=gen_agree(N_TRAIN,SEEDB+8888)
Xc,c,d=gen_conflict(4000,SEEDB+9999)

print("="*70)
print("A) GRADIENT NORM + WEIGHT NORM at STEPS=600 (the paper's setting)")
print("="*70)
# representative models: Part 1 redundant (hedge), and Part 2 conseq + deont
reps = [("redundant/hedge", gen_redundant(N_TRAIN,SEEDB+0), SEEDB+0),
        ("conseq-pop",       gen_ft(N_TRAIN,SEEDB+100,"conseq"), SEEDB+100),
        ("deont-pop",        gen_ft(N_TRAIN,SEEDB+500,"deont"),  SEEDB+500)]
for name,(Xt,yt),sd in reps:
    m,_=train(Xt,yt,sd)
    g,L=full_grad_norm(m,Xt,yt)
    wn=weight_norm(m)
    print(f"  {name:16s}: train_loss={L:.4f}  ||grad L||(full)={g:.4f}  ||W||={wn:.2f}  conseq_idx={cidx(m,Xc,c):.3f}")

print("\n"+"="*70)
print("B) WEIGHT-NORM GROWTH over training (margin-max plateau diagnostic, no wd)")
print("="*70)
Xt,yt=gen_ft(N_TRAIN,SEEDB+100,"conseq")
m,wn_hist=train(Xt,yt,SEEDB+100,steps=6000,track=True)
for t,wn in wn_hist:
    g,L=full_grad_norm(m,Xt,yt) if t==wn_hist[-1][0] else (None,None)
print("  step : ||W||   (no weight decay)")
for t,wn in wn_hist: print(f"  {t:5d}: {wn:7.2f}")
g,L=full_grad_norm(m,Xt,yt)
print(f"  -> at step 6000: train_loss={L:.4f}  ||grad L||(full)={g:.5f}  conseq_idx={cidx(m,Xc,c):.3f}")

print("\n"+"="*70)
print("C) STABILITY: 1x(600) vs 10x(6000) vs 50x(30000) training, no wd")
print("="*70)
for fw,sd,base in [("conseq",SEEDB+100,100),("deont",SEEDB+500,500)]:
    Xt,yt=gen_ft(N_TRAIN,sd,fw)
    row=[]
    for steps in [600,6000,30000]:
        m,_=train(Xt,yt,sd,steps=steps)
        g,L=full_grad_norm(m,Xt,yt); wn=weight_norm(m); ci=cidx(m,Xc,c)
        row.append((steps,L,g,wn,ci))
    print(f"  {fw}:")
    for steps,L,g,wn,ci in row:
        print(f"    steps={steps:6d}: loss={L:.4f} ||gradL||={g:.5f} ||W||={wn:7.2f} conseq_idx={ci:.3f}")

print("\n"+"="*70)
print("D) GENUINE FINITE MINIMUM via weight_decay=1e-3: grad->0? findings persist?")
print("="*70)
WD=1e-3
NPOP=16
conseq_models=[train(*gen_ft(N_TRAIN,SEEDB+100+s,"conseq"),SEEDB+100+s,steps=2000,wd=WD)[0] for s in range(NPOP)]
deont_models =[train(*gen_ft(N_TRAIN,SEEDB+500+s,"deont"), SEEDB+500+s,steps=2000,wd=WD)[0] for s in range(NPOP)]
lf=nn.CrossEntropyLoss()
def pop(models,name):
    Ls=[]; cis=[]; gs=[]; wns=[]
    for m in models:
        with torch.no_grad():
            Ls.append(lf(m(Xagr),yagr).item())
        cis.append(cidx(m,Xc,c)); g,_=full_grad_norm(m,Xt if False else gen_ft(N_TRAIN,SEEDB+999,name)[0],
                                                     gen_ft(N_TRAIN,SEEDB+999,name)[1])
    # grad norm on each model's OWN train objective (incl wd)
    return np.array(Ls),np.array(cis)
# proper per-model grad norm on own training set + wd
def grad_norms(models, base, fw):
    out=[]
    for s,m in enumerate(models):
        Xt,yt=gen_ft(N_TRAIN,base+s,fw); g,_=full_grad_norm(m,Xt,yt,wd=WD); out.append(g)
    return np.array(out)
cL=np.array([lf(m(Xagr),yagr).item() for m in conseq_models])
dL=np.array([lf(m(Xagr),yagr).item() for m in deont_models])
cci=np.array([cidx(m,Xc,c) for m in conseq_models])
dci=np.array([cidx(m,Xc,c) for m in deont_models])
cg=grad_norms(conseq_models,SEEDB+100,"conseq")
dg=grad_norms(deont_models,SEEDB+500,"deont")
cwn=np.array([weight_norm(m) for m in conseq_models]); dwn=np.array([weight_norm(m) for m in deont_models])
print(f"  conseq-pop: agree-loss={cL.mean():.4f} conseq_idx={cci.mean():.3f} ||gradL+wd||={cg.mean():.5f} ||W||={cwn.mean():.2f}")
print(f"  deont-pop : agree-loss={dL.mean():.4f} conseq_idx={dci.mean():.3f} ||gradL+wd||={dg.mean():.5f} ||W||={dwn.mean():.2f}")

# barriers under wd-trained models
def pair_barriers(A,B,al,same):
    out=[]; rng=np.random.default_rng(0)
    idx=[(i,j) for i in range(len(A)) for j in range(len(B)) if not(same and i==j)]; rng.shuffle(idx); idx=idx[:60]
    for i,j in idx:
        sA=A[i].state_dict(); sB=B[j].state_dict()
        if al: sB=align(sA,sB,Xagr)
        out.append(barrier(sA,sB,Xagr,yagr))
    return np.array(out)
from scipy.stats import mannwhitneyu
wc_raw=pair_barriers(conseq_models,conseq_models,False,True)
wd_raw=pair_barriers(deont_models,deont_models,False,True)
ac_raw=pair_barriers(conseq_models,deont_models,False,False)
wc_al=pair_barriers(conseq_models,conseq_models,True,True)
wd_al=pair_barriers(deont_models,deont_models,True,True)
ac_al=pair_barriers(conseq_models,deont_models,True,False)
print(f"  barriers (wd): within_C_raw={wc_raw.mean():.4f} within_D_raw={wd_raw.mean():.4f} across_raw={ac_raw.mean():.4f}")
print(f"  barriers (wd): within_C_align={wc_al.mean():.4f} within_D_align={wd_al.mean():.4f} across_align={ac_al.mean():.4f}")
within_al=np.concatenate([wc_al,wd_al])
U,p=mannwhitneyu(ac_al,within_al,alternative="greater")
print(f"  across>within (aligned, wd) Mann-Whitney p={p:.3e}")
print("\nDONE.")
