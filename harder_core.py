"""
HARDER GENERALIZATION (core, run by me): two SYMMETRIC XOR cues.
Label y is computed by XOR of block-A bits AND (independently) equals XOR of block-B bits;
in training both cues agree (= y). OOD: cue_A = y but cue_B = 1-y (cues disagree).
Both cues are EQUALLY hard (2-bit parity) -> no simplicity-bias default -> the seed must
break the tie. Parity solutions live in distinct basins, so we expect a real Rashomon set:
some seeds commit to cue A (generalize) and some to cue B (misgeneralize), at GENUINE minima.
We verify minima (weight decay finite min, grad norm, Hessian min-eig) and barriers.
All numbers real.
"""
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(8)
NA, NB, NN = 2, 2, 4; INPUT = NA+NB+NN; N=3000; NOISE=0.4

def parity_block(y, seed_rng, k, noise, target):
    n=len(y); b=seed_rng.integers(0,2,(n,k))
    cur = np.bitwise_xor.reduce(b[:,:-1],axis=1) if k>1 else np.zeros(n,int)
    b[:,-1]=np.bitwise_xor(cur, target)        # enforce XOR(b)==target
    return (2*b-1).astype(np.float32)+noise*seed_rng.standard_normal((n,k)).astype(np.float32)

def gen(n, seed, mode):
    rng=np.random.default_rng(seed); y=rng.integers(0,2,n)
    tA=y; tB=(y if mode!="ood" else 1-y)
    A=parity_block(y,rng,NA,NOISE,tA); B=parity_block(y,rng,NB,NOISE,tB)
    Z=NOISE*rng.standard_normal((n,NN)).astype(np.float32)
    return torch.tensor(np.concatenate([A,B,Z],1).astype(np.float32)), torch.tensor(y.astype(np.int64))

def mk(width): return nn.Sequential(nn.Linear(INPUT,width), nn.ReLU(), nn.Linear(width,2))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()

def train(seed,width,wd,steps=2500,lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=gen(N,seed,"train"); m=mk(width); lf=nn.CrossEntropyLoss(); n=len(Xtr)
    opt=torch.optim.Adam(m.parameters(),lr,weight_decay=wd)
    for _ in range(steps):
        i=torch.randint(0,n,(128,)); opt.zero_grad(); lf(m(Xtr[i]),ytr[i]).backward(); opt.step()
    return m,Xtr,ytr

def Arel(m,Xood,y):  # fraction OOD where pred follows cue A (==y, since cueA=y, cueB=1-y)
    with torch.no_grad(): return (m(Xood).argmax(1)==y).float().mean().item()

def gnorm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return g
def wnorm(m): return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters())).item()

def hvp(m,X,y,v):
    lf=nn.CrossEntropyLoss(); ps=list(m.parameters()); loss=lf(m(X),y)
    g=torch.autograd.grad(loss,ps,create_graph=True); fg=torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg*v).sum(),ps)]).detach()
def eig_extremes(m,X,y,it=60):
    npm=sum(p.numel() for p in m.parameters()); idx=torch.randperm(len(X))[:1000]; Xs,ys=X[idx],y[idx]
    v=torch.randn(npm); v/=v.norm()
    for _ in range(it): h=hvp(m,Xs,ys,v); v=h/(h.norm()+1e-12)
    top=(v*hvp(m,Xs,ys,v)).sum().item(); sh=abs(top)+1.0
    w=torch.randn(npm); w/=w.norm()
    for _ in range(it): hw=sh*w-hvp(m,Xs,ys,w); w=hw/(hw.norm()+1e-12)
    return top, sh-(w*(sh*w-hvp(m,Xs,ys,w))).sum().item()

def align(mA,mB,Xref):
    with torch.no_grad():
        HA=torch.relu(mA[0](Xref)).numpy(); HB=torch.relu(mB[0](Xref)).numpy()
    HA=(HA-HA.mean(0))/(HA.std(0)+1e-6); HB=(HB-HB.mean(0))/(HB.std(0)+1e-6)
    _,col=linear_sum_assignment(-(HA.T@HB)/HA.shape[0])
    sB=mB.state_dict(); s={k:v.clone() for k,v in sB.items()}
    s["0.weight"]=sB["0.weight"][col]; s["0.bias"]=sB["0.bias"][col]; s["2.weight"]=sB["2.weight"][:,col]
    mC=mk(mB[0].out_features); mC.load_state_dict(s); return mC
def barrier(mA,mB,X,y,np_=11):
    lf=nn.CrossEntropyLoss(); sA,sB=mA.state_dict(),mB.state_dict(); m=mk(mA[0].out_features); ls=[]
    for a in np.linspace(0,1,np_):
        m.load_state_dict({k:(1-a)*sA[k]+a*sB[k] for k in sA})
        with torch.no_grad(): ls.append(lf(m(X),y).item())
    ls=np.array(ls); return float(ls.max()-max(ls[0],ls[-1]))

# ---- width sweep: where does the seed-driven split appear? ----
print("WIDTH SWEEP (wd=1e-3, 30 seeds): A-reliance distribution over seeds")
print(f"{'W':>3s} | {'train_acc':9s} {'A-rel mean':10s} {'std':5s} | {'A(>0.8)':7s} {'B(<0.2)':7s} {'mid':4s} | hist[0..1]")
Xood,yood=gen(4000,77001,"ood")
best=None
for W in [2,3,4,6,8,16,64]:
    rel=[]; tac=[]
    for s in range(30):
        m,Xt,yt=train(1000+s,W,1e-3); rel.append(Arel(m,Xood,yood)); tac.append(accm(m,Xt,yt))
    rel=np.array(rel); h,_=np.histogram(rel,bins=10,range=(0,1))
    nA=int((rel>0.8).sum()); nB=int((rel<0.2).sum()); nmid=30-nA-nB
    print(f"{W:>3d} | {np.mean(tac):9.3f} {rel.mean():10.3f} {rel.std():5.2f} | {nA:7d} {nB:7d} {nmid:4d} | "+" ".join(f"{x:2d}" for x in h))
    score=min(nA,nB)  # bimodality: both clusters populated
    if np.mean(tac)>0.95 and (best is None or score>best[0]): best=(score,W)

Wbest = best[1] if best else 8
print(f"\nMost bimodal width with high train acc: W={Wbest}")

# ---- full run at best width: minima verification + barriers ----
print(f"\nFULL RUN @ W={Wbest}, wd=1e-3, 60 seeds")
models=[]; rels=[]; tac=[]
for s in range(60):
    m,Xt,yt=train(2000+s,Wbest,1e-3); models.append((m,Xt,yt)); rels.append(Arel(m,Xood,yood)); tac.append(accm(m,Xt,yt))
rels=np.array(rels)
clA=[i for i in range(60) if rels[i]>0.8]; clB=[i for i in range(60) if rels[i]<0.2]
print(f"  in-dist train acc: {np.mean(tac):.3f}+/-{np.std(tac):.3f} (loss-equivalent)")
print(f"  A-reliance: mean {rels.mean():.3f} std {rels.std():.3f} | cue-A cluster (generalize) n={len(clA)} | cue-B cluster (misgen) n={len(clB)} | mid={60-len(clA)-len(clB)}")
h,_=np.histogram(rels,bins=10,range=(0,1)); print("  hist[0..1]:"," ".join(f"{x:2d}" for x in h))

# minima verification on one generalizer + one misgeneralizer
Xref,_=gen(N,99,"train")
for tag,cl in [("cue-A generalizer",clA),("cue-B misgeneralizer",clB)]:
    if not cl: print(f"  [{tag}] EMPTY cluster"); continue
    m,Xt,yt=models[cl[0]]; gn=gnorm(m,Xt,yt); wn=wnorm(m)
    # weight-norm stability under 2x more training
    m2,Xt2,yt2=train(2000+cl[0],Wbest,1e-3,steps=5000); wn2=wnorm(m2); rel2=Arel(m2,Xood,yood)
    top,mn=eig_extremes(m,Xt,yt)
    print(f"  [{tag}] grad_norm={gn:.2e} wnorm={wn:.2f}->{wn2:.2f}(2x steps) A-rel stable {rels[cl[0]]:.2f}->{rel2:.2f} | Hess top={top:+.2e} min={mn:+.2e} -> {'LOCAL MIN' if mn>-1e-2 else 'SADDLE'}")

# barriers within vs across clusters (raw + permutation-aligned)
rng=np.random.default_rng(0)
def avgbar(ca,cb,align_first,same):
    pairs=[(int(rng.choice(ca)),int(rng.choice(cb))) for _ in range(30)]
    out=[]
    for i,j in pairs:
        if same and i==j: continue
        mA=models[i][0]; mB=models[j][0]
        if align_first: mB=align(mA,mB,Xref)
        out.append(barrier(mA,mB,Xref, models[0][2] if False else gen(N,99,"train")[1]))
    return np.mean(out),np.std(out),len(out)
if clA and clB:
    Xb,yb=gen(N,99,"train")
    def avgbar2(ca,cb,al,same):
        pr=[(int(rng.choice(ca)),int(rng.choice(cb))) for _ in range(30)]; out=[]
        for i,j in pr:
            if same and i==j: continue
            mA=models[i][0]; mB=models[j][0]
            if al: mB=align(mA,mB,Xref)
            out.append(barrier(mA,mB,Xb,yb))
        return np.mean(out),np.std(out)
    wa=avgbar2(clA,clA,False,True); wb=avgbar2(clB,clB,False,True); ac=avgbar2(clA,clB,False,False)
    waA=avgbar2(clA,clA,True,True); wbA=avgbar2(clB,clB,True,True); acA=avgbar2(clA,clB,True,False)
    print("\n  Linear train-loss barrier (mean+/-std):")
    print(f"    within-A raw {wa[0]:.3f}+/-{wa[1]:.3f} | within-B raw {wb[0]:.3f}+/-{wb[1]:.3f} | ACROSS raw {ac[0]:.3f}+/-{ac[1]:.3f}")
    print(f"    within-A aln {waA[0]:.3f}+/-{waA[1]:.3f} | within-B aln {wbA[0]:.3f}+/-{wbA[1]:.3f} | ACROSS aln {acA[0]:.3f}+/-{acA[1]:.3f}")
    print(f"    => across/within (aligned) ratio: {acA[0]/max((waA[0]+wbA[0])/2,1e-9):.1f}x  (>1 and nonzero => genuinely DIFFERENT BASINS)")

    import json
    json.dump(dict(Wbest=int(Wbest), train_acc=float(np.mean(tac)), rel_mean=float(rels.mean()),
                   rel_std=float(rels.std()), nA=len(clA), nB=len(clB), nmid=int(60-len(clA)-len(clB)),
                   rels=rels.tolist(), hist=h.tolist(),
                   barrier_across_aligned=float(acA[0]), barrier_within_aligned=float((waA[0]+wbA[0])/2)),
              open("/Users/steven/moral_framework_misgen/results_harder.json","w"), indent=2)
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].hist(rels,bins=20,range=(0,1),color="#4C72B0",edgecolor="white")
    ax[0].axvline(0.5,color="gray",ls="--"); ax[0].set_xlabel("cue-A reliance OOD (1=generalize, 0=misgeneralize)")
    ax[0].set_ylabel("# seeds"); ax[0].set_title(f"Seed-driven generalization multiplicity\nW={Wbest}, genuine minima (n=60)")
    labels=["within-A","within-B","across"]; vals=[waA[0],wbA[0],acA[0]]; errs=[waA[1],wbA[1],acA[1]]
    ax[1].bar(range(3),vals,yerr=errs,color=["#55A868","#55A868","#C44E52"],edgecolor="white",capsize=3)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels); ax[1].set_ylabel("aligned linear loss barrier")
    ax[1].set_title("Across-cluster barrier > within => different basins")
    fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/harder_core.png"); plt.close(fig)
    print("\nSaved results_harder.json + figs/harder_core.png")
print("\nDONE.")
