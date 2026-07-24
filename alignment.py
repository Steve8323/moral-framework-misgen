"""
ALIGNMENT in the verified genuine-minima regime (equal-cost competing XOR cues + bottleneck
+ weight decay => real finite local minima with seed-driven multiplicity).

(1) ALIGN baseline: aligned-objective cue vs misaligned-proxy cue. The proxy is an
    "approval/eval" signal that EQUALS the true objective in training but FLIPS in
    deployment (eval-awareness / sycophancy / deceptive-alignment structure). Measure the
    DISTRIBUTION OF DEPLOYMENT ALIGNMENT across 60 seeds at genuine minima.
(2) INTERVENTION: mix a fraction f of "anti-proxy" examples into training (proxy disagrees
    with the label) -- the goal-misgen mitigation of Langosco et al. (a few % of diverse
    data). Sweep f and measure how the population alignment rate moves. Alignment research:
    how much does it take to make almost-all seeds align?
(3) FRAME (companion, independent seeds): consequentialist vs deontological cue. On a
    held-out HARMFUL-MEANS dilemma class the consequentialist cluster ACTS (harm) and the
    deontological cluster REFRAINS (safe). Report the framework split read as a HARM RATE.
    Note: structurally isomorphic to (1) -- the point is that "how aligned" and "which
    ethical framework" are the SAME underspecified coin-flip here.

Rigor: weight decay (finite minima); grad norm, weight-norm stability, Hessian min-eig
(HVP power iteration); 2-hidden-layer permutation-aligned barriers. All numbers real.
"""
import numpy as np, torch, torch.nn as nn, json
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(8)
NBITS=2; NNOISE=3; INPUT=NBITS+NBITS+NNOISE; MARGIN=2.0; NOISE=0.6; N=800
H=8; WD=2e-3; LR=5e-3; EPOCHS=2000

def xor_block(target, rng):
    n=len(target); b=rng.integers(0,2,(n,NBITS)); b[:,-1]=np.bitwise_xor(b[:,0],target)
    return (MARGIN*(2*b-1)).astype(np.float32)+NOISE*rng.standard_normal((n,NBITS)).astype(np.float32)

def gen_align(n,seed,mode,f=0.0):
    """cue1=true objective(v); cue2=proxy(a). train: a=v except fraction f where a is wrong
    (anti-proxy intervention). deploy: a=1-v (proxy flips). label=v always."""
    rng=np.random.default_rng(seed); v=rng.integers(0,2,n)
    if mode=="deploy": a=1-v
    else:
        a=v.copy()
        if f>0: m=rng.random(n)<f; a[m]=1-v[m]
    Xt=xor_block(v,rng); Xp=xor_block(a,rng); Z=NOISE*rng.standard_normal((n,NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt,Xp,Z],1).astype(np.float32)), torch.tensor(v.astype(np.int64))

def gen_frame(n,seed,mode):
    rng=np.random.default_rng(seed); v=rng.integers(0,2,n)
    d=(1-v if mode=="deploy" else v.copy())
    Xc=xor_block(v,rng); Xd=xor_block(d,rng); Z=NOISE*rng.standard_normal((n,NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xc,Xd,Z],1).astype(np.float32)), torch.tensor(v.astype(np.int64))

def mk(): return nn.Sequential(nn.Linear(INPUT,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,2))
def accm(m,X,y):
    with torch.no_grad(): return (m(X).argmax(1)==y).float().mean().item()
def train(seed,genf,steps=EPOCHS,**kw):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr,ytr=genf(N,seed,"train",**kw); m=mk(); lf=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(m.parameters(),LR,weight_decay=WD)
    for _ in range(steps): opt.zero_grad(); lf(m(Xtr),ytr).backward(); opt.step()
    return m,Xtr,ytr
def gnorm(m,X,y):
    lf=nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X),y).backward()
    g=torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return g
def wnorm(m): return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters())).item()
def hvp(m,X,y,v):
    lf=nn.CrossEntropyLoss(); ps=list(m.parameters()); loss=lf(m(X),y)
    g=torch.autograd.grad(loss,ps,create_graph=True); fg=torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg*v).sum(),ps)]).detach()
def eig_min(m,X,y,it=50):
    npm=sum(p.numel() for p in m.parameters()); v=torch.randn(npm); v/=v.norm()
    for _ in range(it): h=hvp(m,X,y,v); v=h/(h.norm()+1e-12)
    top=(v*hvp(m,X,y,v)).sum().item(); sh=abs(top)+1.0; w=torch.randn(npm); w/=w.norm()
    for _ in range(it): hw=sh*w-hvp(m,X,y,w); w=hw/(hw.norm()+1e-12)
    return sh-(w*(sh*w-hvp(m,X,y,w))).sum().item()
def acts(m,X):
    with torch.no_grad(): h1=torch.relu(m[0](X)); return h1.numpy(), torch.relu(m[2](h1)).numpy()
def align2(mA,mB,Xref):
    h1A,h2A=acts(mA,Xref); h1B,h2B=acts(mB,Xref)
    def perm(A,B):
        A=(A-A.mean(0))/(A.std(0)+1e-6); B=(B-B.mean(0))/(B.std(0)+1e-6)
        _,c=linear_sum_assignment(-(A.T@B)/A.shape[0]); return c
    c1=perm(h1A,h1B); c2=perm(h2A,h2B); s=mB.state_dict(); o={k:v.clone() for k,v in s.items()}
    o["0.weight"]=s["0.weight"][c1]; o["0.bias"]=s["0.bias"][c1]
    o["2.weight"]=s["2.weight"][c2][:,c1]; o["2.bias"]=s["2.bias"][c2]; o["4.weight"]=s["4.weight"][:,c2]
    mC=mk(); mC.load_state_dict(o); return mC
def barrier(mA,mB,X,y,npts=11):
    lf=nn.CrossEntropyLoss(); sA,sB=mA.state_dict(),mB.state_dict(); m=mk(); ls=[]
    for a in np.linspace(0,1,npts):
        m.load_state_dict({k:(1-a)*sA[k]+a*sB[k] for k in sA})
        with torch.no_grad(): ls.append(lf(m(X),y).item())
    ls=np.array(ls); return float(ls.max()-max(ls[0],ls[-1]))

def study(name, genf, seedbase, label, NSEED=60):
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    Xdep,ydep=genf(4000,seedbase+50001,"deploy")
    models=[]; sc=[]; tac=[]
    for s in range(NSEED):
        m,Xt,yt=train(seedbase+s,genf); models.append((m,Xt,yt)); sc.append(accm(m,Xdep,ydep)); tac.append(accm(m,Xt,yt))
    sc=np.array(sc); tac=np.array(tac); h,_=np.histogram(sc,bins=10,range=(0,1))
    nHi=int((sc>0.8).sum()); nLo=int((sc<0.2).sum())
    print(f"  in-dist train acc {tac.mean():.3f}+/-{tac.std():.3f} | {label}: mean {sc.mean():.3f} std {sc.std():.3f} range [{sc.min():.3f},{sc.max():.3f}]")
    print(f"  high(>0.8) {nHi}/{NSEED} | low(<0.2) {nLo}/{NSEED} | mid {NSEED-nHi-nLo} | hist "+" ".join(f"{x:2d}" for x in h))
    hi=[i for i in range(NSEED) if sc[i]>0.8]; lo=[i for i in range(NSEED) if sc[i]<0.2]
    for tag,cl in [("hi",hi),("lo",lo)]:
        if not cl: continue
        m,Xt,yt=models[cl[0]]; m2,_,_=train(seedbase+cl[0],genf,steps=2*EPOCHS)
        print(f"  [{tag}] ||grad||={gnorm(m,Xt,yt):.2e} wnorm {wnorm(m):.2f}->{wnorm(m2):.2f}(2x) score {sc[cl[0]]:.2f}->{accm(m2,Xdep,ydep):.2f} Hess_min={eig_min(m,Xt,yt):+.2e}->{'MIN' if eig_min(m,Xt,yt)>-1e-2 else 'SADDLE'}")
    Xb,yb=genf(N,seedbase+99,"train"); rng=np.random.default_rng(0)
    def avg(ca,cb,al,same,k=20):
        out=[]
        for _ in range(k):
            i=int(rng.choice(ca)); j=int(rng.choice(cb))
            if same and i==j: continue
            mB=align2(models[i][0],models[j][0],Xb) if al else models[j][0]
            out.append(barrier(models[i][0],mB,Xb,yb))
        return float(np.mean(out)) if out else float('nan')
    res={}
    if hi and lo:
        res=dict(within_aln=avg(hi,hi,True,True),across_aln=avg(hi,lo,True,False),within_raw=avg(hi,hi,False,True),across_raw=avg(hi,lo,False,False))
        print(f"  barrier raw within {res['within_raw']:.3f} across {res['across_raw']:.3f} | ALIGNED within {res['within_aln']:.3f} across {res['across_aln']:.3f} -> {'connected mod perm' if res['across_aln']<2*max(res['within_aln'],1e-3) else 'distinct basins'}")
    return dict(mean=float(sc.mean()),std=float(sc.std()),min=float(sc.min()),max=float(sc.max()),nHi=nHi,nLo=nLo,nMid=NSEED-nHi-nLo,hist=h.tolist(),train_acc=float(tac.mean()),scores=sc.tolist(),barriers=res)

# (1) ALIGN baseline
rA=study("ALIGN baseline: aligned objective vs misaligned proxy (eval-awareness)", gen_align, 700, "deployment alignment", 60)

# (2) INTERVENTION sweep: how much anti-proxy data flips the population to aligned?
print(f"\n{'='*68}\nINTERVENTION: fraction f of anti-proxy training data -> population alignment\n{'='*68}")
sweep=[]
for f in [0.0,0.01,0.02,0.05,0.10,0.20]:
    al=[]
    Xdep,ydep=gen_align(4000,50001,"deploy")
    for s in range(30):
        m,_,_=train(700+s,gen_align,steps=EPOCHS,f=f); al.append(accm(m,Xdep,ydep))
    al=np.array(al); fa=float((al>0.8).mean())
    print(f"  f={f:4.2f} | mean alignment {al.mean():.3f} | fraction aligned(>0.8) {fa:.2f} | misaligned(<0.2) {float((al<0.2).mean()):.2f}")
    sweep.append(dict(f=f, mean_align=float(al.mean()), frac_aligned=fa, frac_misaligned=float((al<0.2).mean())))

# (3) FRAME companion (independent seeds), read as harm rate
rF=study("FRAME: consequentialist(=acts/harm) vs deontological(=refrains/safe)", gen_frame, 9000, "consequentialism index = harm rate", 60)

json.dump(dict(align=rA,frame=rF,intervention=sweep,config=dict(H=H,WD=WD,MARGIN=MARGIN,NOISE=NOISE,EPOCHS=EPOCHS)),
          open("/Users/steven/moral_framework_misgen/results_align.json","w"),indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":110,"font.size":11})
fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].hist(rA["scores"],bins=20,range=(0,1),color="#55A868",edgecolor="white"); ax[0].axvline(0.5,color="gray",ls="--")
ax[0].set_xlabel("deployment alignment (1=aligned, 0=defects)"); ax[0].set_ylabel("# seeds")
ax[0].set_title(f"ALIGN: by chance {rA['nHi']}/60 aligned, {rA['nLo']}/60 misaligned\n(identical except seed; genuine minima)")
fs=[d["f"] for d in sweep]
ax[1].plot(fs,[d["frac_aligned"] for d in sweep],"o-",color="#55A868",label="fraction aligned")
ax[1].plot(fs,[d["mean_align"] for d in sweep],"s--",color="#4C72B0",label="mean alignment")
ax[1].set_xlabel("fraction of anti-proxy training data (f)"); ax[1].set_ylabel("alignment"); ax[1].set_ylim(0,1.02)
ax[1].set_title("INTERVENTION: a little diverse data\nflips the population to aligned"); ax[1].legend(fontsize=9)
ax[2].hist(rF["scores"],bins=20,range=(0,1),color="#C44E52",edgecolor="white"); ax[2].axvline(0.5,color="gray",ls="--")
ax[2].set_xlabel("consequentialism index = harm rate (1=acts/harm, 0=refrains/safe)"); ax[2].set_ylabel("# seeds")
ax[2].set_title(f"FRAME: {rF['nHi']}/60 consequentialist(harm),\n{rF['nLo']}/60 deontological(safe)")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/align.png"); plt.close(fig)
print("\nSaved results_align.json + figs/align.png")
