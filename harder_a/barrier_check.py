"""
Barrier robustness: compute raw + permutation-aligned TRAIN-loss interpolation
barriers across MULTIPLE (block-B-gen, block-C-gen) pairs, plus a same-cluster
baseline, to characterize the basin structure honestly.

Uses the exact final config (width=8, margin=1.8).
"""
import os, json, copy
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

torch.set_num_threads(4)
N_NOISE = 3; IN_DIM = 4 + N_NOISE
CFG = dict(width=8, depth=2, wd=2e-3, margin=1.8, epochs=3000, lr=5e-3)

def make_data(n, seed, regime, margin):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 1000:
        tries += 1; m = target * 8
        B = rng.randn(m, 2).astype(np.float32); C = rng.randn(m, 2).astype(np.float32)
        for arr in (B, C):
            arr[:, 0] = np.sign(arr[:, 0]) * (np.abs(arr[:, 0]) + margin)
            arr[:, 1] = np.sign(arr[:, 1]) * (np.abs(arr[:, 1]) + margin)
        rB = ((B[:,0]>0)^(B[:,1]>0)).astype(np.int64)
        rC = ((C[:,0]>0)^(C[:,1]>0)).astype(np.int64)
        keep = (rB==rC) if regime=="train" else (rB!=rC)
        lab = rB[keep]
        Bk,Ck = B[keep],C[keep]
        noise = rng.randn(Bk.shape[0], N_NOISE).astype(np.float32)
        feat = np.concatenate([Bk,Ck,noise],axis=1)
        for i in range(Bk.shape[0]):
            if len(xs)>=target: break
            xs.append(feat[i]); ys.append(lab[i])
    return (torch.tensor(np.array(xs,np.float32)), torch.tensor(np.array(ys,np.int64)))

class MLP(nn.Module):
    def __init__(self,w,d):
        super().__init__()
        layers=[]; dd=IN_DIM
        for _ in range(d): layers+=[nn.Linear(dd,w),nn.ReLU()]; dd=w
        layers+=[nn.Linear(dd,2)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def acc(m,X,Y):
    with torch.no_grad(): return (m(X).argmax(1)==Y).float().mean().item()

def train(seed,Xtr,Ytr):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(CFG["width"],CFG["depth"])
    opt=torch.optim.Adam(m.parameters(),lr=CFG["lr"],weight_decay=CFG["wd"])
    crit=nn.CrossEntropyLoss()
    for _ in range(CFG["epochs"]):
        opt.zero_grad(); crit(m(Xtr),Ytr).backward(); opt.step()
    return m

def flat(m): return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def setflat(m,f):
    i=0
    for p in m.parameters():
        n=p.numel(); p.data.copy_(f[i:i+n].reshape(p.shape)); i+=n
def interp(mA,mB,X,Y,steps=21):
    pa,pb=flat(mA),flat(mB); tmp=MLP(CFG["width"],CFG["depth"]); crit=nn.CrossEntropyLoss(); L=[]
    for t in np.linspace(0,1,steps):
        setflat(tmp,(1-t)*pa+t*pb)
        with torch.no_grad(): L.append(crit(tmp(X),Y).item())
    return L
def bar(L): return max(L)-max(L[0],L[-1])
def acts(m,X):
    a=[]; h=X
    for l in m.net:
        h=l(h)
        if isinstance(l,nn.ReLU): a.append(h.detach())
    return a
def align(mA,mB,X):
    mB=copy.deepcopy(mB)
    li=[i for i,l in enumerate(mB.net) if isinstance(l,nn.Linear)]; hl=li[:-1]
    for k,idx in enumerate(hl):
        aA=acts(mA,X)[k]; aB=acts(mB,X)[k]
        Ac=aA-aA.mean(0,keepdim=True); Bc=aB-aB.mean(0,keepdim=True)
        corr=(Ac.t()@Bc)/((Ac.norm(dim=0)+1e-8).unsqueeze(1)*(Bc.norm(dim=0)+1e-8).unsqueeze(0))
        _,col=linear_sum_assignment(-corr.numpy()); perm=torch.tensor(col)
        lin=mB.net[idx]; lin.weight.data=lin.weight.data[perm,:]; lin.bias.data=lin.bias.data[perm]
        nx=mB.net[li[k+1]]; nx.weight.data=nx.weight.data[:,perm]
    return mB

if __name__=="__main__":
    mg=CFG["margin"]
    Xtr,Ytr=make_data(700,12345,"train",mg)
    Xood,Yood=make_data(2000,999,"ood",mg)
    # train enough seeds to find several B-gens and C-gens
    models={}; ood={}
    for s in range(40):
        m=train(s,Xtr,Ytr); models[s]=m; ood[s]=acc(m,Xood,Yood)
    Bg=[s for s in range(40) if ood[s]>0.9]
    Cg=[s for s in range(40) if ood[s]<0.1]
    print("blockB-gen seeds:",Bg[:6]); print("blockC-gen seeds:",Cg[:6],flush=True)
    out={"pairs_cross":[], "pairs_same":[]}
    # cross-cluster pairs
    for i in range(min(4,len(Bg),len(Cg))):
        b,c=Bg[i],Cg[i]
        Lr=interp(models[b],models[c],Xtr,Ytr)
        mca=align(models[b],models[c],Xtr); La=interp(models[b],mca,Xtr,Ytr)
        out["pairs_cross"].append(dict(B=b,C=c,raw=bar(Lr),aligned=bar(La),
                                       end=max(Lr[0],Lr[-1])))
        print(f"CROSS B{b} C{c}: raw={bar(Lr):.4f} aligned={bar(La):.4f}",flush=True)
    # same-cluster baseline (two B-gens) -> expect small even raw, smaller aligned
    for i in range(min(3,len(Bg)//2)):
        b1,b2=Bg[2*i],Bg[2*i+1]
        Lr=interp(models[b1],models[b2],Xtr,Ytr)
        mca=align(models[b1],models[b2],Xtr); La=interp(models[b1],mca,Xtr,Ytr)
        out["pairs_same"].append(dict(A=b1,B=b2,raw=bar(Lr),aligned=bar(La)))
        print(f"SAME B{b1} B{b2}: raw={bar(Lr):.4f} aligned={bar(La):.4f}",flush=True)
    import numpy as np
    cr=[p["raw"] for p in out["pairs_cross"]]; ca=[p["aligned"] for p in out["pairs_cross"]]
    out["summary"]=dict(cross_raw_mean=float(np.mean(cr)), cross_aligned_mean=float(np.mean(ca)))
    with open("/Users/steven/moral_framework_misgen/harder_a/barrier_check.json","w") as f:
        json.dump(out,f,indent=2)
    print(f"\nCROSS raw mean={np.mean(cr):.4f} aligned mean={np.mean(ca):.4f}")
    print("BARRIERCHECK_DONE",flush=True)
