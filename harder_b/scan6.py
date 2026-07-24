import numpy as np
from task import gen_symbits, symbits_dim
from train import train_one, accuracy
V,mp,mh=8,6,3; nd=8; in_dim=symbits_dim(nd)
Xood,yood,ht=gen_symbits(3000,V,mp,mh,seed=999,ood=True,n_distract=nd)
def sweep(ntr,w,wd,ep,lr,seeds=30):
    Xtr,ytr,_=gen_symbits(ntr,V,mp,mh,seed=12345,ood=False,n_distract=nd)
    oa=[];ta=[]
    for s in range(seeds):
        m,_=train_one(Xtr,ytr,s,in_dim,width=w,depth=2,wd=wd,epochs=ep,lr=lr)
        oa.append(accuracy(m,Xood,yood)); ta.append(accuracy(m,Xtr,ytr))
    oa=np.array(oa)
    # bimodality: count near-0.5 (heur) vs near-1.0 (rule)
    nlow=(oa<0.6).sum(); nhigh=(oa>0.9).sum()
    print(f'ntr={ntr} w={w} wd={wd} ep={ep} lr={lr}: trmin={min(ta):.3f} OODm={oa.mean():.3f} '
          f's={oa.std():.3f} min={oa.min():.3f} max={oa.max():.3f} rng={oa.max()-oa.min():.3f} '
          f'|low(<.6)={nlow} high(>.9)={nhigh}',flush=True)

print('=== push commitment: less data, more wd ===',flush=True)
for ntr in [400,200,120,80]:
    for wd in [2e-3,3e-3]:
        sweep(ntr,64,wd,800,3e-3)
print('=== higher lr to jump basins ===',flush=True)
for lr in [1e-2,3e-2]:
    sweep(200,64,3e-3,800,lr)
