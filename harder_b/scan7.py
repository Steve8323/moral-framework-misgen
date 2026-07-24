import numpy as np
from task import gen_symbits, symbits_dim
from train import train_one, accuracy
V,mp,mh=8,6,3
def run(nd,ntr,w,depth,wd,ep,lr,seeds=40):
    in_dim=symbits_dim(nd)
    Xood,yood,ht=gen_symbits(3000,V,mp,mh,seed=999,ood=True,n_distract=nd)
    Xtr,ytr,_=gen_symbits(ntr,V,mp,mh,seed=12345,ood=False,n_distract=nd)
    oa=[];ta=[]
    for s in range(seeds):
        m,_=train_one(Xtr,ytr,s,in_dim,width=w,depth=depth,wd=wd,epochs=ep,lr=lr)
        oa.append(accuracy(m,Xood,yood)); ta.append(accuracy(m,Xtr,ytr))
    oa=np.array(oa)
    print(f'nd={nd} ntr={ntr} w={w} d={depth} wd={wd} ep={ep} lr={lr}: trmin={min(ta):.3f} '
          f'OODm={oa.mean():.3f} s={oa.std():.3f} min={oa.min():.3f} max={oa.max():.3f} '
          f'rng={oa.max()-oa.min():.3f} low<.6={(oa<0.6).sum()} high>.8={(oa>0.8).sum()}',flush=True)
# best-looking regime + variations
run(8,400,64,2,2e-3,800,3e-3)
run(16,400,64,2,2e-3,800,3e-3)
run(16,300,48,3,1.5e-3,1000,3e-3)
run(8,250,64,2,1.5e-3,1000,5e-3)
run(12,400,96,2,1.5e-3,1000,3e-3)
