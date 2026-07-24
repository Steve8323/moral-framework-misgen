import numpy as np
from task import gen_train_match, gen_ood_match, match_dim
from train import train_one, accuracy

def sweep(V,max_p,max_h,ntr,width,wd,epochs,lr,seeds=12,depth=2,oodn=3000):
    in_dim=match_dim(max_p,max_h)
    Xtr,ytr=gen_train_match(ntr,V,max_p,max_h,seed=12345)
    Xood,yood,ht=gen_ood_match(oodn,V,max_p,max_h,seed=999)
    oods=[]; tracc=[]
    for s in range(seeds):
        m,_=train_one(Xtr,ytr,s,in_dim,width=width,depth=depth,wd=wd,epochs=epochs,lr=lr)
        oods.append(accuracy(m,Xood,yood)); tracc.append(accuracy(m,Xtr,ytr))
    oods=np.array(oods)
    print(f'mp={max_p} mh={max_h} ntr={ntr} w={width} d={depth} wd={wd} ep={epochs} lr={lr}: '
          f'trmin={min(tracc):.3f} OODm={oods.mean():.3f} s={oods.std():.3f} '
          f'min={oods.min():.3f} max={oods.max():.3f} rng={oods.max()-oods.min():.3f}',flush=True)

# vary LR (which basin) and epochs; longer mh to make true rule richer
print('=== mp8 mh4 vary lr ===',flush=True)
for lr in [1e-3,3e-3,1e-2,3e-2]:
    sweep(8,8,4,1500,128,2e-3,800,lr)
print('=== mp6 mh3 high lr short train ===',flush=True)
for lr in [1e-2,3e-2,5e-2]:
    for ep in [150,300]:
        sweep(8,6,3,800,128,2e-3,ep,lr)
print('=== bigger task mp10 mh5 V12 ===',flush=True)
for ntr in [1500,4000]:
    sweep(12,10,5,ntr,128,2e-3,800,3e-3)
