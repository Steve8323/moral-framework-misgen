import numpy as np
from task import gen_train_match, gen_ood_match, match_dim
from train import train_one, accuracy

def sweep(V,max_p,max_h,ntr,width,wd,epochs,lr,seeds=10,depth=2):
    in_dim=match_dim(max_p,max_h)
    Xtr,ytr=gen_train_match(ntr,V,max_p,max_h,seed=12345)
    Xood,yood,ht=gen_ood_match(3000,V,max_p,max_h,seed=999)
    oods=[]; tracc=[]
    for s in range(seeds):
        m,_=train_one(Xtr,ytr,s,in_dim,width=width,depth=depth,wd=wd,epochs=epochs,lr=lr)
        oods.append(accuracy(m,Xood,yood)); tracc.append(accuracy(m,Xtr,ytr))
    oods=np.array(oods)
    print(f'V={V} mp={max_p} mh={max_h} ntr={ntr} w={width} d={depth} wd={wd} ep={epochs}: '
          f'tracc_min={min(tracc):.3f} OOD mean={oods.mean():.3f} std={oods.std():.3f} '
          f'min={oods.min():.3f} max={oods.max():.3f} range={oods.max()-oods.min():.3f}',flush=True)

for mp,mh in [(6,3),(8,4),(10,4)]:
    for ntr in [1000,2000,4000]:
        sweep(8,mp,mh,ntr,128,2e-3,600,3e-3)
