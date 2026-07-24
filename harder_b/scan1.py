import numpy as np
from task import gen_train, gen_ood, feature_dim
from train import train_one, accuracy

def sweep(V,max_p,max_h,ntr,width,wd,epochs,lr,seeds=10,depth=2):
    in_dim=feature_dim(V,max_p,max_h)
    Xtr,ytr=gen_train(ntr,V,max_p,max_h,seed=12345)
    Xood,yood,ht=gen_ood(3000,V,max_p,max_h,seed=999)
    oods=[]; tracc=[]
    for s in range(seeds):
        m,_=train_one(Xtr,ytr,s,in_dim,width=width,depth=depth,wd=wd,epochs=epochs,lr=lr)
        oods.append(accuracy(m,Xood,yood)); tracc.append(accuracy(m,Xtr,ytr))
    oods=np.array(oods)
    print(f'V={V} mp={max_p} mh={max_h} ntr={ntr} w={width} d={depth} wd={wd} ep={epochs}: '
          f'tracc_min={min(tracc):.3f} OOD mean={oods.mean():.3f} std={oods.std():.3f} '
          f'min={oods.min():.3f} max={oods.max():.3f} range={oods.max()-oods.min():.3f}',flush=True)

# max_h=2: mp=3 valid; order matters (>=2 needed to scramble)
print('=== max_h=2 ===',flush=True)
for mp in [3,4,5]:
    for ntr in [1500,3000,6000]:
        sweep(8,mp,2,ntr,128,2e-3,600,3e-3)
print('=== max_h=3, mp=4,5 ===',flush=True)
for mp in [4,5]:
    for ntr in [1500,3000,6000]:
        sweep(8,mp,3,ntr,128,2e-3,600,3e-3)
