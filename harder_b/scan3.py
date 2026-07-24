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
    print(f'mp={max_p} mh={max_h} V={V} ntr={ntr} w={width} d={depth} wd={wd}: '
          f'trmin={min(tracc):.3f} OOD m={oods.mean():.3f} s={oods.std():.3f} '
          f'min={oods.min():.3f} max={oods.max():.3f} rng={oods.max()-oods.min():.3f}',flush=True)

# small data -> init matters; vary width and wd; V>=max_p
print('=== small data, vary width ===',flush=True)
for ntr in [300,500,800]:
    for w in [32,64,128,256]:
        sweep(8,6,3,ntr,w,2e-3,800,3e-3)
print('=== vary wd at ntr=500 w=64 ===',flush=True)
for wd in [1e-3,2e-3,3e-3]:
    sweep(8,6,3,500,64,wd,800,3e-3)
