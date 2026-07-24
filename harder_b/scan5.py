import time, numpy as np, torch
torch.set_num_threads(4)
from task import gen_tokens
from atrain import train_a, acc_a

V,max_p,max_h=8,6,3
ood=gen_tokens(2500,V,max_p,max_h,seed=999,ood=True)

def sweep(ntr,dm,ff,wd,ep,lr,seeds=10):
    tr=gen_tokens(ntr,V,max_p,max_h,seed=12345,ood=False)
    oa=[];ta=[]
    for s in range(seeds):
        m=train_a(tr,s,V,max_p,max_h,d_model=dm,nhead=4,dim_ff=ff,wd=wd,epochs=ep,lr=lr)
        oa.append(acc_a(m,ood)); ta.append(acc_a(m,tr))
    oa=np.array(oa)
    print(f'ntr={ntr} dm={dm} ff={ff} wd={wd} ep={ep} lr={lr}: trmin={min(ta):.3f} '
          f'OODm={oa.mean():.3f} s={oa.std():.3f} min={oa.min():.3f} max={oa.max():.3f} '
          f'rng={oa.max()-oa.min():.3f} | vals={np.round(np.sort(oa),3).tolist()}',flush=True)

# tipping point search: moderate data, vary capacity & lr
sweep(800,48,96,1e-3,1500,3e-3)
sweep(800,64,128,1e-3,1500,3e-3)
sweep(1200,48,96,1e-3,1500,5e-3)
sweep(800,48,96,1e-3,1500,1e-2)
