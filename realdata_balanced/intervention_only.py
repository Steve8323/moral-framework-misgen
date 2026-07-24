"""
Standalone LEAN intervention sweep (run separately if the main run is slow).
Uses the SAME locked balanced config. Writes intervention_results.json so the
report can pick it up if results.json's intervention block is incomplete.
"""
import os, json, time, numpy as np
import core
from core import CFG, train, accm, gen

CFG.LABEL_FLIP=0.05; CFG.P_CORR_TRAIN=0.85; CFG.P_CORR_TEST=0.10
CFG.TINT=0.30; CFG.PATCH=0; CFG.CNOISE=0.0; CFG.GRAY_BASE=True
CFG.BOTTLE=2; CFG.WD=1e-3; CFG.LR=2e-3; CFG.N=1000
INT_EPOCHS=600
NS=25
SEEDBASE=7000
HERE=os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__":
    t0=time.time(); Xdep,ydep=gen(8000,999001,"deploy")
    sweep=[]
    for f in [0.0,0.01,0.02,0.05,0.10,0.20]:
        al=[]
        for s in range(NS):
            m,_,_=train(SEEDBASE+s,steps=INT_EPOCHS,f=f); al.append(accm(m,Xdep,ydep))
        al=np.array(al); fa=float((al>0.55).mean()); fm=float((al<0.45).mean())
        print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | uses-true {fa:.2f} | uses-spur {fm:.2f} ({time.time()-t0:.0f}s)",flush=True)
        sweep.append(dict(f=f,mean_ood=float(al.mean()),frac_uses_true=fa,frac_uses_spur=fm,ood_scores=al.tolist()))
        json.dump(dict(epochs=INT_EPOCHS,nseed=NS,sweep=sweep),open(os.path.join(HERE,"intervention_results.json"),"w"),indent=2)
    print(f"done {time.time()-t0:.0f}s",flush=True)
