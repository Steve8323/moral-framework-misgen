"""
Lean intervention sweep (15 seeds/f) at the equal-cost config, to deliver the
counter-spurious mitigation curve without waiting on the contended 40-seed sweep
inside run_benchmark2.py. Same construction; uses patch_fmnist module defaults
(pc=1.0, LABEL_FLIP=0, amp=0.055, BOTTLE=2). Writes intervention.json.
"""
import os, json, time, numpy as np, torch
os.environ.setdefault("BOTTLE", "2")
import patch_fmnist as P
torch.set_num_threads(4)
t0 = time.time()
NS = 15
SEEDBASE = 4000
Xdep, ydep = P.gen(8000, 999001, "deploy")
print(f"lean intervention: {NS} seeds/f, config amp={P.PATCH_AMP} pc={P.P_CORR_TRAIN} flip={P.LABEL_FLIP}", flush=True)
sweep = []
for f in [0.0, 0.01, 0.02, 0.05, 0.10]:
    al = []
    for s in range(NS):
        m, _, _ = P.train(SEEDBASE + s, f=f)
        al.append(P.accm(m, Xdep, ydep))
    al = np.array(al)
    fa = float((al > 0.55).mean()); fm = float((al < 0.45).mean())
    print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | frac uses-true(>0.55) {fa:.2f} | uses-spur(<0.45) {fm:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    sweep.append(dict(f=f, mean_ood=float(al.mean()), frac_uses_true=fa, frac_uses_spur=fm, ood_scores=al.tolist()))
    json.dump(dict(nseeds=NS, sweep=sweep), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "intervention.json"), "w"), indent=2)
print(f"DONE {time.time()-t0:.0f}s. Wrote intervention.json", flush=True)
