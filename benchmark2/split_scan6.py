"""
Scan6: FINE bracket around the crossover (amp 0.045-0.065, pn=0.08) in the
perfect-train regime (pc=1.0, LABEL_FLIP=0, BOTTLE=2). 12 seeds/config to resolve
whether seeds genuinely SPLIT (both clusters populated) at the equal-difficulty amp.
"""
import os
os.environ["BOTTLE"] = "2"
import time, numpy as np, torch
import patch_fmnist as P
torch.set_num_threads(4)
P.LABEL_FLIP = 0.0
t0 = time.time()
STEPS = 400

def run(seed, amp, sn=0.20, pn=0.08, n=P.N):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    P.P_CORR_TRAIN = 1.0
    Xtr, ytr = P.gen(n, seed, "train", amp=amp, snoise=sn, pnoise=pn)
    m = P.mk(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), P.LR, weight_decay=P.WD)
    for _ in range(STEPS): opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

P.P_CORR_TEST = 0.0
amps = [0.045, 0.050, 0.055, 0.060, 0.065]
print(f"scan6 fine bracket  {len(amps)} amps x 12 seeds, pn=0.08 sn=0.20", flush=True)
res = []
for amp in amps:
    Xdep, ydep = P.gen(4000, 999001, "deploy", amp=amp, snoise=0.20, pnoise=0.08)
    ood = []; tr = []
    for s in range(12):
        m, Xt, yt = run(3000 + s, amp)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood); hi = int((ood > 0.55).sum()); lo = int((ood < 0.45).sum())
    print(f"amp={amp:.3f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi {hi} lo {lo} mid {12-hi-lo} | split={min(hi,lo)}  ({time.time()-t0:.0f}s)", flush=True)
    res.append((min(hi, lo), amp, hi, lo, float(ood.std()), float(ood.mean()), float(np.mean(tr))))
res.sort(key=lambda r: (-r[0], -r[4]))
print("\nBEST:", flush=True)
for r in res[:3]:
    print(f"  split={r[0]} amp={r[1]:.3f} hi={r[2]} lo={r[3]} std={r[4]:.3f} mean={r[5]:.3f} train={r[6]:.3f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
