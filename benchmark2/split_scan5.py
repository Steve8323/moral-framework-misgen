"""
Scan5: perfect-train regime (pc=1.0, LABEL_FLIP=0, BOTTLE=2) but drive the patch
contrast VERY LOW and/or add patch noise so the patch becomes HARDER than shape ->
find the equal-difficulty crossover where seeds SPLIT between the (perfect-train)
patch solution and the (perfect-train) shape solution.
8 seeds/config, 400 steps.
"""
import os
os.environ["BOTTLE"] = "2"
import time, numpy as np, torch
import patch_fmnist as P
torch.set_num_threads(4)
P.LABEL_FLIP = 0.0
t0 = time.time()
STEPS = 400

def run(seed, pc, sn, amp, pn, n=P.N):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    P.P_CORR_TRAIN = pc
    Xtr, ytr = P.gen(n, seed, "train", amp=amp, snoise=sn, pnoise=pn)
    m = P.mk(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), P.LR, weight_decay=P.WD)
    for _ in range(STEPS): opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

# very low amp and/or higher patch noise -> handicap the patch toward shape's difficulty
# (pc=1.0 so the patch, when readable, is still a perfect-train solution)
grid = [(1.0, 0.20, 0.04, 0.06), (1.0, 0.20, 0.03, 0.10), (1.0, 0.20, 0.05, 0.12),
        (1.0, 0.25, 0.04, 0.10), (1.0, 0.15, 0.03, 0.12), (1.0, 0.20, 0.02, 0.08)]
print(f"scan5 perfect-train, LOW-amp/high-pnoise  {len(grid)} configs x 8 seeds", flush=True)
res = []
for (pc, sn, amp, pn) in grid:
    P.P_CORR_TEST = 0.0
    Xdep, ydep = P.gen(4000, 999001, "deploy", amp=amp, snoise=sn, pnoise=pn)
    ood = []; tr = []
    for s in range(8):
        m, Xt, yt = run(3000 + s, pc, sn, amp, pn)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood); hi = int((ood > 0.55).sum()); lo = int((ood < 0.45).sum())
    print(f"pc={pc} sn={sn} amp={amp:.2f} pn={pn:.2f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi {hi} lo {lo} mid {8-hi-lo} | split={min(hi,lo)}  ({time.time()-t0:.0f}s)", flush=True)
    res.append((min(hi, lo), ood.std(), pc, sn, amp, pn, hi, lo, float(ood.mean()), float(np.mean(tr))))
res.sort(key=lambda r: (-r[0], -r[1]))
print("\nTOP (by split then spread):", flush=True)
for r in res[:5]:
    print(f"  split={r[0]} std={r[1]:.3f} pc={r[2]} sn={r[3]} amp={r[4]:.2f} pn={r[5]:.2f} hi={r[6]} lo={r[7]} mean={r[8]:.3f} train={r[9]:.3f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
