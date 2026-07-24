"""
Scan3: BOTTLE=1 (single-unit bottleneck) forces a near-binary cue commitment ->
test whether the crossover now produces a genuine BIMODAL SPLIT instead of the
BOTTLE=2 blended mid-OOD basin. Scan pc around the crossover at low amp.
8 seeds/config, 300 steps. (BOTTLE is read from env by patch_fmnist.)
"""
import os
os.environ["BOTTLE"] = "1"
import time, numpy as np, torch
import patch_fmnist as P
torch.set_num_threads(4)
assert P.BOTTLE == 1, P.BOTTLE
t0 = time.time()
STEPS = 300

def run(seed, pc, sn, amp, n=P.N):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    P.P_CORR_TRAIN = pc
    Xtr, ytr = P.gen(n, seed, "train", amp=amp, snoise=sn)
    m = P.mk(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), P.LR, weight_decay=P.WD)
    for _ in range(STEPS): opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

grid = [(0.68, 0.20, 0.10), (0.70, 0.20, 0.10), (0.72, 0.20, 0.10),
        (0.70, 0.25, 0.12), (0.72, 0.25, 0.12), (0.74, 0.20, 0.12)]
print(f"scan3 BOTTLE=1  {len(grid)} configs x 8 seeds, {STEPS} steps", flush=True)
res = []
for (pc, sn, amp) in grid:
    P.P_CORR_TEST = 1 - pc
    Xdep, ydep = P.gen(4000, 999001, "deploy", amp=amp, snoise=sn)
    ood = []; tr = []
    for s in range(8):
        m, Xt, yt = run(3000 + s, pc, sn, amp)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood); hi = int((ood > 0.55).sum()); lo = int((ood < 0.45).sum())
    print(f"pc={pc} sn={sn} amp={amp:.2f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi {hi} lo {lo} mid {8-hi-lo} | split={min(hi,lo)}  ({time.time()-t0:.0f}s)", flush=True)
    res.append((min(hi, lo), ood.std(), pc, sn, amp, hi, lo, float(ood.mean())))
res.sort(key=lambda r: (-r[0], -r[1]))
print("\nTOP:", flush=True)
for r in res[:4]:
    print(f"  split={r[0]} std={r[1]:.3f} pc={r[2]} sn={r[3]} amp={r[4]:.2f} hi={r[5]} lo={r[6]} mean={r[7]:.3f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
