"""
Scan4: the RECOGNIZED 'two perfect-train solutions' regime (HANS/cmnist-with-p->1 style).
If BOTH cues are individually PERFECT on train, train loss has (at least) two distinct
zero-loss solutions -> the seed picks which -> genuine bimodal split, not a blend.

Levers:
  - LABEL_FLIP=0  (no label noise; shape can reach ~1.0 train)
  - p_corr_train=1.0  (patch PERFECTLY predicts the train label -> patch-only is also ~1.0 train)
  - vary patch contrast amp and shape difficulty so NEITHER is the trivially-cheaper solution
BOTTLE=2 (so each cue can be cleanly represented and reach genuine minima).
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

def run(seed, pc, sn, amp, n=P.N):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    P.P_CORR_TRAIN = pc
    Xtr, ytr = P.gen(n, seed, "train", amp=amp, snoise=sn)
    m = P.mk(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), P.LR, weight_decay=P.WD)
    for _ in range(STEPS): opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

# patch PERFECT on train (pc=1.0); vary how separable each cue is
grid = [(1.0, 0.30, 0.08), (1.0, 0.30, 0.15), (1.0, 0.30, 0.25),
        (1.0, 0.45, 0.10), (1.0, 0.45, 0.18), (1.0, 0.60, 0.12)]
print(f"scan4 BOTTLE=2 LABEL_FLIP=0 pc=1.0  {len(grid)} configs x 8 seeds, {STEPS} steps", flush=True)
res = []
for (pc, sn, amp) in grid:
    P.P_CORR_TEST = 0.0  # patch fully FLIPPED at OOD
    Xdep, ydep = P.gen(4000, 999001, "deploy", amp=amp, snoise=sn)
    ood = []; tr = []
    for s in range(8):
        m, Xt, yt = run(3000 + s, pc, sn, amp)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood); hi = int((ood > 0.55).sum()); lo = int((ood < 0.45).sum())
    print(f"pc={pc} sn={sn} amp={amp:.2f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi {hi} lo {lo} mid {8-hi-lo} | split={min(hi,lo)}  ({time.time()-t0:.0f}s)", flush=True)
    res.append((min(hi, lo), ood.std(), pc, sn, amp, hi, lo, float(ood.mean()), float(np.mean(tr))))
res.sort(key=lambda r: (-r[0], -r[1]))
print("\nTOP (by split then spread):", flush=True)
for r in res[:5]:
    print(f"  split={r[0]} std={r[1]:.3f} pc={r[2]} sn={r[3]} amp={r[4]:.2f} hi={r[5]} lo={r[6]} mean={r[7]:.3f} train={r[8]:.3f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
