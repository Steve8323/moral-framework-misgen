"""
Minimal decisive scan for the equal-cost / SPLIT point. Lever: shape difficulty
(shape_noise) at fixed patch contrast, plus patch predictiveness p_corr_train.
6 seeds/config, 300 steps. Goal: find a config where OOD accuracy SPLITS
(both hi=shape and lo=patch clusters non-empty), not collapses.
"""
import time, numpy as np, torch
import patch_fmnist as P
torch.set_num_threads(4)
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

# focus: vary shape_noise (true-feature difficulty) and p_corr (patch predictiveness)
grid = [(0.80, 0.40, 0.18), (0.80, 0.60, 0.18),
        (0.85, 0.40, 0.18), (0.85, 0.60, 0.18),
        (0.90, 0.60, 0.18), (0.90, 0.80, 0.18)]
print(f"scan {len(grid)} configs x 6 seeds, {STEPS} steps", flush=True)
res = []
for (pc, sn, amp) in grid:
    P.P_CORR_TEST = 1 - pc
    Xdep, ydep = P.gen(4000, 999001, "deploy", amp=amp, snoise=sn)
    ood = []; tr = []
    for s in range(6):
        m, Xt, yt = run(3000 + s, pc, sn, amp)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood); hi = int((ood > 0.55).sum()); lo = int((ood < 0.45).sum())
    print(f"pc={pc} sn={sn} amp={amp:.2f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi {hi} lo {lo} mid {6-hi-lo} | split={min(hi,lo)}  ({time.time()-t0:.0f}s)", flush=True)
    res.append((min(hi, lo), ood.std(), pc, sn, amp, hi, lo, float(ood.mean())))
res.sort(key=lambda r: (-r[0], -r[1]))
print("\nTOP:", flush=True)
for r in res[:4]:
    print(f"  split={r[0]} std={r[1]:.3f} pc={r[2]} sn={r[3]} amp={r[4]:.2f} hi={r[5]} lo={r[6]} mean={r[7]:.3f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
