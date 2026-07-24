"""
Find the EQUAL-COST patch amplitude: train shape-ONLY and patch-ONLY models and
locate the patch contrast where both reach the SAME train-accuracy ceiling AND the
combined-cue model genuinely SPLITS across seeds.

Equal-cost means: with ONLY the patch visible, a model fits train about as well as
with ONLY the shape visible. If patch >> shape (cmnist's color), every seed takes
the shortcut (no split). If patch << shape, no seed does (no split). We want the tie.
"""
import time, numpy as np, torch
import patch_fmnist as P

torch.set_num_threads(8)
t0 = time.time()

def quick_train(seed, gen_kw, steps=300, n=800):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    Xtr, ytr = P.gen(n, seed, "train", **gen_kw)
    m = P.mk(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), P.LR, weight_decay=P.WD)
    for _ in range(steps): opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

# 1) shape-only ceiling (patch off): how well can shape alone fit train?
print("=== shape-only train ceiling (patch OFF) ===", flush=True)
sh = []
for s in range(6):
    m, Xt, yt = quick_train(1000 + s, dict(patch_on=False, shape_on=True))
    sh.append(P.accm(m, Xt, yt))
print(f"  shape-only train acc: mean {np.mean(sh):.3f}  range [{min(sh):.3f},{max(sh):.3f}]  ({time.time()-t0:.0f}s)", flush=True)

# 2) patch-only ceiling vs amplitude (shape off): scan PATCH_AMP
print("\n=== patch-only train ceiling (shape OFF) vs amplitude ===", flush=True)
for amp in [0.08, 0.12, 0.18, 0.25, 0.35, 0.5]:
    pa = []
    for s in range(5):
        m, Xt, yt = quick_train(2000 + s, dict(patch_on=True, shape_on=False, amp=amp))
        pa.append(P.accm(m, Xt, yt))
    print(f"  amp={amp:.2f} | patch-only train acc mean {np.mean(pa):.3f} range [{min(pa):.3f},{max(pa):.3f}]  ({time.time()-t0:.0f}s)", flush=True)

# 3) combined-cue split check at a few amplitudes (5-seed smoke, OOD spread is the signal)
print("\n=== combined-cue OOD spread (5 seeds) vs amplitude -> do seeds split? ===", flush=True)
Xdep, ydep = P.gen(4000, 999001, "deploy")
for amp in [0.12, 0.18, 0.25, 0.35]:
    ood = []; tr = []
    for s in range(5):
        m, Xt, yt = quick_train(3000 + s, dict(amp=amp), steps=400, n=P.N)
        ood.append(P.accm(m, Xdep, ydep)); tr.append(P.accm(m, Xt, yt))
    ood = np.array(ood)
    print(f"  amp={amp:.2f} | train {np.mean(tr):.3f} | OOD mean {ood.mean():.3f} std {ood.std():.3f} "
          f"range [{ood.min():.3f},{ood.max():.3f}] | hi(>0.55) {(ood>0.55).sum()} lo(<0.45) {(ood<0.45).sum()}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
