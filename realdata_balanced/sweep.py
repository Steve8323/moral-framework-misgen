"""
Fast config sweep to find the EQUAL-COST balanced regime.

For each config we measure, with a SMALL number of seeds:
  - shape-only achievable acc (train on color-DEcorrelated data -> must use shape)
  - color-only achievable acc (train on shape-DEstroyed data -> must use color)
  - multiplicity on the real spurious task (how seeds split shape vs color)
Equal-cost target: shape-only ~= color-only acc, AND the seeds SPLIT.
"""
import sys, time, json, numpy as np, torch, torch.nn as nn
import core
from core import CFG, mk, accm, gen, color_reliance, _make_colored, _TRX, _TRY, _TEX, _TEY, H, W

def set_cfg(**kw):
    for k, v in kw.items(): setattr(CFG, k, v)

def quick_train(seed, steps, Xtr, ytr):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    m = mk(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), CFG.LR, weight_decay=CFG.WD)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m

def shape_only_acc(seed, steps):
    # color fully decorrelated in train -> color is useless -> model must use shape
    old = CFG.P_CORR_TRAIN; CFG.P_CORR_TRAIN = 0.5
    Xtr, ytr = gen(CFG.N, seed, "train")
    CFG.P_CORR_TRAIN = old
    m = quick_train(seed, steps, Xtr, ytr)
    Xte, yte = gen(4000, seed + 7, "deploy")   # OOD color flipped (irrelevant since shape)
    return accm(m, Xte, yte), accm(m, Xtr, ytr)

def color_only_acc(seed, steps):
    # destroy shape: use a BLANK digit (zeros) so only the color tint carries signal.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(_TRX), size=CFG.N, replace=False)
    blank = np.zeros((CFG.N, H, W), np.float32) + 0.5   # mid-gray so tint is visible
    Xtr, ytr = _make_colored(blank, _TRY[idx], seed, CFG.P_CORR_TRAIN)
    m = quick_train(seed, steps, Xtr, ytr)
    # in-train-distribution color test (color still agrees w.p. P_CORR_TRAIN)
    idx2 = rng.choice(len(_TEX), size=4000, replace=False)
    blank2 = np.zeros((4000, H, W), np.float32) + 0.5
    Xte, yte = _make_colored(blank2, _TEY[idx2], seed + 7, CFG.P_CORR_TRAIN)
    return accm(m, Xte, yte), accm(m, Xtr, ytr)

def multiplicity(nseed, steps, base=7000):
    oods = []; crs = []
    Xdep, ydep = gen(4000, 999001, "deploy")
    for s in range(nseed):
        Xtr, ytr = gen(CFG.N, base + s, "train")
        m = quick_train(base + s, steps, Xtr, ytr)
        oods.append(accm(m, Xdep, ydep)); crs.append(color_reliance(m, n=1000))
    return np.array(oods), np.array(crs)

if __name__ == "__main__":
    t0 = time.time()
    # configs to try: (LABEL_FLIP, P_CORR_TRAIN, TINT, PATCH, CNOISE, GRAY_BASE)
    STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    NS    = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    configs = [
        dict(LABEL_FLIP=0.05, P_CORR_TRAIN=0.90, TINT=0.30, PATCH=0,  CNOISE=0.0, GRAY_BASE=True),
        dict(LABEL_FLIP=0.05, P_CORR_TRAIN=0.85, TINT=0.30, PATCH=0,  CNOISE=0.0, GRAY_BASE=True),
        dict(LABEL_FLIP=0.05, P_CORR_TRAIN=0.85, TINT=0.20, PATCH=0,  CNOISE=0.10, GRAY_BASE=True),
        dict(LABEL_FLIP=0.05, P_CORR_TRAIN=0.80, TINT=0.25, PATCH=6,  CNOISE=0.05, GRAY_BASE=True),
        dict(LABEL_FLIP=0.02, P_CORR_TRAIN=0.85, TINT=0.20, PATCH=0,  CNOISE=0.15, GRAY_BASE=True),
        dict(LABEL_FLIP=0.05, P_CORR_TRAIN=0.82, TINT=0.18, PATCH=0,  CNOISE=0.12, GRAY_BASE=True),
    ]
    out = []
    for ci, cfg in enumerate(configs):
        set_cfg(**cfg)
        # equal-cost measurement (2 seeds each, cheap)
        sh = [shape_only_acc(100 + i, STEPS) for i in range(2)]
        co = [color_only_acc(200 + i, STEPS) for i in range(2)]
        sh_ood = np.mean([a for a, _ in sh]); co_acc = np.mean([a for a, _ in co])
        oods, crs = multiplicity(NS, STEPS)
        ft = float((oods > 0.55).mean()); fm = float((oods < 0.45).mean())
        rec = dict(cfg=cfg, shape_only_ood=float(sh_ood), color_only_acc=float(co_acc),
                   ood_mean=float(oods.mean()), ood_std=float(oods.std()),
                   ood_min=float(oods.min()), ood_max=float(oods.max()),
                   frac_true=ft, frac_spur=fm, cr_mean=float(crs.mean()),
                   oods=oods.tolist())
        out.append(rec)
        print(f"[{ci}] flip={cfg['LABEL_FLIP']} pcorr={cfg['P_CORR_TRAIN']} tint={cfg['TINT']} "
              f"patch={cfg['PATCH']} cn={cfg['CNOISE']} | shapeOOD={sh_ood:.2f} colorAcc={co_acc:.2f} "
              f"| OOD mean={oods.mean():.2f} sd={oods.std():.2f} rng[{oods.min():.2f},{oods.max():.2f}] "
              f"true={ft:.2f} spur={fm:.2f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open("sweep_results.json", "w"), indent=2)
    print(f"done {time.time()-t0:.0f}s", flush=True)
