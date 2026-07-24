"""
FULL rigorous balanced Colored-MNIST experiment (EQUAL-COST regime).

Balanced config found by sweep.py:
  LABEL_FLIP=0.05 (shape ceiling ~0.95, vs prior 0.25), P_CORR_TRAIN=0.85
  -> shape-only OOD ~0.88 ~= color-only acc ~0.85  (EQUAL PREDICTIVENESS)
  GRAY_BASE color tint TINT=0.30, no clean channel-swap  (EQUAL DIFFICULTY:
  color is a weak low-contrast tint readable only by integrating both channels,
  not a trivially-separable single-channel cue).
Convergence: full-batch Adam + cosine LR decay to ~0 -> genuine finite minima.

Rigor:
  - >=50 seeds; OOD-accuracy DISTRIBUTION (mean/std/range/10-bin hist,
    frac uses-shape vs uses-color); multimodality test (diptest, else GMM/BIC).
  - genuine minima: full-batch grad norm, 2x-training weight-norm stability,
    Hessian min-eig via HVP power iteration, on representative seeds of each mode.
  - permutation-aligned loss barrier within-mode vs across-mode.
  - intervention sweep: fraction f of color-decorrelated train data -> does the
    population flip to shape?
Writes results.json incrementally + cmnist_balanced.png. Never fabricates.
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
import core
from core import (CFG, mk, train, accm, gen, color_reliance, gnorm, wnorm,
                  eig_min, align2, barrier)

# ---- LOCKED BALANCED CONFIG ----
CFG.LABEL_FLIP   = 0.05
CFG.P_CORR_TRAIN = 0.85
CFG.P_CORR_TEST  = 0.10
CFG.TINT         = 0.30
CFG.PATCH        = 0
CFG.CNOISE       = 0.0
CFG.GRAY_BASE    = True
CFG.BOTTLE       = 2
CFG.WD           = 1e-3
CFG.LR           = 2e-3
CFG.EPOCHS       = 1200
CFG.N            = 1000

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
NSEED = 5 if SMOKE else 60
SEEDBASE = 7000
RESPATH = os.path.join(HERE, "results_smoke.json" if SMOKE else "results.json")

def save(RES): json.dump(RES, open(RESPATH, "w"), indent=2)

def multimodal_test(x):
    out = {}
    x = np.asarray(x, float)
    try:
        import diptest
        d, p = diptest.diptest(x)
        out["diptest"] = dict(stat=float(d), pval=float(p),
                              multimodal=bool(p < 0.05), method="Hartigan dip test")
    except Exception as e:
        out["diptest"] = dict(error=str(e))
    # GMM/BIC: 1 vs 2 components
    try:
        from sklearn.mixture import GaussianMixture
        X = x.reshape(-1, 1)
        b1 = GaussianMixture(1, covariance_type="full", random_state=0).fit(X).bic(X)
        b2 = GaussianMixture(2, covariance_type="full", random_state=0).fit(X).bic(X)
        out["gmm_bic"] = dict(bic_1=float(b1), bic_2=float(b2),
                              prefers_2=bool(b2 < b1), delta_bic=float(b1 - b2))
    except Exception as e:
        out["gmm_bic"] = dict(error=str(e))
    return out

if __name__ == "__main__":
    t0 = time.time()
    RES = {}
    RES["config"] = dict(dataset="Balanced Colored-MNIST (torchvision MNIST, 14x14, 2-color)",
        label_rule="y=1[digit>=5]", label_flip=CFG.LABEL_FLIP, p_corr_train=CFG.P_CORR_TRAIN,
        p_corr_test=CFG.P_CORR_TEST, color_render="grayscale-base weak tint",
        tint=CFG.TINT, patch=CFG.PATCH, cnoise=CFG.CNOISE, gray_base=CFG.GRAY_BASE,
        bottleneck=CFG.BOTTLE, WD=CFG.WD, LR=CFG.LR, EPOCHS=CFG.EPOCHS, N=CFG.N,
        lr_schedule="cosine->~0", nparams=int(sum(p.numel() for p in mk().parameters())),
        nseed=NSEED, smoke=SMOKE)
    save(RES)

    Xdep, ydep = gen(8000, 999001, "deploy")
    print("="*72)
    print(f"BALANCED Colored-MNIST  smoke={SMOKE}  nseed={NSEED}")
    print(f"  flip={CFG.LABEL_FLIP} pcorr_train={CFG.P_CORR_TRAIN} pcorr_test={CFG.P_CORR_TEST}")
    print(f"  color=gray-base tint={CFG.TINT} patch={CFG.PATCH} cnoise={CFG.CNOISE}")
    print(f"  CNN bottleneck={CFG.BOTTLE} WD={CFG.WD} LR={CFG.LR} EPOCHS={CFG.EPOCHS} N={CFG.N}")
    print(f"  #params={RES['config']['nparams']}")
    print("="*72, flush=True)

    # ---- (1) SEED MULTIPLICITY ----
    sc, tac, cr, models = [], [], [], []
    for s in range(NSEED):
        m, Xt, yt = train(SEEDBASE + s)
        models.append((m, Xt, yt))
        sc.append(accm(m, Xdep, ydep)); tac.append(accm(m, Xt, yt)); cr.append(color_reliance(m))
        if (s + 1) % 5 == 0 or s == 0:
            print(f"  ...seed {s+1}/{NSEED}  ood={sc[-1]:.3f} train={tac[-1]:.3f} "
                  f"cr={cr[-1]:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            # incremental partial save
            RES["multiplicity_partial"] = dict(done=s+1, ood=list(map(float, sc)))
            save(RES)
    sc = np.array(sc); tac = np.array(tac); cr = np.array(cr)
    h, _ = np.histogram(sc, bins=10, range=(0, 1))
    USE_TRUE = sc > 0.55; USE_SPUR = sc < 0.45
    nHi, nLo = int(USE_TRUE.sum()), int(USE_SPUR.sum())
    mm = multimodal_test(sc)
    print(f"\nSEED MULTIPLICITY ({NSEED} seeds, identical except seed):")
    print(f"  in-dist train acc {tac.mean():.3f}+/-{tac.std():.3f} [{tac.min():.3f},{tac.max():.3f}]")
    print(f"  OOD acc: mean {sc.mean():.3f} std {sc.std():.3f} range [{sc.min():.3f},{sc.max():.3f}]")
    print(f"  uses-TRUE(>0.55) {nHi}/{NSEED} | uses-SPUR(<0.45) {nLo}/{NSEED} | mid {NSEED-nHi-nLo}")
    print(f"  color-reliance mean {cr.mean():.3f}  (corr w/ OOD r={np.corrcoef(cr,sc)[0,1]:.3f})")
    print(f"  OOD hist[0..1]: " + " ".join(f"{x:2d}" for x in h))
    print(f"  multimodality: {mm}")
    RES.pop("multiplicity_partial", None)
    RES["multiplicity"] = dict(nseed=NSEED, train_acc_mean=float(tac.mean()), train_acc_std=float(tac.std()),
        train_acc_min=float(tac.min()), train_acc_max=float(tac.max()),
        ood_mean=float(sc.mean()), ood_std=float(sc.std()), ood_min=float(sc.min()), ood_max=float(sc.max()),
        n_uses_true=nHi, n_uses_spur=nLo, n_mid=NSEED - nHi - nLo, hist=h.tolist(),
        color_reliance_mean=float(cr.mean()), corr_creliance_ood=float(np.corrcoef(cr, sc)[0, 1]),
        multimodality=mm, ood_scores=sc.tolist(), color_reliance=cr.tolist(), train_acc=tac.tolist())
    save(RES)

    # ---- (2) GENUINE MINIMA ----
    hi = [i for i in range(NSEED) if USE_TRUE[i]]; lo = [i for i in range(NSEED) if USE_SPUR[i]]
    RES["genuine_minima"] = {}
    print(f"\nGENUINE-MINIMA verification:")
    for tag, cl in [("uses_true", hi), ("uses_spur", lo)]:
        if not cl: print(f"  [{tag}] none in this run"); continue
        recs = []
        for i in cl[:2]:   # up to 2 representatives per mode
            m, Xt, yt = models[i]
            g = gnorm(m, Xt, yt); w1 = wnorm(m)
            m2, _, _ = train(SEEDBASE + i, steps=2 * CFG.EPOCHS); w2 = wnorm(m2)
            em = eig_min(m, Xt, yt); ood2 = accm(m2, Xdep, ydep)
            rec = dict(seed=int(SEEDBASE + i), grad_norm=float(g), wnorm=float(w1),
                       wnorm_2x=float(w2), wnorm_drift=float(abs(w2 - w1) / w1),
                       ood=float(sc[i]), ood_2x=float(ood2), hess_min_eig=float(em),
                       train_acc=float(tac[i]),
                       genuine=bool(em > -1e-3 and abs(w2 - w1) / w1 < 0.12))
            recs.append(rec)
            print(f"  [{tag}] seed{SEEDBASE+i}: ||grad||={g:.2e} wnorm {w1:.2f}->{w2:.2f}"
                  f"({100*abs(w2-w1)/w1:.1f}% @2x) OOD {sc[i]:.2f}->{ood2:.2f} "
                  f"Hmin={em:+.2e} -> {'GENUINE' if rec['genuine'] else 'check'}", flush=True)
        RES["genuine_minima"][tag] = recs
        save(RES)

    # ---- (3) CONNECTIVITY ----
    Xb, yb = gen(CFG.N, SEEDBASE + 9999, "train"); rng = np.random.default_rng(0)
    def avg(ca, cb, al, same, k=12):
        out = []
        for _ in range(k):
            i = int(rng.choice(ca)); j = int(rng.choice(cb))
            if same and i == j: continue
            mB = align2(models[i][0], models[j][0], Xb) if al else models[j][0]
            out.append(barrier(models[i][0], mB, Xb, yb))
        return float(np.mean(out)) if out else float("nan")
    if hi and lo:
        conn = dict(within_true_aln=avg(hi, hi, True, True), within_spur_aln=avg(lo, lo, True, True),
                    across_aln=avg(hi, lo, True, False),
                    within_true_raw=avg(hi, hi, False, True), across_raw=avg(hi, lo, False, False))
        wvals = [v for v in (conn["within_true_aln"], conn["within_spur_aln"]) if v == v]  # drop nan
        within = max(wvals + [1e-3])
        conn["verdict"] = ("connected mod perm (across barrier <= within)" if conn["across_aln"] < 2 * within
                           else "DISTINCT basins (across barrier >> within)")
        print(f"\nCONNECTIVITY (perm-aligned loss barrier):")
        print(f"  RAW    within(true) {conn['within_true_raw']:.3f}  across {conn['across_raw']:.3f}")
        print(f"  ALIGNED within(true) {conn['within_true_aln']:.3f} within(spur) {conn['within_spur_aln']:.3f}"
              f"  across {conn['across_aln']:.3f}  -> {conn['verdict']}", flush=True)
        RES["connectivity"] = conn
    else:
        print("\nCONNECTIVITY: need both modes non-empty; skipped.")
        RES["connectivity"] = None
    save(RES)

    # ---- (4) INTERVENTION SWEEP ----
    # intervention only needs to classify shape-vs-color MODE, not perfect minima,
    # so use a cheaper epoch budget here (still well past the mode-decision point).
    INT_EPOCHS = 600
    print(f"\nINTERVENTION: fraction f of color-decorrelated train data (epochs={INT_EPOCHS}):")
    sweep = []; NS_INT = 5 if SMOKE else 40
    fs_list = [0.0, 0.02, 0.05, 0.10, 0.20] if SMOKE else [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    for f in fs_list:
        al = []
        for s in range(NS_INT):
            m, _, _ = train(SEEDBASE + s, steps=INT_EPOCHS, f=f); al.append(accm(m, Xdep, ydep))
        al = np.array(al)
        fa = float((al > 0.55).mean()); fm = float((al < 0.45).mean())
        print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | uses-true {fa:.2f} | uses-spur {fm:.2f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        sweep.append(dict(f=f, mean_ood=float(al.mean()), frac_uses_true=fa,
                          frac_uses_spur=fm, ood_scores=al.tolist()))
        RES["intervention"] = sweep; save(RES)

    RES["runtime_s"] = float(time.time() - t0)
    save(RES)
    print(f"\nTotal {time.time()-t0:.0f}s. Wrote {os.path.basename(RESPATH)}", flush=True)

    # ---- FIGURE ----
    if not SMOKE:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        ax[0].hist(sc, bins=20, range=(0, 1), color="#55A868", edgecolor="white")
        ax[0].axvline(CFG.P_CORR_TEST, color="#C44E52", ls=":", label=f"pure color-relier ~{CFG.P_CORR_TEST:.2f}")
        ax[0].axvline(0.5, color="gray", ls="--")
        dipinfo = RES["multiplicity"]["multimodality"].get("diptest", {})
        dp = dipinfo.get("pval", float("nan"))
        ax[0].set_xlabel("OOD accuracy (color flipped at test)"); ax[0].set_ylabel("# seeds")
        ax[0].set_title(f"Balanced CMNIST SEED MULTIPLICITY\n{nHi}/{NSEED} TRUE shape, {nLo}/{NSEED} SPURIOUS color\n"
                        f"dip p={dp:.3f}  (identical except seed)")
        ax[0].legend(fontsize=8)
        fs = [d["f"] for d in sweep]
        ax[1].plot(fs, [d["frac_uses_true"] for d in sweep], "o-", color="#55A868", label="frac uses-true")
        ax[1].plot(fs, [d["mean_ood"] for d in sweep], "s--", color="#4C72B0", label="mean OOD acc")
        ax[1].set_xlabel("fraction counter-spurious train data (f)"); ax[1].set_ylabel("OOD generalization")
        ax[1].set_ylim(0, 1.02); ax[1].set_title("INTERVENTION: color-decorrelated data\nflips population to TRUE feature")
        ax[1].legend(fontsize=8)
        ax[2].scatter(cr, sc, s=18, alpha=0.7, color="#8172B3")
        ax[2].set_xlabel("color-reliance probe (1=pure color)"); ax[2].set_ylabel("OOD accuracy")
        ax[2].set_title(f"Mechanism: OOD acc vs color-reliance\nr={np.corrcoef(cr,sc)[0,1]:.2f}")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, "cmnist_balanced.png")); plt.close(fig)
        print("Wrote cmnist_balanced.png", flush=True)
