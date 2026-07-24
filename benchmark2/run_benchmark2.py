"""
SECOND recognized spurious-correlation benchmark: Patch/Watermark-Fashion-MNIST.
Equal-cost regime -> seed-driven aligned(true-shape) vs misaligned(spurious-patch)
multiplicity at genuine minima. Mirrors ../realdata/cmnist.py and ../alignment.py.

Rigor:
  - >=50 seeds; OOD-accuracy distribution (mean/std/range/hist + fraction true vs spurious)
  - multimodality test: Hartigan dip test (diptest) AND a GMM-BIC (1 vs 2 components) check
  - genuine finite minima: weight decay; full-batch grad norm; 2x-train weight-norm stability;
    Hessian min-eig via HVP power iteration
  - permutation-aligned loss barrier within-cluster vs across-cluster (mode connectivity)
  - intervention sweep: fraction of counter-spurious (patch-decorrelated) train data -> flip?

Run:  python3 -u run_benchmark2.py [--smoke]   (writes results.json incrementally + benchmark2.png)
"""
import sys, os, time, json, numpy as np, torch
import patch_fmnist as P

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))  # good-neighbor: machine is shared

# allow equal-cost config override from environment (set by the split scan winner)
for k in ("P_CORR_TRAIN", "P_CORR_TEST", "PATCH_AMP", "SHAPE_NOISE", "PATCH_NOISE"):
    if k in os.environ:
        setattr(P, k, float(os.environ[k]))

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = "--smoke" in sys.argv
NSEED   = 5 if SMOKE else 60
NS_INT  = 5 if SMOKE else 40
SEEDBASE = 4000
RESFILE = os.path.join(HERE, "results_smoke.json" if SMOKE else "results.json")

def save(RES):
    json.dump(RES, open(RESFILE, "w"), indent=2)

def dip_pvalue(x):
    try:
        import diptest
        d, p = diptest.diptest(np.asarray(x, dtype=float))
        return float(d), float(p)
    except Exception as e:
        return None, None

def gmm_bic(x):
    """BIC for 1- vs 2-component GMM. Returns (bic1, bic2, prefer_2comp)."""
    try:
        from sklearn.mixture import GaussianMixture
        X = np.asarray(x, dtype=float).reshape(-1, 1)
        g1 = GaussianMixture(1, n_init=3, random_state=0).fit(X)
        g2 = GaussianMixture(2, n_init=3, random_state=0).fit(X)
        b1, b2 = g1.bic(X), g2.bic(X)
        return float(b1), float(b2), bool(b2 < b1)
    except Exception:
        return None, None, None

def main():
    t0 = time.time()
    RES = {}
    cfg = dict(dataset="Patch/Watermark-Fashion-MNIST (torchvision FashionMNIST, 14x14, 1ch + corner patch)",
               label_rule="y=1[class>=5]", label_flip=P.LABEL_FLIP,
               p_corr_train=P.P_CORR_TRAIN, p_corr_test=P.P_CORR_TEST,
               patch_amp=P.PATCH_AMP, patch_noise=P.PATCH_NOISE, shape_noise=P.SHAPE_NOISE,
               patch_sz=P.PATCH_SZ, bottleneck=P.BOTTLE, WD=P.WD, LR=P.LR, EPOCHS=P.EPOCHS, N=P.N,
               nparams=int(sum(p.numel() for p in P.mk().parameters())), nseed=NSEED, smoke=SMOKE)
    RES["config"] = cfg; save(RES)
    print(f"{'='*72}\nPatch-Fashion-MNIST  ({'SMOKE ' if SMOKE else ''}NSEED={NSEED})")
    print(f"  y=1[class>=5], label_flip={P.LABEL_FLIP}, patch corr train={P.P_CORR_TRAIN} OOD={P.P_CORR_TEST}")
    print(f"  patch_amp={P.PATCH_AMP} patch_noise={P.PATCH_NOISE} shape_noise={P.SHAPE_NOISE} patch_sz={P.PATCH_SZ}")
    print(f"  model tiny CNN bottleneck={P.BOTTLE} WD={P.WD} LR={P.LR} EPOCHS={P.EPOCHS} N={P.N} #params={cfg['nparams']}")
    print('='*72, flush=True)

    # large fixed OOD test set + sanity baselines (shape-only / patch-only ceilings)
    Xdep, ydep = P.gen(8000, 999001, "deploy")

    # ---- (b) SEED MULTIPLICITY ----
    sc = []; tac = []; pr = []; models = []
    for s in range(NSEED):
        m, Xt, yt = P.train(SEEDBASE + s)
        models.append((m, Xt, yt))
        sc.append(P.accm(m, Xdep, ydep)); tac.append(P.accm(m, Xt, yt)); pr.append(P.patch_reliance(m))
        if (s + 1) % 5 == 0:
            print(f"  ...{s+1}/{NSEED} seeds  (last OOD {sc[-1]:.3f}, patch-rel {pr[-1]:.2f})  {time.time()-t0:.0f}s", flush=True)
    sc = np.array(sc); tac = np.array(tac); pr = np.array(pr)
    h, _ = np.histogram(sc, bins=10, range=(0, 1))
    USE_TRUE = sc > 0.55; USE_SPUR = sc < 0.45
    nHi = int(USE_TRUE.sum()); nLo = int(USE_SPUR.sum())
    dipstat, dipp = dip_pvalue(sc)
    bic1, bic2, prefer2 = gmm_bic(sc)
    print(f"\nSEED MULTIPLICITY ({NSEED} seeds, identical except seed):")
    print(f"  in-dist train acc {tac.mean():.3f}+/-{tac.std():.3f}  range [{tac.min():.3f},{tac.max():.3f}]")
    print(f"  OOD acc: mean {sc.mean():.3f} std {sc.std():.3f} range [{sc.min():.3f},{sc.max():.3f}]")
    print(f"  uses-TRUE(>0.55) {nHi}/{NSEED} | uses-SPUR(<0.45) {nLo}/{NSEED} | mid {NSEED-nHi-nLo}")
    print(f"  patch-reliance probe: mean {pr.mean():.3f}  corr(patch_rel,OOD) r={np.corrcoef(pr,sc)[0,1]:.3f}")
    print(f"  multimodality: diptest stat={dipstat} p={dipp} | GMM BIC1={bic1} BIC2={bic2} prefer2comp={prefer2}")
    print(f"  OOD hist[0..1]: " + " ".join(f"{x:2d}" for x in h), flush=True)
    RES["multiplicity"] = dict(nseed=NSEED, train_acc_mean=float(tac.mean()), train_acc_std=float(tac.std()),
        train_acc_min=float(tac.min()), train_acc_max=float(tac.max()),
        ood_mean=float(sc.mean()), ood_std=float(sc.std()), ood_min=float(sc.min()), ood_max=float(sc.max()),
        n_uses_true=nHi, n_uses_spur=nLo, n_mid=NSEED - nHi - nLo, hist=h.tolist(),
        patch_reliance_mean=float(pr.mean()), corr_preliance_ood=float(np.corrcoef(pr, sc)[0, 1]),
        dip_stat=dipstat, dip_p=dipp, gmm_bic1=bic1, gmm_bic2=bic2, gmm_prefer_2comp=prefer2,
        ood_scores=sc.tolist(), patch_reliance=pr.tolist(), train_acc=tac.tolist())
    save(RES)

    # ---- (d) GENUINE MINIMA ----
    hi = [i for i in range(NSEED) if USE_TRUE[i]]; lo = [i for i in range(NSEED) if USE_SPUR[i]]
    RES["genuine_minima"] = {}
    print(f"\nGENUINE-MINIMA verification (representative seeds):")
    for tag, cl in [("uses_true", hi), ("uses_spur", lo)]:
        if not cl: print(f"  [{tag}] none in this run"); continue
        i = cl[0]; m, Xt, yt = models[i]
        g = P.gnorm(m, Xt, yt); w1 = P.wnorm(m)
        m2, _, _ = P.train(SEEDBASE + i, steps=2 * P.EPOCHS); w2 = P.wnorm(m2)
        em = P.eig_min(m, Xt, yt)
        print(f"  [{tag}] seed{SEEDBASE+i}: ||grad||={g:.2e}  wnorm {w1:.2f}->{w2:.2f}(2x)  "
              f"OOD {sc[i]:.2f}->{P.accm(m2,Xdep,ydep):.2f}  Hess_min={em:+.2e} -> "
              f"{'GENUINE MIN' if em > -1e-3 else 'saddle?'}", flush=True)
        RES["genuine_minima"][tag] = dict(seed=int(SEEDBASE + i), grad_norm=float(g), wnorm=float(w1),
            wnorm_2x=float(w2), ood=float(sc[i]), ood_2x=float(P.accm(m2, Xdep, ydep)),
            hess_min_eig=float(em), train_acc=float(tac[i]))
    save(RES)

    # ---- (e) CONNECTIVITY ----
    Xb, yb = P.gen(P.N, SEEDBASE + 99, "train"); rng = np.random.default_rng(0)
    def avg(ca, cb, al, same, k=15):
        out = []
        for _ in range(k):
            i = int(rng.choice(ca)); j = int(rng.choice(cb))
            if same and i == j: continue
            mB = P.align2(models[i][0], models[j][0], Xb) if al else models[j][0]
            out.append(P.barrier(models[i][0], mB, Xb, yb))
        return float(np.mean(out)) if out else float('nan')
    if hi and lo:
        conn = dict(within_true_aln=avg(hi, hi, True, True), across_aln=avg(hi, lo, True, False),
                    within_true_raw=avg(hi, hi, False, True), across_raw=avg(hi, lo, False, False))
        verdict = 'connected mod perm' if conn['across_aln'] < 2 * max(conn['within_true_aln'], 1e-3) else 'DISTINCT basins'
        conn["verdict"] = verdict
        print(f"\nCONNECTIVITY (loss barrier on train data):")
        print(f"  RAW    : within(true) {conn['within_true_raw']:.3f}  across(true,spur) {conn['across_raw']:.3f}")
        print(f"  ALIGNED: within(true) {conn['within_true_aln']:.3f}  across(true,spur) {conn['across_aln']:.3f}  -> {verdict}", flush=True)
        RES["connectivity"] = conn
    else:
        print("\nCONNECTIVITY: need both clusters non-empty; skipped.")
        RES["connectivity"] = None
    save(RES)

    # ---- (c) INTERVENTION sweep ----
    print(f"\nINTERVENTION: fraction f of counter-spurious (patch-decorrelated) train data:")
    sweep = []
    for f in [0.0, 0.01, 0.02, 0.05, 0.10]:
        al = []
        for s in range(NS_INT):
            m, _, _ = P.train(SEEDBASE + s, f=f); al.append(P.accm(m, Xdep, ydep))
        al = np.array(al)
        fa = float((al > 0.55).mean()); fm = float((al < 0.45).mean())
        print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | frac uses-true(>0.55) {fa:.2f} | uses-spur(<0.45) {fm:.2f}  {time.time()-t0:.0f}s", flush=True)
        sweep.append(dict(f=f, mean_ood=float(al.mean()), frac_uses_true=fa, frac_uses_spur=fm, ood_scores=al.tolist()))
    RES["intervention"] = sweep; save(RES)

    RES["runtime_sec"] = time.time() - t0; save(RES)
    print(f"\nTotal {time.time()-t0:.0f}s. Wrote {os.path.basename(RESFILE)}", flush=True)

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(sc, bins=20, range=(0, 1), color="#55A868", edgecolor="white")
    ax[0].axvline(P.P_CORR_TEST, color="#C44E52", ls=":", label=f"pure patch-relier ~{1-P.P_CORR_TRAIN:.2f}")
    ax[0].axvline(0.5, color="gray", ls="--")
    ax[0].set_xlabel("OOD accuracy (patch flipped at test)"); ax[0].set_ylabel("# seeds")
    ax[0].set_title(f"Patch-FashionMNIST SEED MULTIPLICITY\n{nHi}/{NSEED} use TRUE shape, {nLo}/{NSEED} use SPURIOUS patch\n"
                    f"dip p={dipp if dipp is None else round(dipp,3)}  GMM prefer-2={prefer2}")
    ax[0].legend(fontsize=8)
    fs = [d["f"] for d in sweep]
    ax[1].plot(fs, [d["frac_uses_true"] for d in sweep], "o-", color="#55A868", label="frac uses-true (OOD>0.55)")
    ax[1].plot(fs, [d["mean_ood"] for d in sweep], "s--", color="#4C72B0", label="mean OOD acc")
    ax[1].set_xlabel("fraction counter-spurious train data (f)"); ax[1].set_ylabel("OOD generalization")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("INTERVENTION: a little patch-decorrelated\ndata flips population to TRUE feature")
    ax[1].legend(fontsize=8)
    ax[2].scatter(pr, sc, s=18, alpha=0.7, color="#8172B3")
    ax[2].set_xlabel("patch-reliance probe (1=pure patch)"); ax[2].set_ylabel("OOD accuracy")
    ax[2].set_title(f"Mechanism check: OOD acc vs patch-reliance\nr={np.corrcoef(pr,sc)[0,1]:.2f} (patch use -> OOD failure)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "benchmark2_smoke.png" if SMOKE else "benchmark2.png")); plt.close(fig)
    print("Wrote figure.", flush=True)

if __name__ == "__main__":
    main()
