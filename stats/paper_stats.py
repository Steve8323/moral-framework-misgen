"""
Paper-grade statistics for the genuine-minima goal-misgeneralization setup.

Built ON TOP OF the established recipe in ../alignment.py (read, not modified):
  equal-cost competing XOR cues (aligned objective vs misaligned proxy that flips
  in deployment) + capacity bottleneck + weight decay  =>  genuine finite minima
  with seed-driven multiplicity.

Five deliverables (numbers over prose):
  1. MULTIMODALITY  : Hartigan dip test (p) + 1/2/3-component GMM BIC on >=100 seeds.
  2. THRESHOLD      : fine f-grid 0..3% + robustness over margin{1,2,3} x noise{.3,.6,1} x width{4,8,16}.
  3. BASIN CENSUS   : full pairwise permutation-aligned barrier matrix (>=60 seeds),
                      connected-component clustering -> # genuinely-separated basins;
                      within- vs across-cluster barrier stats + permutation p-value.
  4. GENUINE MINIMA : Hessian min-eig (HVP power iteration) for >=10 seeds spanning
                      aligned/misaligned; grad norms; weight-norm stability under 2x train.
  5. PERSISTENCE    : misaligned-by-chance models given a short "safety fine-tune" on the
                      ORIGINAL train distribution (aligned==misaligned there). Does the
                      misalignment persist?  (Sleeper-Agents / Hubinger et al. 2024.)

All numbers real. Weight decay -> finite minima. Negatives reported honestly.
CIs via seed-resampled (cluster) bootstrap where an effect is stated.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy import stats as sstats
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
import diptest

torch.set_num_threads(8)
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- config (matches alignment.py defaults) ----
NBITS = 2; NNOISE = 3; INPUT = NBITS + NBITS + NNOISE
MARGIN = 2.0; NOISE = 0.6; N = 800
H = 8; WD = 2e-3; LR = 5e-3; EPOCHS = 2000
SEEDBASE = 700                # same family as alignment.py ALIGN study
RNG_GLOBAL = np.random.default_rng(12345)


# ----------------------------------------------------------------------------
# data + model (faithful to alignment.py, parameterized for the robustness grid)
# ----------------------------------------------------------------------------
def xor_block(target, rng, margin, noise):
    n = len(target); b = rng.integers(0, 2, (n, NBITS))
    b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (margin * (2 * b - 1)).astype(np.float32) + noise * rng.standard_normal((n, NBITS)).astype(np.float32)


def gen_align(n, seed, mode, f=0.0, margin=MARGIN, noise=NOISE):
    """cue1=true objective(v); cue2=proxy(a). train: a=v except fraction f flipped
    (anti-proxy intervention). deploy: a=1-v. label=v always."""
    rng = np.random.default_rng(seed); v = rng.integers(0, 2, n)
    if mode == "deploy":
        a = 1 - v
    else:
        a = v.copy()
        if f > 0:
            m = rng.random(n) < f; a[m] = 1 - v[m]
    Xt = xor_block(v, rng, margin, noise); Xp = xor_block(a, rng, margin, noise)
    Z = noise * rng.standard_normal((n, NNOISE)).astype(np.float32)
    return (torch.tensor(np.concatenate([Xt, Xp, Z], 1).astype(np.float32)),
            torch.tensor(v.astype(np.int64)))


def mk(width=H):
    return nn.Sequential(nn.Linear(INPUT, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, 2))


def accm(m, X, y):
    with torch.no_grad():
        return (m(X).argmax(1) == y).float().mean().item()


def train(seed, steps=EPOCHS, f=0.0, margin=MARGIN, noise=NOISE, width=H,
          init_state=None, wd=WD):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_align(N, seed, "train", f=f, margin=margin, noise=noise)
    m = mk(width)
    if init_state is not None:
        m.load_state_dict(init_state)
    lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


# ---- diagnostics (faithful to alignment.py) ----
def gnorm(m, X, y):
    lf = nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X), y).backward()
    g = torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())).item()
    m.zero_grad(); return g


def wnorm(m):
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in m.parameters())).item()


def hvp(m, X, y, v):
    lf = nn.CrossEntropyLoss(); ps = list(m.parameters()); loss = lf(m(X), y)
    g = torch.autograd.grad(loss, ps, create_graph=True)
    fg = torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg * v).sum(), ps)]).detach()


def eig_min(m, X, y, it=50):
    npm = sum(p.numel() for p in m.parameters()); v = torch.randn(npm); v /= v.norm()
    for _ in range(it):
        h = hvp(m, X, y, v); v = h / (h.norm() + 1e-12)
    top = (v * hvp(m, X, y, v)).sum().item(); sh = abs(top) + 1.0
    w = torch.randn(npm); w /= w.norm()
    for _ in range(it):
        hw = sh * w - hvp(m, X, y, w); w = hw / (hw.norm() + 1e-12)
    return sh - (w * (sh * w - hvp(m, X, y, w))).sum().item()


def acts(m, X):
    with torch.no_grad():
        h1 = torch.relu(m[0](X)); return h1.numpy(), torch.relu(m[2](h1)).numpy()


def align2(mA, mB, Xref, width=H):
    """permutation-align both hidden layers of mB onto mA (2-hidden-layer)."""
    h1A, h2A = acts(mA, Xref); h1B, h2B = acts(mB, Xref)

    def perm(A, B):
        A = (A - A.mean(0)) / (A.std(0) + 1e-6); B = (B - B.mean(0)) / (B.std(0) + 1e-6)
        _, c = linear_sum_assignment(-(A.T @ B) / A.shape[0]); return c

    c1 = perm(h1A, h1B); c2 = perm(h2A, h2B)
    s = mB.state_dict(); o = {k: v.clone() for k, v in s.items()}
    o["0.weight"] = s["0.weight"][c1]; o["0.bias"] = s["0.bias"][c1]
    o["2.weight"] = s["2.weight"][c2][:, c1]; o["2.bias"] = s["2.bias"][c2]
    o["4.weight"] = s["4.weight"][:, c2]
    mC = mk(width); mC.load_state_dict(o); return mC


def barrier(mA, mB, X, y, npts=11, width=H):
    lf = nn.CrossEntropyLoss(); sA, sB = mA.state_dict(), mB.state_dict()
    m = mk(width); ls = []
    for a in np.linspace(0, 1, npts):
        m.load_state_dict({k: (1 - a) * sA[k] + a * sB[k] for k in sA})
        with torch.no_grad():
            ls.append(lf(m(X), y).item())
    ls = np.array(ls); return float(ls.max() - max(ls[0], ls[-1]))


def boot_ci(x, fn=np.mean, B=2000, seed=0):
    """seed-resampled bootstrap CI for a scalar statistic over a 1-D sample."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    bs = [fn(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


# ============================================================================
# 1. MULTIMODALITY  (>=100 seeds): dip test + GMM BIC
# ============================================================================
def deliverable_1_multimodality(nseed=120):
    print(f"\n{'='*72}\n1. MULTIMODALITY  ({nseed} seeds)\n{'='*72}")
    t0 = time.time()
    Xdep, ydep = gen_align(4000, SEEDBASE + 50001, "deploy")
    scores = []; train_acc = []; models = []
    for s in range(nseed):
        m, Xt, yt = train(SEEDBASE + s)
        scores.append(accm(m, Xdep, ydep)); train_acc.append(accm(m, Xt, yt))
        models.append((m, Xt, yt))
    scores = np.array(scores); train_acc = np.array(train_acc)

    # Hartigan dip test for unimodality
    dip, p_dip = diptest.diptest(scores)

    # GMM 1 vs 2 vs 3 components by BIC (reshape to column)
    Xg = scores.reshape(-1, 1)
    gmm_rows = []
    best = None
    for k in (1, 2, 3):
        # several restarts for stability
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=10, random_state=0, reg_covar=1e-4).fit(Xg)
        bic = gm.bic(Xg); aic = gm.aic(Xg)
        means = sorted(gm.means_.ravel().tolist())
        gmm_rows.append(dict(k=k, bic=float(bic), aic=float(aic),
                             means=means,
                             weights=sorted(gm.weights_.tolist())))
        if best is None or bic < best[1]:
            best = (k, bic)
    best_k = best[0]
    dbic = gmm_rows[0]["bic"] - gmm_rows[best_k - 1]["bic"]  # 1-comp minus best

    nHi = int((scores > 0.8).sum()); nLo = int((scores < 0.2).sum())
    nMid = nseed - nHi - nLo
    verdict_dip = "MULTIMODAL/non-unimodal (reject unimodality)" if p_dip < 0.05 else "unimodal not rejected by dip"
    verdict_gmm = (f"{best_k}-component favored by BIC (dBIC vs 1-comp = {dbic:.1f})"
                   if best_k > 1 else "1-component (unimodal) favored by BIC")

    print(f"  train acc {train_acc.mean():.3f}+/-{train_acc.std():.3f}")
    print(f"  deploy-alignment: mean {scores.mean():.3f} std {scores.std():.3f} "
          f"range [{scores.min():.3f},{scores.max():.3f}]")
    print(f"  high(>0.8) {nHi}/{nseed} | low(<0.2) {nLo}/{nseed} | mid {nMid}/{nseed}")
    print(f"  DIP statistic={dip:.4f}  p={p_dip:.4g}  -> {verdict_dip}")
    for r in gmm_rows:
        print(f"  GMM k={r['k']}: BIC={r['bic']:8.2f} AIC={r['aic']:8.2f} "
              f"means={[round(x,3) for x in r['means']]} weights={[round(x,2) for x in r['weights']]}")
    print(f"  GMM verdict: {verdict_gmm}")
    print(f"  [t={time.time()-t0:.0f}s]")

    return dict(nseed=nseed, scores=scores.tolist(), train_acc=float(train_acc.mean()),
                mean=float(scores.mean()), std=float(scores.std()),
                min=float(scores.min()), max=float(scores.max()),
                nHi=nHi, nLo=nLo, nMid=nMid,
                dip_stat=float(dip), dip_p=float(p_dip), dip_verdict=verdict_dip,
                gmm=gmm_rows, gmm_best_k=best_k, gmm_dBIC_vs1=float(dbic),
                gmm_verdict=verdict_gmm), models, scores


# ============================================================================
# 2. INTERVENTION THRESHOLD  (fine f-grid) + ROBUSTNESS GRID
# ============================================================================
def _frac_aligned(f, margin, noise, width, nseed, seed0):
    Xdep, ydep = gen_align(4000, SEEDBASE + 50001, "deploy", margin=margin, noise=noise)
    al = []
    for s in range(nseed):
        m, _, _ = train(seed0 + s, f=f, margin=margin, noise=noise, width=width)
        al.append(accm(m, Xdep, ydep))
    al = np.array(al)
    return al


def deliverable_2_threshold(fine_nseed=40, robust_nseed=24):
    print(f"\n{'='*72}\n2. INTERVENTION THRESHOLD + ROBUSTNESS\n{'='*72}")
    t0 = time.time()

    # ---- fine f-grid 0..3% at the canonical config ----
    fgrid = [0.0, 0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.025, 0.030]
    fine = []
    print(f"  fine f-grid @ margin={MARGIN} noise={NOISE} width={H} ({fine_nseed} seeds/cell):")
    for f in fgrid:
        al = _frac_aligned(f, MARGIN, NOISE, H, fine_nseed, SEEDBASE)
        fa = float((al > 0.8).mean()); fm = float((al < 0.2).mean())
        lo, hi = boot_ci((al > 0.8).astype(float), seed=1)
        fine.append(dict(f=f, mean_align=float(al.mean()), frac_aligned=fa,
                         frac_misaligned=fm, ci_aligned=[lo, hi]))
        print(f"    f={f*100:5.2f}% | mean {al.mean():.3f} | aligned {fa:.2f} "
              f"[{lo:.2f},{hi:.2f}] | misaligned {fm:.2f}")

    # threshold f* = smallest f with frac_aligned >= 0.9 (interp)
    fstar = None
    for d in fine:
        if d["frac_aligned"] >= 0.9:
            fstar = d["f"]; break
    print(f"  threshold f* (>=90% aligned) = {None if fstar is None else f'{fstar*100:.2f}%'}")

    # ---- robustness grid: margin x noise x width, find f* in each cell ----
    margins = [1.0, 2.0, 3.0]; noises = [0.3, 0.6, 1.0]; widths = [4, 8, 16]
    grid_f = [0.0, 0.005, 0.010, 0.015, 0.020, 0.030]
    print(f"  robustness grid ({robust_nseed} seeds/cell) -> f* (>=90% aligned), '-' = never reached:")
    robust = []
    seed0 = SEEDBASE + 1000
    for margin in margins:
        for noise in noises:
            for width in widths:
                fa_by_f = []
                fstar_c = None; baseline_fa = None
                for f in grid_f:
                    al = _frac_aligned(f, margin, noise, width, robust_nseed, seed0)
                    fa = float((al > 0.8).mean())
                    fa_by_f.append(fa)
                    if f == 0.0:
                        baseline_fa = fa
                    if fstar_c is None and fa >= 0.9:
                        fstar_c = f
                        break  # found threshold for this cell; no need to climb higher
                robust.append(dict(margin=margin, noise=noise, width=width,
                                   baseline_frac_aligned=baseline_fa,
                                   fstar=fstar_c, fa_by_f=fa_by_f))
                fs = "-" if fstar_c is None else f"{fstar_c*100:.1f}%"
                print(f"    m={margin} noise={noise} w={width:2d} | base_aligned={baseline_fa:.2f} "
                      f"| f*={fs} | fa_by_f={[round(x,2) for x in fa_by_f]}")
    fstars = [r["fstar"] for r in robust if r["fstar"] is not None]
    print(f"  reached >=90% in {len(fstars)}/{len(robust)} cells; "
          f"f* range = [{min(fstars)*100:.1f}%,{max(fstars)*100:.1f}%]" if fstars else "  no cell reached 90%")
    print(f"  [t={time.time()-t0:.0f}s]")
    return dict(fgrid=fine, fstar=fstar, robust=robust,
                grid_f=grid_f, margins=margins, noises=noises, widths=widths)


# ============================================================================
# 3. DISTINCT-BASIN CENSUS (full pairwise permutation-aligned barrier matrix)
# ============================================================================
def deliverable_3_basins(models, scores, nseed=60):
    print(f"\n{'='*72}\n3. DISTINCT-BASIN CENSUS  ({nseed} seeds)\n{'='*72}")
    t0 = time.time()
    models = models[:nseed]; scores = np.asarray(scores[:nseed])
    Xb, yb = gen_align(N, SEEDBASE + 99, "train")  # common reference batch

    # full pairwise permutation-aligned barrier matrix
    Bm = np.zeros((nseed, nseed))
    for i in range(nseed):
        for j in range(i + 1, nseed):
            mB = align2(models[i][0], models[j][0], Xb)
            b = barrier(models[i][0], mB, Xb, yb)
            Bm[i, j] = Bm[j, i] = b
    iu = np.triu_indices(nseed, 1)
    allb = Bm[iu]
    # choose connectivity threshold: models in the SAME basin merge into one min
    # under permutation; barrier ~ 0 there. Use a small absolute threshold AND a
    # data-driven knee for robustness.
    thr_abs = 0.05
    med = float(np.median(allb))
    print(f"  pairwise aligned-barrier: min {allb.min():.3f} median {med:.3f} max {allb.max():.3f}")

    def census(thr):
        adj = (Bm <= thr).astype(int); np.fill_diagonal(adj, 1)
        nc, labels = connected_components(adj, directed=False)
        sizes = np.bincount(labels).tolist()
        return nc, labels, sizes

    results_thr = {}
    for thr in [0.02, 0.05, 0.10, 0.20]:
        nc, labels, sizes = census(thr)
        results_thr[thr] = dict(n_basins=int(nc), sizes=sizes)
        print(f"  thr={thr:.2f} -> {nc} basins (component sizes {sorted(sizes, reverse=True)})")

    # primary verdict at thr_abs
    nc, labels, sizes = census(thr_abs)

    # within- vs across-cluster barrier distributions
    within, across = [], []
    for a, b in zip(*iu):
        (within if labels[a] == labels[b] else across).append(Bm[a, b])
    within = np.array(within); across = np.array(across)
    # significance: Mann-Whitney + label-permutation test on mean gap
    if len(within) and len(across):
        mw_u, mw_p = sstats.mannwhitneyu(across, within, alternative="greater")
        obs_gap = across.mean() - within.mean()
        rng = np.random.default_rng(7); perm_gaps = []
        comb = np.concatenate([within, across]); nw = len(within)
        for _ in range(5000):
            rng.shuffle(comb)
            perm_gaps.append(comb[nw:].mean() - comb[:nw].mean())
        perm_p = float((np.sum(np.array(perm_gaps) >= obs_gap) + 1) / (5001))
        wlo, whi = boot_ci(within, seed=2); alo, ahi = boot_ci(across, seed=3)
        print(f"  within-cluster barrier: mean {within.mean():.3f} [{wlo:.3f},{whi:.3f}] (n={len(within)})")
        print(f"  across-cluster barrier: mean {across.mean():.3f} [{alo:.3f},{ahi:.3f}] (n={len(across)})")
        print(f"  gap={obs_gap:.3f}  MannWhitneyU p={mw_p:.3g}  perm-test p={perm_p:.4g}")
    else:
        mw_p = perm_p = float("nan"); obs_gap = float("nan")
        wlo = whi = alo = ahi = float("nan")
        print("  only one cluster -> no across-cluster pairs")

    # relate basins to function (deploy alignment) per cluster
    cluster_align = {}
    for lab in set(labels.tolist()):
        idx = np.where(labels == lab)[0]
        cluster_align[int(lab)] = dict(size=int(len(idx)),
                                       mean_align=float(scores[idx].mean()),
                                       n_hi=int((scores[idx] > 0.8).sum()),
                                       n_lo=int((scores[idx] < 0.2).sum()))
    # also: are aligned and misaligned models in the SAME or DIFFERENT basins?
    hi_idx = np.where(scores > 0.8)[0]; lo_idx = np.where(scores < 0.2)[0]
    hi_labels = set(labels[hi_idx].tolist()); lo_labels = set(labels[lo_idx].tolist())
    shared = hi_labels & lo_labels
    print(f"  PRIMARY (thr={thr_abs}): {nc} distinct basins; "
          f"component sizes {sorted(sizes, reverse=True)}")
    print(f"  aligned-occupied basin labels {sorted(hi_labels)} | "
          f"misaligned-occupied {sorted(lo_labels)} | shared {sorted(shared)}")
    print(f"  -> aligned & misaligned models {'SHARE basins (connected mod perm within label)' if shared else 'occupy DIFFERENT basins'}")
    print(f"  [t={time.time()-t0:.0f}s]")

    return dict(nseed=nseed, barrier_matrix=Bm.tolist(),
                barrier_min=float(allb.min()), barrier_median=med, barrier_max=float(allb.max()),
                thr_primary=thr_abs, n_basins=int(nc), basin_sizes=sorted(sizes, reverse=True),
                census_by_thr={str(k): v for k, v in results_thr.items()},
                within_mean=float(within.mean()) if len(within) else None,
                within_ci=[wlo, whi], across_mean=float(across.mean()) if len(across) else None,
                across_ci=[alo, ahi], gap=float(obs_gap),
                mannwhitney_p=float(mw_p), perm_p=float(perm_p),
                cluster_align=cluster_align,
                aligned_basins=sorted(hi_labels), misaligned_basins=sorted(lo_labels),
                shared_basins=sorted(shared))


# ============================================================================
# 4. GENUINE MINIMA AT SCALE
# ============================================================================
def deliverable_4_minima(models, scores, nsample=12):
    print(f"\n{'='*72}\n4. GENUINE MINIMA AT SCALE  (Hessian min-eig, grad, weight stability)\n{'='*72}")
    t0 = time.time()
    scores = np.asarray(scores)
    Xdep, ydep = gen_align(4000, SEEDBASE + 50001, "deploy")
    # sample spanning aligned + misaligned
    hi = np.where(scores > 0.8)[0]; lo = np.where(scores < 0.2)[0]
    mid = np.where((scores >= 0.2) & (scores <= 0.8))[0]
    rng = np.random.default_rng(11)
    take = []
    for pool, k in [(hi, nsample // 2), (lo, nsample // 2), (mid, max(0, nsample - 2 * (nsample // 2)))]:
        if len(pool):
            take += rng.choice(pool, min(k, len(pool)), replace=False).tolist()
    take = sorted(set(take))[:max(nsample, 10)]
    rows = []
    for i in take:
        m, Xt, yt = models[i]
        emin = eig_min(m, Xt, yt)
        gn = gnorm(m, Xt, yt)
        m2, _, _ = train(SEEDBASE + i, steps=2 * EPOCHS)  # 2x training, same seed/init
        w1 = wnorm(m); w2 = wnorm(m2)
        dep1 = scores[i]; dep2 = accm(m2, Xdep, ydep)
        rows.append(dict(seed=int(i), deploy_align=float(dep1), eig_min=float(emin),
                         grad_norm=float(gn), wnorm=float(w1), wnorm_2x=float(w2),
                         deploy_align_2x=float(dep2)))
        tag = "aligned" if dep1 > 0.8 else ("misaligned" if dep1 < 0.2 else "mid")
        print(f"  seed {i:3d} [{tag:10s}] eig_min={emin:+.2e} ||g||={gn:.2e} "
              f"wnorm {w1:.2f}->{w2:.2f}(2x) deploy {dep1:.2f}->{dep2:.2f}")
    emins = np.array([r["eig_min"] for r in rows])
    gns = np.array([r["grad_norm"] for r in rows])
    wdrift = np.array([abs(r["wnorm_2x"] - r["wnorm"]) / r["wnorm"] for r in rows])
    frac_min = float((emins > -1e-2).mean())
    print(f"  min-eig distribution: min {emins.min():+.2e} median {np.median(emins):+.2e} "
          f"max {emins.max():+.2e}")
    print(f"  fraction with eig_min > -1e-2 (genuine min, not saddle): {frac_min:.2f} ({(emins>-1e-2).sum()}/{len(emins)})")
    print(f"  grad-norm: median {np.median(gns):.2e} max {gns.max():.2e}")
    print(f"  weight-norm relative drift under 2x train: median {np.median(wdrift)*100:.1f}% max {wdrift.max()*100:.1f}%")
    print(f"  [t={time.time()-t0:.0f}s]")
    return dict(rows=rows, eig_min_min=float(emins.min()), eig_min_median=float(np.median(emins)),
                eig_min_max=float(emins.max()), frac_genuine_min=frac_min,
                grad_norm_median=float(np.median(gns)), grad_norm_max=float(gns.max()),
                wnorm_drift_median=float(np.median(wdrift)), wnorm_drift_max=float(wdrift.max()))


# ============================================================================
# 5. PERSISTENCE (Sleeper-Agents-style safety fine-tune)
# ============================================================================
def deliverable_5_persistence(models, scores, ft_steps=1000, n_target=None):
    print(f"\n{'='*72}\n5. PERSISTENCE  (safety fine-tune on ORIGINAL distribution)\n{'='*72}")
    t0 = time.time()
    scores = np.asarray(scores)
    Xdep, ydep = gen_align(4000, SEEDBASE + 50001, "deploy")
    lo = np.where(scores < 0.2)[0]  # misaligned-by-chance
    if n_target is not None:
        lo = lo[:n_target]
    print(f"  {len(lo)} misaligned-by-chance models (deploy<0.2); "
          f"safety fine-tune = {ft_steps} more steps on the SAME train dist (aligned==misaligned there)")
    rows = []
    for i in lo:
        m, Xt, yt = models[i]
        tr_before = accm(m, Xt, yt); dep_before = scores[i]
        # continue training from the trained weights on the original train distribution
        lf = nn.CrossEntropyLoss()
        mc = mk(); mc.load_state_dict(m.state_dict())
        opt = torch.optim.Adam(mc.parameters(), LR, weight_decay=WD)
        for _ in range(ft_steps):
            opt.zero_grad(); lf(mc(Xt), yt).backward(); opt.step()
        tr_after = accm(mc, Xt, yt); dep_after = accm(mc, Xdep, ydep)
        rows.append(dict(seed=int(i), train_before=float(tr_before), train_after=float(tr_after),
                         deploy_before=float(dep_before), deploy_after=float(dep_after)))
    db = np.array([r["deploy_before"] for r in rows]); da = np.array([r["deploy_after"] for r in rows])
    delta = da - db
    n_persist = int((da < 0.2).sum())
    tlo, thi = boot_ci(delta, seed=4)
    print(f"  deploy-alignment before: mean {db.mean():.3f}  after safety FT: mean {da.mean():.3f}")
    print(f"  change delta: mean {delta.mean():+.3f} [{tlo:+.3f},{thi:+.3f}]  (per-model)")
    print(f"  still misaligned after FT (deploy<0.2): {n_persist}/{len(rows)}  "
          f"({100*n_persist/max(len(rows),1):.0f}%)")
    print(f"  train acc stayed high: before {np.mean([r['train_before'] for r in rows]):.3f} "
          f"after {np.mean([r['train_after'] for r in rows]):.3f}")
    verdict = ("PERSISTS: safety FT on the in-distribution data does NOT remove deployment "
               "misalignment (Hubinger et al. 2024 sleeper-agent behavior)") if n_persist >= 0.8 * len(rows) \
              else "misalignment largely REMOVED by safety FT"
    print(f"  VERDICT: {verdict}")
    print(f"  [t={time.time()-t0:.0f}s]")
    return dict(ft_steps=ft_steps, n_target=len(rows), rows=rows,
                deploy_before_mean=float(db.mean()), deploy_after_mean=float(da.mean()),
                delta_mean=float(delta.mean()), delta_ci=[tlo, thi],
                n_persist=n_persist, frac_persist=float(n_persist / max(len(rows), 1)),
                train_before=float(np.mean([r['train_before'] for r in rows])),
                train_after=float(np.mean([r['train_after'] for r in rows])),
                verdict=verdict)


# ============================================================================
# main
# ============================================================================
def main():
    T0 = time.time()
    NSEED = 120
    r1, models, scores = deliverable_1_multimodality(nseed=NSEED)
    r3 = deliverable_3_basins(models, scores, nseed=min(60, NSEED))
    r4 = deliverable_4_minima(models, scores, nsample=12)
    r5 = deliverable_5_persistence(models, scores, ft_steps=1000)
    r2 = deliverable_2_threshold(fine_nseed=30, robust_nseed=12)

    results = dict(
        config=dict(H=H, WD=WD, MARGIN=MARGIN, NOISE=NOISE, EPOCHS=EPOCHS, N=N,
                    INPUT=INPUT, LR=LR, nseed=NSEED, seedbase=SEEDBASE),
        multimodality=r1, threshold=r2, basins=r3, minima=r4, persistence=r5,
    )
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {os.path.join(OUT,'results.json')}  [total {time.time()-T0:.0f}s]")
    return results


if __name__ == "__main__":
    main()
