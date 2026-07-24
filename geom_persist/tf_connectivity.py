"""
PART 2 -- FIRM UP TRANSFORMER CONNECTIVITY (functional connectivity that sidesteps
weight-space permutation symmetry).

The prior arch result (../arch/results.json) was INCONCLUSIVE for the Transformer:
weight-space barriers under the limited d_model channel-permutation symmetry did NOT
collapse (within_aln=3.0 > across_aln=2.3 -- both larger than the raw barriers ~1.5-2.0),
so a barrier-based verdict is meaningless. MultiheadAttention/LayerNorm/FF do NOT linearly
realign under a single channel permutation, so the permutation-aligned linear interpolation
is just two unrelated networks averaged in weight space.

This module measures FUNCTIONAL connectivity along the linear weight interpolation between
two TF solutions -- no weight symmetry needed:

  (i)  train loss + train accuracy along the path (the classic barrier, but read functionally);
  (ii) FUNCTIONAL similarity along the path:
         - prediction AGREEMENT of the interpolated model with each endpoint and with the
           consensus, on the train set (does the midpoint still compute *a* good function?);
         - linear CKA between the midpoint's pooled representation and each endpoint's;
  (iii) REPAIR-style activation renormalization: at each interpolation point, rescale every
        neuron's pre-pool activation (per-channel affine: mean+std) to the linear interpolation
        of the two endpoints' per-channel statistics. This is the BN-free REPAIR fix
        (Jordan et al. 2023) -- it removes "variance collapse" of averaged features and is the
        standard way to expose linear-mode connectivity in nets that do not realign in weight
        space. We re-estimate the head's working point by refitting ONLY the final LayerNorm/
        head input statistics (cheap, functional), never the body.

Verdict logic: a pair is FUNCTIONALLY CONNECTED if, after REPAIR, the loss barrier on the
path is small AND the midpoint keeps high prediction agreement with a good endpoint. We
compare WITHIN-cluster (aligned<->aligned) vs ACROSS-cluster (aligned<->misaligned) barriers.
If across >> within, the two function classes sit in DISTINCT basins (not one manifold).

CPU only. Run:  python3 -u tf_connectivity.py [--nseed 60] [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arch"))
import common as C
from common import gen_features, accuracy, grad_norm, weight_norm, hess_eigs, N
import models as M

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
HERE = os.path.dirname(os.path.abspath(__file__))
LR, WD, EPOCHS, DEPLOY_N = 5e-3, 2e-3, 2000, 4000
SEEDBASE = 3000          # same family as arch/run.py Transformer study


def train(seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_features(N, seed, "train")
    m = M.tf_make(8, 1, 1)(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr


# ---------------------------------------------------------------------------
# functional probes
# ---------------------------------------------------------------------------
def logits(m, X):
    with torch.no_grad():
        return m(X)


def preds(m, X):
    return logits(m, X).argmax(1)


def pooled_rep(m, X):
    """Pooled (pre-head) representation, the d_model feature vector feeding the head."""
    return torch.as_tensor(M.tf_pooled(m, X))


def linear_cka(Xa, Xb):
    """Linear CKA between two [n,d] representation matrices (centered)."""
    Xa = Xa - Xa.mean(0, keepdim=True); Xb = Xb - Xb.mean(0, keepdim=True)
    hsic = (Xa.T @ Xb).pow(2).sum()
    na = (Xa.T @ Xa).pow(2).sum().sqrt(); nb = (Xb.T @ Xb).pow(2).sum().sqrt()
    return float(hsic / (na * nb + 1e-12))


def interp_state(sA, sB, alpha):
    return {k: (1 - alpha) * sA[k] + alpha * sB[k] for k in sA}


# ---------------------------------------------------------------------------
# REPAIR-style activation renormalization (BN-free)
# ---------------------------------------------------------------------------
def channel_stats_pre_pool(m, X):
    """Per-token-pooled feature statistics in the d_model basis, just before mean-pool.
    Returns (mean[d], std[d]) of the pooled representation over the dataset."""
    rep = pooled_rep(m, X)              # [n, d]
    return rep.mean(0), rep.std(0) + 1e-6


def repair_pooled_barrier(mA, mB, X, y, npts=11):
    """REPAIR (Jordan et al. 2023) adapted to the pre-head pooled feature basis.

    For each interpolation coefficient alpha, build the weight-interpolated model and
    re-normalize its pooled representation per-channel so its mean/std MATCH the linear
    interpolation of the two endpoints' pooled statistics (the 'goal' statistics REPAIR
    targets). Then read the head on the renormalized features. This removes the variance
    collapse that makes naive weight averaging look like a high barrier when the two nets
    do not realign in weight space.

    Returns dict with loss[], acc[], and the (repaired) barrier."""
    lf = nn.CrossEntropyLoss()
    sA, sB = mA.state_dict(), mB.state_dict()
    muA, sdA = channel_stats_pre_pool(mA, X)
    muB, sdB = channel_stats_pre_pool(mB, X)
    head_w = sA["head.weight"]          # endpoints share architecture; use A's head fn shape
    m = M.tf_make(8, 1, 1)()
    raw_loss, raw_acc, rep_loss, rep_acc = [], [], [], []
    for alpha in np.linspace(0, 1, npts):
        m.load_state_dict(interp_state(sA, sB, alpha))
        # --- raw (no repair) ---
        with torch.no_grad():
            lo = m(X)
        raw_loss.append(float(lf(lo, y))); raw_acc.append(float((lo.argmax(1) == y).float().mean()))
        # --- repaired: renormalize pooled rep to goal stats, then apply head ---
        goal_mu = (1 - alpha) * muA + alpha * muB
        goal_sd = (1 - alpha) * sdA + alpha * sdB
        rep = pooled_rep(m, X)                       # [n,d]
        cur_mu = rep.mean(0); cur_sd = rep.std(0) + 1e-6
        rep_fixed = (rep - cur_mu) / cur_sd * goal_sd + goal_mu
        with torch.no_grad():
            # apply the (interpolated) head to the repaired features
            hw = m.head.weight; hb = m.head.bias
            lo_r = rep_fixed @ hw.T + hb
        rep_loss.append(float(lf(lo_r, y))); rep_acc.append(float((lo_r.argmax(1) == y).float().mean()))
    raw_loss = np.array(raw_loss); rep_loss = np.array(rep_loss)
    raw_acc = np.array(raw_acc); rep_acc = np.array(rep_acc)
    return dict(
        raw_loss=raw_loss.tolist(), raw_acc=raw_acc.tolist(),
        repaired_loss=rep_loss.tolist(), repaired_acc=rep_acc.tolist(),
        raw_barrier=float(raw_loss.max() - max(raw_loss[0], raw_loss[-1])),
        repaired_barrier=float(rep_loss.max() - max(rep_loss[0], rep_loss[-1])),
        raw_acc_drop=float(max(raw_acc[0], raw_acc[-1]) - raw_acc.min()),
        repaired_acc_drop=float(max(rep_acc[0], rep_acc[-1]) - rep_acc.min()),
    )


def _block_acts(m, X):
    """Per-channel (d_model) statistics at every internal point of the TF body:
    attn output, post-ln1, ff hidden, post-ln2/pooled. Used as REPAIR goal stats."""
    Xs = torch.as_tensor(C.to_seq(X) if X.dim() == 2 else X)
    with torch.no_grad():
        h = m.embed(Xs) + m.pos
        stats = {}
        for i, blk in enumerate(m.blocks):
            a, _ = blk["attn"](h, h, h)
            stats[f"attn{i}"] = a
            h = blk["ln1"](h + a)
            ffh = torch.relu(blk["ff"][0](h))
            stats[f"ffh{i}"] = ffh
            h = blk["ln2"](h + blk["ff"][2](ffh))
            stats[f"post{i}"] = h
        stats["pool"] = h.mean(1)
    return stats


def _stat(t):
    """per-channel mean/std over (batch[,seq]) -> [C]."""
    if t.dim() == 3:
        flat = t.reshape(-1, t.shape[-1])
    else:
        flat = t
    return flat.mean(0), flat.std(0) + 1e-6


def repair_full_barrier(mA, mB, X, y, npts=11):
    """FULL REPAIR (Jordan et al. 2023, BN-free variant): at each interpolation alpha,
    run the weight-interpolated model but renormalize EVERY internal activation
    (attention output, FF hidden, post-block residual, pooled feature) per-channel to the
    linear interpolation of the two endpoints' activation statistics. This is the standard
    fix that exposes linear-mode connectivity in nets that do not realign in weight space:
    it removes the variance collapse of averaged features at every depth, not just the head.

    Returns repaired loss/acc curve + barrier."""
    lf = nn.CrossEntropyLoss()
    sA, sB = mA.state_dict(), mB.state_dict()
    Xs = torch.as_tensor(C.to_seq(X) if X.dim() == 2 else X)
    stA = _block_acts(mA, X); stB = _block_acts(mB, X)
    goal = {}
    m = M.tf_make(8, 1, 1)()
    rep_loss, rep_acc = [], []
    for alpha in np.linspace(0, 1, npts):
        m.load_state_dict(interp_state(sA, sB, alpha))
        # precompute goal stats for this alpha
        gmu = {}; gsd = {}
        for k in stA:
            ma, sa = _stat(stA[k]); mb, sb = _stat(stB[k])
            gmu[k] = (1 - alpha) * ma + alpha * mb
            gsd[k] = (1 - alpha) * sa + alpha * sb

        def renorm(t, k):
            cm, cs = _stat(t)
            return (t - cm) / cs * gsd[k] + gmu[k]

        with torch.no_grad():
            h = m.embed(Xs) + m.pos
            for i, blk in enumerate(m.blocks):
                a, _ = blk["attn"](h, h, h)
                a = renorm(a, f"attn{i}")
                h = blk["ln1"](h + a)
                ffh = torch.relu(blk["ff"][0](h)); ffh = renorm(ffh, f"ffh{i}")
                h = blk["ln2"](h + blk["ff"][2](ffh))
                h = renorm(h, f"post{i}")
            pool = renorm(h.mean(1), "pool")
            lo = m.head(pool)
        rep_loss.append(float(lf(lo, y))); rep_acc.append(float((lo.argmax(1) == y).float().mean()))
    rep_loss = np.array(rep_loss); rep_acc = np.array(rep_acc)
    return dict(
        repaired_loss=rep_loss.tolist(), repaired_acc=rep_acc.tolist(),
        repaired_barrier=float(rep_loss.max() - max(rep_loss[0], rep_loss[-1])),
        repaired_acc_drop=float(max(rep_acc[0], rep_acc[-1]) - rep_acc.min()),
        repaired_acc_mid=float(rep_acc[len(rep_acc) // 2]),
    )


def functional_path(mA, mB, X, y, npts=11):
    """Prediction agreement + CKA of the interpolated model with the two endpoints
    along the linear weight path (no repair, no symmetry)."""
    sA, sB = mA.state_dict(), mB.state_dict()
    pA = preds(mA, X); pB = preds(mB, X)
    repA = pooled_rep(mA, X); repB = pooled_rep(mB, X)
    m = M.tf_make(8, 1, 1)()
    agreeA, agreeB, agree_best, ckaA, ckaB = [], [], [], [], []
    for alpha in np.linspace(0, 1, npts):
        m.load_state_dict(interp_state(sA, sB, alpha))
        pm = preds(m, X); repm = pooled_rep(m, X)
        aA = float((pm == pA).float().mean()); aB = float((pm == pB).float().mean())
        agreeA.append(aA); agreeB.append(aB); agree_best.append(max(aA, aB))
        ckaA.append(linear_cka(repm, repA)); ckaB.append(linear_cka(repm, repB))
    return dict(agreeA=agreeA, agreeB=agreeB, agree_best=agree_best,
                ckaA=ckaA, ckaB=ckaB,
                min_agree_best=float(np.min(agree_best)),
                min_cka_to_nearest=float(np.min(np.maximum(ckaA, ckaB))))


# ---------------------------------------------------------------------------
# main study
# ---------------------------------------------------------------------------
def run(nseed=60, npairs=24, npts=11, smoke=False):
    if smoke:
        nseed, npairs, npts = 12, 6, 7
    t0 = time.time()
    print(f"[tf_connectivity] training {nseed} TF seeds (d8 L1)...", flush=True)
    Xdep, ydep = gen_features(DEPLOY_N, SEEDBASE + 50001, "deploy")
    Xb, yb = gen_features(N, SEEDBASE + 99, "train")       # common reference batch
    models, scores = [], []
    for s in range(nseed):
        m, _, _ = train(SEEDBASE + s)
        models.append(m); scores.append(accuracy(m, Xdep, ydep))
        if (s + 1) % 5 == 0 or s == nseed - 1:
            print(f"  seed {s+1}/{nseed}  deploy-align={scores[-1]:.2f}  [t={time.time()-t0:.0f}s]", flush=True)
    scores = np.array(scores)
    hi = [i for i in range(nseed) if scores[i] > 0.8]
    lo = [i for i in range(nseed) if scores[i] < 0.2]
    print(f"  clusters: aligned(hi) {len(hi)}  misaligned(lo) {len(lo)}  "
          f"mid {nseed-len(hi)-len(lo)}  (mean {scores.mean():.2f} std {scores.std():.2f})", flush=True)

    # genuine-minima checks on representative TF seeds
    minima = {}
    for tag, idx in [("aligned", hi[0] if hi else None), ("misaligned", lo[0] if lo else None)]:
        if idx is None:
            continue
        Xt, yt = gen_features(N, SEEDBASE + idx, "train")
        gn = grad_norm(models[idx], Xt, yt)
        top, mn = hess_eigs(models[idx], Xt, yt)
        minima[tag] = dict(seed=SEEDBASE + idx, score=float(scores[idx]), grad_norm=float(gn),
                           hess_top=float(top), hess_min=float(mn), genuine_min=bool(mn > -1e-2))
        print(f"    [{tag}] ||g||={gn:.2e} hess_min={mn:+.2e} "
              f"({'MIN' if mn>-1e-2 else 'SADDLE'})", flush=True)

    rng = np.random.default_rng(0)

    def sample_pairs(ca, cb, same, k):
        out = []; tries = 0
        while len(out) < k and tries < k * 20:
            tries += 1
            i = int(rng.choice(ca)); j = int(rng.choice(cb))
            if same and i == j:
                continue
            if (i, j) in out or (j, i) in out:
                continue
            out.append((i, j))
        return out

    def measure(pairs):
        rows = []
        for (i, j) in pairs:
            rep = repair_pooled_barrier(models[i], models[j], Xb, yb, npts)
            full = repair_full_barrier(models[i], models[j], Xb, yb, npts)
            fn = functional_path(models[i], models[j], Xb, yb, npts)
            rows.append(dict(i=i, j=j,
                             raw_barrier=rep["raw_barrier"],
                             repaired_barrier=rep["repaired_barrier"],          # pooled-only REPAIR
                             full_repaired_barrier=full["repaired_barrier"],    # full-network REPAIR
                             full_repaired_acc_drop=full["repaired_acc_drop"],
                             raw_acc_drop=rep["raw_acc_drop"], repaired_acc_drop=rep["repaired_acc_drop"],
                             min_agree_best=fn["min_agree_best"], min_cka=fn["min_cka_to_nearest"],
                             curves=dict(rep=rep, full=full, fn=fn)))
        return rows

    out = {}
    if len(hi) >= 2 and len(lo) >= 1:
        wp = sample_pairs(hi, hi, True, npairs)
        ap = sample_pairs(hi, lo, False, npairs)
        print(f"  measuring {len(wp)} within-cluster + {len(ap)} across-cluster pairs "
              f"(REPAIR + functional)...", flush=True)
        within = measure(wp); across = measure(ap)

        def agg(rows, key):
            v = np.array([r[key] for r in rows]); return float(v.mean()), float(v.std())

        for label, rows in [("WITHIN (aligned<->aligned)", within),
                            ("ACROSS (aligned<->misaligned)", across)]:
            rb_m, rb_s = agg(rows, "raw_barrier")
            pb_m, pb_s = agg(rows, "repaired_barrier")
            fb_m, fb_s = agg(rows, "full_repaired_barrier")
            ag_m, ag_s = agg(rows, "min_agree_best")
            ck_m, ck_s = agg(rows, "min_cka")
            print(f"    {label}:\n"
                  f"        raw_barrier        {rb_m:.3f}+/-{rb_s:.3f}\n"
                  f"        REPAIR(pool)       {pb_m:.3f}+/-{pb_s:.3f}\n"
                  f"        REPAIR(full)       {fb_m:.3f}+/-{fb_s:.3f}   <- primary functional barrier\n"
                  f"        min_pred_agree     {ag_m:.3f}+/-{ag_s:.3f}\n"
                  f"        min_CKA(to endpt)  {ck_m:.3f}+/-{ck_s:.3f}", flush=True)

        # PRIMARY functional measure = full-network REPAIR barrier (variance-collapse removed
        # at every depth, no weight symmetry assumed).
        w_rep = np.array([r["full_repaired_barrier"] for r in within])
        a_rep = np.array([r["full_repaired_barrier"] for r in across])
        w_ag = np.array([r["min_agree_best"] for r in within])
        a_ag = np.array([r["min_agree_best"] for r in across])

        from scipy import stats as sstats
        mw_bar = sstats.mannwhitneyu(a_rep, w_rep, alternative="greater")
        mw_ag = sstats.mannwhitneyu(w_ag, a_ag, alternative="greater")  # within agrees MORE

        # verdict: connected if across full-REPAIR barrier is small in absolute terms AND
        # not dramatically larger than within; distinct if across barrier and agreement
        # gap are large + significant.
        across_low = a_rep.mean() < 0.20 and a_ag.mean() > 0.85
        gap_sig = (mw_bar.pvalue < 0.05) and (a_rep.mean() > 2 * max(w_rep.mean(), 1e-3))
        if across_low and not gap_sig:
            verdict = ("CONNECTED (one functional manifold): after REPAIR the across-cluster "
                       "barrier is small and the midpoint keeps high prediction agreement with a "
                       "good endpoint -- aligned and misaligned TF solutions are linearly "
                       "(functionally) connected.")
        elif gap_sig or (a_rep.mean() > 0.20 and a_rep.mean() > 1.5 * max(w_rep.mean(), 1e-3)):
            verdict = ("DISTINCT basins: across-cluster REPAIR barrier is substantially larger "
                       "than within-cluster and prediction agreement collapses at the midpoint -- "
                       "aligned and misaligned TF solutions are NOT on one linearly-connected "
                       "manifold.")
        else:
            verdict = ("AMBIGUOUS: REPAIR reduces but does not eliminate the across barrier; the "
                       "best functional measure (min prediction agreement at midpoint) is reported "
                       "and should be read as the primary statistic.")

        out = dict(
            within=dict(rows=[{k: v for k, v in r.items() if k != "curves"} for r in within],
                        raw_barrier_mean=agg(within, "raw_barrier")[0],
                        pooled_repair_barrier_mean=agg(within, "repaired_barrier")[0],
                        full_repair_barrier_mean=float(w_rep.mean()),
                        full_repair_barrier_std=float(w_rep.std()),
                        min_agree_mean=float(w_ag.mean()), min_agree_std=float(w_ag.std()),
                        min_cka_mean=agg(within, "min_cka")[0]),
            across=dict(rows=[{k: v for k, v in r.items() if k != "curves"} for r in across],
                        raw_barrier_mean=agg(across, "raw_barrier")[0],
                        pooled_repair_barrier_mean=agg(across, "repaired_barrier")[0],
                        full_repair_barrier_mean=float(a_rep.mean()),
                        full_repair_barrier_std=float(a_rep.std()),
                        min_agree_mean=float(a_ag.mean()), min_agree_std=float(a_ag.std()),
                        min_cka_mean=agg(across, "min_cka")[0]),
            mannwhitney_p_barrier=float(mw_bar.pvalue),
            mannwhitney_p_agreement=float(mw_ag.pvalue),
            verdict=verdict,
            # keep a couple of representative curves for plotting
            example_within_curve=within[0]["curves"] if within else None,
            example_across_curve=across[0]["curves"] if across else None,
        )
        print(f"\n  VERDICT: {verdict}", flush=True)
        print(f"  (MannWhitney across>within repaired-barrier p={mw_bar.pvalue:.3g}; "
              f"within>across agreement p={mw_ag.pvalue:.3g})", flush=True)
    else:
        print("  not enough hi/lo seeds for connectivity comparison", flush=True)

    result = dict(
        nseed=nseed, npairs=npairs, npts=npts,
        scores=scores.tolist(), n_hi=len(hi), n_lo=len(lo),
        mean=float(scores.mean()), std=float(scores.std()),
        minima=minima, connectivity=out,
        config=dict(LR=LR, WD=WD, EPOCHS=EPOCHS, N=N, seedbase=SEEDBASE,
                    arch="TinyTransformer d_model=8 L=1 H=1"),
    )
    path = os.path.join(HERE, "tf_connectivity.json")
    json.dump(result, open(path, "w"), indent=2)
    print(f"\nWrote {path}  [total {time.time()-t0:.0f}s]", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=60)
    ap.add_argument("--npairs", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(nseed=args.nseed, npairs=args.npairs, smoke=args.smoke)
