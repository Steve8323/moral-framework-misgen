"""
ARCHITECTURE-GENERALITY of seed-driven aligned-vs-misaligned generalization at
genuine local minima.

Same aligned-vs-proxy XOR-cue task (equal-cost competing cues that agree in training,
proxy flips in deployment) presented to THREE model families:
  MLP (width x depth sweep), CNN (cue strip image), Transformer (cue token sequence).

For each architecture reports:
  - seed-multiplicity distribution of deployment alignment (>=50 seeds)
  - genuine-minima rigor: full-batch grad norm, weight-norm stability under 2x training,
    train-accuracy equivalence across seeds, Hessian min eigenvalue (HVP power iteration)
  - permutation-aligned linear-interpolation barriers: across-cluster (aligned<->misaligned)
    vs within-cluster (aligned<->aligned) control.

CPU only, small models. Run:  python run.py
Writes results.json + figures into this directory.
"""
import os, json, time, argparse
import numpy as np, torch
import common as C
from common import (gen_features, accuracy, grad_norm, weight_norm, hess_eigs,
                    linear_interp_barrier, summarize, N)
import models as M

# Tiny models: 1 thread avoids thread-dispatch overhead dominating per-op cost
# (grouped conv in particular is ~5x slower under many threads on these sizes).
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
HERE = os.path.dirname(os.path.abspath(__file__))

LR = 5e-3
WD = 2e-3
EPOCHS = 2000
DEPLOY_N = 4000

# ---------------- training ----------------
def train(make, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_features(N, seed, "train")
    m = make(); lf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

def cluster_indices(scores):
    hi = [i for i, s in enumerate(scores) if s > 0.8]
    lo = [i for i, s in enumerate(scores) if s < 0.2]
    return hi, lo

# ---------------- connectivity ----------------
def barrier_stats(make, permute_align, models, statedicts, Xb, yb, hi, lo, rng, k=20):
    """across-cluster (hi<->lo) vs within-cluster (hi<->hi) permutation-aligned barriers,
    plus the RAW (no permutation) controls."""
    def avg(ca, cb, aligned, same, kk):
        out = []
        tries = 0
        while len(out) < kk and tries < kk * 5:
            tries += 1
            i = int(rng.choice(ca)); j = int(rng.choice(cb))
            if same and i == j:
                continue
            sdA = statedicts[i]
            if aligned:
                sdB = permute_align(models[i], models[j], Xb)
            else:
                sdB = statedicts[j]
            out.append(linear_interp_barrier(make, sdA, sdB, Xb, yb))
        return float(np.mean(out)) if out else float("nan")
    return dict(
        within_raw=avg(hi, hi, False, True, k),
        across_raw=avg(hi, lo, False, False, k),
        within_aln=avg(hi, hi, True, True, k),
        across_aln=avg(hi, lo, True, False, k),
    )

# ---------------- per-architecture study ----------------
def study(name, make, permute_align, seedbase, nseed, label, do_connectivity=True,
          do_hess=True):
    t0 = time.time()
    Xdep, ydep = gen_features(DEPLOY_N, seedbase + 50001, "deploy")
    models = []; statedicts = []; scores = []; train_accs = []
    for s in range(nseed):
        m, Xt, yt = train(make, seedbase + s)
        models.append(m); statedicts.append({k: v.clone() for k, v in m.state_dict().items()})
        scores.append(accuracy(m, Xdep, ydep)); train_accs.append(accuracy(m, Xt, yt))
    summ = summarize(scores, train_accs, label)
    hi, lo = cluster_indices(scores)

    # genuine-minima rigor on representative seeds (one hi, one lo if available)
    rigor = {}
    reps = []
    if hi: reps.append(("aligned", hi[0]))
    if lo: reps.append(("misaligned", lo[0]))
    if not reps: reps.append(("mid", int(np.argmin(np.abs(np.array(scores) - 0.5)))))
    for tag, idx in reps:
        m = models[idx]
        Xt, yt = gen_features(N, seedbase + idx, "train")
        gn = grad_norm(m, Xt, yt)
        w1 = weight_norm(m)
        m2, _, _ = train(make, seedbase + idx, epochs=2 * EPOCHS)
        w2 = weight_norm(m2); sc2 = accuracy(m2, Xdep, ydep)
        ent = dict(seed=seedbase + idx, score=scores[idx], grad_norm=gn,
                   wnorm=w1, wnorm_2x=w2, wnorm_drift=abs(w2 - w1) / w1,
                   score_2x=sc2)
        if do_hess:
            top, mn = hess_eigs(m, Xt, yt)
            ent["hess_top"] = top; ent["hess_min"] = mn
            ent["genuine_min"] = bool(mn > -1e-2)
        rigor[tag] = ent

    conn = {}
    if do_connectivity and hi and lo:
        Xb, yb = gen_features(N, seedbase + 99, "train")
        rng = np.random.default_rng(0)
        conn = barrier_stats(make, permute_align, models, statedicts, Xb, yb, hi, lo, rng)
        conn["verdict"] = ("connected_mod_perm"
                           if conn["across_aln"] < 2 * max(conn["within_aln"], 1e-3)
                           else "distinct_basins")

    elapsed = time.time() - t0
    print(f"[{name}] {nseed} seeds in {elapsed:.0f}s | trainacc {summ['train_acc_mean']:.3f}"
          f"+/-{summ['train_acc_std']:.3f} | {label}: mean {summ['mean']:.2f} std {summ['std']:.2f}"
          f" range [{summ['min']:.2f},{summ['max']:.2f}] | hi {summ['nHi']} lo {summ['nLo']}"
          f" mid {summ['nMid']} | hist {summ['hist']}", flush=True)
    for tag, e in rigor.items():
        hs = f" hess_min {e.get('hess_min', float('nan')):+.2e}" if do_hess else ""
        print(f"    [{tag}] ||g|| {e['grad_norm']:.2e} wnorm {e['wnorm']:.2f}->{e['wnorm_2x']:.2f}"
              f" (drift {e['wnorm_drift']*100:.1f}%) score {e['score']:.2f}->{e['score_2x']:.2f}{hs}", flush=True)
    if conn:
        print(f"    barrier raw within {conn['within_raw']:.3f} across {conn['across_raw']:.3f} |"
              f" ALIGNED within {conn['within_aln']:.3f} across {conn['across_aln']:.3f}"
              f" -> {conn['verdict']}", flush=True)
    return dict(name=name, summary=summ, rigor=rigor, connectivity=conn, seconds=elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nseed = 12 if args.quick else args.nseed

    results = {"config": dict(LR=LR, WD=WD, EPOCHS=EPOCHS, N=N, MARGIN=C.MARGIN,
                              NOISE=C.NOISE, nseed=nseed)}

    # -------- MLP capacity map: width x depth --------
    print("\n=== MLP capacity map (width x depth) ===", flush=True)
    mlp_map = []
    for depth in [1, 2, 3]:
        for width in [4, 8, 16, 64]:
            r = study(f"MLP w{width} d{depth}", M.mlp_make(width, depth), M.mlp_permute_align,
                      seedbase=700 + 1000 * depth + width, nseed=nseed,
                      label="deployment alignment",
                      do_connectivity=(width <= 16), do_hess=(width <= 16))
            r["width"] = width; r["depth"] = depth
            mlp_map.append(r)
    results["mlp_map"] = mlp_map

    # representative MLP for the headline table: small, in the multiplicity regime
    # (small width commits; large width collapses to hedging -- see capacity map).
    results["mlp"] = next(r for r in mlp_map if r["width"] == 8 and r["depth"] == 1)

    # -------- CNN --------
    print("\n=== CNN ===", flush=True)
    results["cnn"] = study("CNN c4", M.cnn_make(4), M.cnn_permute_align,
                           seedbase=2000, nseed=nseed, label="deployment alignment")

    # -------- Transformer --------
    print("\n=== Transformer ===", flush=True)
    results["transformer"] = study("TF d8 L1", M.tf_make(8, 1, 1), M.tf_permute_align,
                                   seedbase=3000, nseed=nseed, label="deployment alignment")

    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print("\nWrote results.json", flush=True)
    make_figures(results)
    return results


def make_figures(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    # Fig 1: deployment-alignment histograms for the 3 families
    fams = [("MLP (w8 d1)", results["mlp"]), ("CNN (c4, grouped)", results["cnn"]),
            ("Transformer (d8 L1)", results["transformer"])]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, (nm, r) in zip(ax, fams):
        s = r["summary"]
        a.hist(s["scores"], bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
        a.axvline(0.5, color="gray", ls="--")
        a.set_xlabel("deployment alignment (1=aligned, 0=proxy)")
        a.set_ylabel("# seeds")
        a.set_title(f"{nm}\naligned {s['nHi']}/{s['n']}, misaligned {s['nLo']}/{s['n']}"
                    f"\nmean {s['mean']:.2f} std {s['std']:.2f}, trainacc {s['train_acc_mean']:.3f}")
    fig.suptitle("Seed-driven aligned-vs-misaligned multiplicity is architecture-general", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_distributions.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 2: MLP capacity map heatmaps (std of alignment, and frac in {hi,lo})
    widths = [4, 8, 16, 64]; depths = [1, 2, 3]
    std_grid = np.zeros((len(depths), len(widths)))
    commit_grid = np.zeros((len(depths), len(widths)))  # frac of seeds that COMMIT (hi or lo)
    for r in results["mlp_map"]:
        i = depths.index(r["depth"]); j = widths.index(r["width"]); s = r["summary"]
        std_grid[i, j] = s["std"]
        commit_grid[i, j] = (s["nHi"] + s["nLo"]) / s["n"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, grid, ttl in [(ax[0], std_grid, "std(deployment alignment)"),
                         (ax[1], commit_grid, "fraction of seeds committed (>0.8 or <0.2)")]:
        im = a.imshow(grid, cmap="viridis", aspect="auto", origin="lower", vmin=0)
        a.set_xticks(range(len(widths))); a.set_xticklabels(widths)
        a.set_yticks(range(len(depths))); a.set_yticklabels(depths)
        a.set_xlabel("width"); a.set_ylabel("depth"); a.set_title(ttl)
        for i in range(len(depths)):
            for j in range(len(widths)):
                a.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center",
                       color="white" if grid[i, j] < grid.max() * 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("MLP capacity map: where multiplicity appears vs collapses", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_mlp_capacity.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 3: connectivity barriers, raw vs permutation-aligned, within vs across.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    names = []; rw = []; ra = []; aw = []; aa = []
    for nm, key in [("MLP w8d1", "mlp"), ("CNN c4", "cnn"), ("TF d8L1", "transformer")]:
        c = results[key].get("connectivity", {})
        if c:
            names.append(nm)
            rw.append(c["within_raw"]); ra.append(c["across_raw"])
            aw.append(c["within_aln"]); aa.append(c["across_aln"])
    x = np.arange(len(names)); w = 0.35
    for a, (within, across, ttl) in zip(
            ax, [(rw, ra, "RAW (no permutation)"), (aw, aa, "permutation-ALIGNED")]):
        a.bar(x - w / 2, within, w, label="within-cluster (aligned<->aligned)", color="#55A868")
        a.bar(x + w / 2, across, w, label="across-cluster (aligned<->misaligned)", color="#C44E52")
        a.set_xticks(x); a.set_xticklabels(names); a.set_ylabel("linear-interp loss barrier")
        a.set_title(ttl); a.legend(fontsize=8)
    fig.suptitle("Connectivity: permutation alignment collapses MLP/CNN barriers (basins "
                 "connected mod perm);\nTF channel-perm is insufficient for attention (barriers "
                 "stay high) -> inconclusive", y=1.04, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_connectivity.png"), bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig_distributions.png, fig_mlp_capacity.png, fig_connectivity.png", flush=True)


if __name__ == "__main__":
    main()
