"""
Main experiment: mini-HANS heuristic-vs-rule seed multiplicity + minima verification + basins.

Config (locked from sweeps): symbits encoding (2 cues + 8 distractors), MLP width 64 depth 2,
weight_decay 2e-3, lr 3e-3, full-batch Adam, 800 epochs, ntr=400.
"""
import json
import time
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from task import gen_symbits, symbits_dim
from train import train_one, accuracy, ce_loss, full_batch_grad_norm, weight_norm
from hessian import min_eigenvalue
from barrier import barrier_report

OUT = "/Users/steven/moral_framework_misgen/harder_b"

CFG = dict(
    V=8, max_p=6, max_h=3, n_distract=8,
    ntr=400, n_ood=4000,
    width=64, depth=2, wd=2e-3, lr=3e-3, epochs=800,
    n_seeds=60, train_data_seed=12345, ood_data_seed=999,
)


def main():
    t0 = time.time()
    c = CFG
    in_dim = symbits_dim(c["n_distract"])
    Xtr, ytr, _ = gen_symbits(c["ntr"], c["V"], c["max_p"], c["max_h"],
                              seed=c["train_data_seed"], ood=False, n_distract=c["n_distract"])
    Xood, yood, ht = gen_symbits(c["n_ood"], c["V"], c["max_p"], c["max_h"],
                                 seed=c["ood_data_seed"], ood=True, n_distract=c["n_distract"])
    # sanity: heuristic and rule both fit train; disagree on OOD
    h_train = (Xtr[:, 0] == ytr).mean()
    r_train = (Xtr[:, 1] == ytr).mean()
    h_ood = (Xood[:, 0] == yood).mean()
    r_ood = (Xood[:, 1] == yood).mean()

    results = {"config": c, "feature_dim": in_dim,
               "sanity": {"heuristic_train_acc": float(h_train), "rule_train_acc": float(r_train),
                          "heuristic_ood_acc": float(h_ood), "rule_ood_acc": float(r_ood),
                          "ood_label_balance": float(yood.mean())}}
    print("SANITY:", results["sanity"], flush=True)

    # ---------- 1+3: train all seeds, record OOD dist + minima numbers ----------
    seeds = list(range(c["n_seeds"]))
    ood_acc, tr_acc, tr_loss, grad_norms, wnorms = [], [], [], [], []
    models = {}
    for s in seeds:
        m, _ = train_one(Xtr, ytr, s, in_dim, width=c["width"], depth=c["depth"],
                         wd=c["wd"], epochs=c["epochs"], lr=c["lr"])
        ta = accuracy(m, Xtr, ytr); oa = accuracy(m, Xood, yood)
        gn = full_batch_grad_norm(m, Xtr, ytr, wd=c["wd"]); wn = weight_norm(m)
        tl = ce_loss(m, Xtr, ytr)
        ood_acc.append(oa); tr_acc.append(ta); tr_loss.append(tl)
        grad_norms.append(gn); wnorms.append(wn)
        models[s] = m
        if s % 10 == 0:
            print(f"  seed {s}: tr={ta:.3f} ood={oa:.3f} grad={gn:.2e} wn={wn:.2f}", flush=True)
    ood_acc = np.array(ood_acc); tr_acc = np.array(tr_acc); tr_loss = np.array(tr_loss)
    grad_norms = np.array(grad_norms); wnorms = np.array(wnorms)

    hist_counts, hist_edges = np.histogram(ood_acc, bins=10, range=(0.45, 0.95))
    results["seed_multiplicity"] = {
        "n_seeds": c["n_seeds"],
        "ood_acc_mean": float(ood_acc.mean()), "ood_acc_std": float(ood_acc.std()),
        "ood_acc_min": float(ood_acc.min()), "ood_acc_max": float(ood_acc.max()),
        "ood_acc_range": float(ood_acc.max() - ood_acc.min()),
        "ood_acc_all": ood_acc.round(4).tolist(),
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": hist_edges.round(3).tolist(),
        "all_train_acc_min": float(tr_acc.min()), "all_train_acc_max": float(tr_acc.max()),
        "train_loss_mean": float(tr_loss.mean()), "train_loss_std": float(tr_loss.std()),
        "train_loss_min": float(tr_loss.min()), "train_loss_max": float(tr_loss.max()),
    }
    results["minima_finite"] = {
        "grad_norm_mean": float(grad_norms.mean()), "grad_norm_max": float(grad_norms.max()),
        "grad_norm_min": float(grad_norms.min()),
        "weight_norm_mean": float(wnorms.mean()), "weight_norm_std": float(wnorms.std()),
        "weight_norm_max": float(wnorms.max()),
    }
    print("OOD spread:", results["seed_multiplicity"]["ood_acc_min"], "-",
          results["seed_multiplicity"]["ood_acc_max"], flush=True)

    # ---------- 1b: weight-norm stability under 2x more training ----------
    # take 3 representative seeds (min, median, max OOD)
    order = np.argsort(ood_acc)
    ood_by_seed = {seeds[i]: float(ood_acc[i]) for i in range(len(seeds))}
    rep = [seeds[order[0]], seeds[order[len(order) // 2]], seeds[order[-1]]]
    stab = []
    for s in rep:
        wn_before = weight_norm(models[s])
        m2, _ = train_one(Xtr, ytr, s, in_dim, width=c["width"], depth=c["depth"],
                          wd=c["wd"], epochs=c["epochs"] * 2, lr=c["lr"])
        wn_after = weight_norm(m2)
        oa2 = accuracy(m2, Xood, yood)
        gn2 = full_batch_grad_norm(m2, Xtr, ytr, wd=c["wd"])
        stab.append({"seed": int(s), "ood_acc_1x": ood_by_seed[s],
                     "wnorm_1x": float(wn_before), "wnorm_2x": float(wn_after),
                     "ood_acc_2x": float(oa2), "grad_norm_2x": float(gn2),
                     "wnorm_rel_change": float(abs(wn_after - wn_before) / (wn_before + 1e-9))})
    results["weight_norm_stability_2x"] = stab
    print("2x stability:", stab, flush=True)

    # ---------- 2: Hessian min-eigenvalue on representative seeds ----------
    hess = []
    for s in rep:
        lam_min, l_top = min_eigenvalue(models[s], Xtr, ytr, c["wd"], n_iter=100, seed=0)
        hess.append({"seed": int(s), "lambda_min": float(lam_min), "lambda_top": float(l_top),
                     "genuine_local_min": bool(lam_min > -1e-3)})
    results["hessian"] = hess
    print("Hessian:", hess, flush=True)

    # ---------- 4: basin barrier between high-OOD and low-OOD seed ----------
    s_low = seeds[order[0]]; s_high = seeds[order[-1]]
    br = barrier_report(models[s_low], models[s_high], Xtr, ytr, n_alpha=21)
    br["seed_low_ood"] = int(s_low); br["seed_high_ood"] = int(s_high)
    br["ood_low"] = float(ood_acc[order[0]]); br["ood_high"] = float(ood_acc[order[-1]])
    results["basin_barrier"] = br
    print(f"Barrier raw={br['raw_barrier']:.4f} aligned={br['aligned_barrier']:.4f} "
          f"(low seed {s_low} ood {br['ood_low']:.3f} vs high seed {s_high} ood {br['ood_high']:.3f})",
          flush=True)

    results["runtime_sec"] = time.time() - t0
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    make_figures(results)
    print(f"DONE in {results['runtime_sec']:.1f}s. JSON + figures written to {OUT}", flush=True)
    return results


def order_index(seeds, s, ood_acc):
    return seeds.index(s)


def make_figures(r):
    sm = r["seed_multiplicity"]
    ood = np.array(sm["ood_acc_all"])
    # Fig 1: OOD histogram + heuristic/rule reference lines
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(ood, bins=sm["histogram_edges"], color="#4477aa", edgecolor="k", alpha=0.85)
    ax[0].axvline(r["sanity"]["heuristic_ood_acc"], color="crimson", ls="--",
                  label=f"pure heuristic = {r['sanity']['heuristic_ood_acc']:.2f}")
    ax[0].axvline(r["sanity"]["rule_ood_acc"], color="green", ls="--",
                  label=f"pure true rule = {r['sanity']['rule_ood_acc']:.2f}")
    ax[0].axvline(ood.mean(), color="black", ls="-", lw=2, label=f"mean = {ood.mean():.3f}")
    ax[0].set_xlabel("OOD (HANS-probe) accuracy")
    ax[0].set_ylabel("# seeds")
    ax[0].set_title(f"Seed multiplicity, n={sm['n_seeds']}\n"
                    f"range {sm['ood_acc_min']:.3f}-{sm['ood_acc_max']:.3f} "
                    f"(spread {sm['ood_acc_range']:.3f})")
    ax[0].legend(fontsize=8)
    # all seeds at 100% train (sanity strip)
    ax[1].scatter(range(len(ood)), np.sort(ood), c="#4477aa", s=18)
    ax[1].axhline(r["sanity"]["heuristic_ood_acc"], color="crimson", ls="--")
    ax[1].axhline(r["sanity"]["rule_ood_acc"], color="green", ls="--")
    ax[1].set_xlabel("seed rank")
    ax[1].set_ylabel("OOD accuracy")
    ax[1].set_title(f"All seeds train_acc in "
                    f"[{sm['all_train_acc_min']:.3f},{sm['all_train_acc_max']:.3f}], "
                    f"train_loss {sm['train_loss_min']:.3f}-{sm['train_loss_max']:.3f}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_multiplicity.png", dpi=130)
    plt.close()

    # Fig 2: barrier raw vs aligned
    br = r["basin_barrier"]
    a = br["alphas"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a, br["raw_curve"], "-o", ms=3, label=f"raw  (barrier {br['raw_barrier']:.3f})",
            color="crimson")
    ax.plot(a, br["aligned_curve"], "-s", ms=3,
            label=f"aligned (barrier {br['aligned_barrier']:.3f})", color="navy")
    ax.set_xlabel(r"interpolation $\alpha$ (low-OOD seed $\to$ high-OOD seed)")
    ax.set_ylabel("train CE loss")
    ax.set_title(f"Linear-mode-connectivity barrier\n"
                 f"seed {br['seed_low_ood']} (OOD {br['ood_low']:.2f}) "
                 f"-> seed {br['seed_high_ood']} (OOD {br['ood_high']:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_barrier.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
