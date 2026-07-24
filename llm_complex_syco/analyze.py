"""
Summarize + plot the complex-sycophancy sweep. Reads results.json (and stability.json if present).
Produces:
  - fig_sweep.png : deploy-alignment distribution vs p_spur, colored by whether content_acc>0.8
  - a printed sweep table with grad-norm and content-acc
All real data; no fabrication. Trustworthy seeds = content_acc_no_stance > 0.8.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT_BAR = 0.8


def load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    res = load("results.json")
    if res is None:
        print("no results.json yet"); return
    rows = res["rows"]
    ps = sorted(set(r["p_spur"] for r in rows))
    print(f"meta: {res['meta']}")
    print(f"{'p_spur':>7} {'n':>3} {'align mean':>10} {'align std':>9} {'align rng':>14} "
          f"{'content mean':>12} {'trust(n>0.8)':>12} {'gnorm med':>9}")
    summary = []
    for p in ps:
        rs = [r for r in rows if r["p_spur"] == p]
        al = np.array([r["deploy_alignment"] for r in rs])
        ca = np.array([r["content_acc_no_stance"] for r in rs])
        gn = np.array([r["grad_norm"] for r in rs])
        trust = int((ca > CONTENT_BAR).sum())
        print(f"{p:>7.2f} {len(rs):>3} {al.mean():>10.3f} {al.std():>9.3f} "
              f"[{al.min():.2f},{al.max():.2f}]".rjust(0) + f" {ca.mean():>12.3f} "
              f"{trust:>6}/{len(rs):<5} {np.median(gn):>9.3f}")
        summary.append(dict(p_spur=p, n=len(rs), align_mean=float(al.mean()), align_std=float(al.std()),
                            align_min=float(al.min()), align_max=float(al.max()),
                            content_mean=float(ca.mean()), trust=trust, gnorm_med=float(np.median(gn))))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for r in rows:
        trustworthy = r["content_acc_no_stance"] > CONTENT_BAR
        ax.scatter(r["p_spur"] + np.random.uniform(-0.006, 0.006), r["deploy_alignment"],
                   c=("tab:green" if trustworthy else "tab:red"), alpha=0.7,
                   s=40, edgecolors="k", linewidths=0.3)
    ax.axhline(0.5, ls=":", c="gray", lw=1)
    ax.set_xlabel("p_spur (stance-truth correlation in training)")
    ax.set_ylabel("deployment alignment  P(pred==truth)")
    ax.set_title("Seed alignment vs p_spur\ngreen=content_acc>0.8 (trustworthy), red=degenerate")
    ax.set_ylim(-0.05, 1.05)

    ax2 = axes[1]
    for r in rows:
        ax2.scatter(r["p_spur"] + np.random.uniform(-0.006, 0.006), r["content_acc_no_stance"],
                    c="tab:blue", alpha=0.6, s=35, edgecolors="k", linewidths=0.3)
    ax2.axhline(CONTENT_BAR, ls="--", c="tab:green", lw=1.2, label="non-degeneracy bar 0.8")
    ax2.axhline(0.5, ls=":", c="gray", lw=1, label="chance")
    ax2.set_xlabel("p_spur")
    ax2.set_ylabel("content_acc with stance REMOVED (cue-ablation)")
    ax2.set_title("Non-degeneracy: can the model judge claims\nwithout the user's stance?")
    ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_sweep.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

    stab = load("stability.json")
    if stab:
        print("\nSTABILITY (1x vs 2x epochs):")
        for s in stab:
            print(f"  seed{s['seed']} p={s['p_spur']}: align {s['align_1x']:.2f}->{s['align_2x']:.2f}  "
                  f"content {s['content_1x']:.2f}->{s['content_2x']:.2f}  "
                  f"gnorm {s['gnorm_1x']:.3f}->{s['gnorm_2x']:.3f}")


if __name__ == "__main__":
    main()
