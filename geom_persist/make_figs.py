"""
Figures for geom_persist:
  fig_persistence_size.png  -- persistence vs model size (Part 1a)
  fig_removal_curve.png      -- removal curve + adversarial (Part 1b/c)
  fig_tf_connectivity.png    -- TF functional connectivity (Part 2)
Reads persistence.json + tf_connectivity.json in this dir. Run: python3 make_figs.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})


def fig_persistence():
    P = json.load(open(os.path.join(HERE, "persistence.json")))
    pv = P["persistence_vs_size"]
    labels = [f"{r['kind']}\n({r['n_params']} params)" for r in pv]
    fracs = [r["frac_persist"] for r in pv]
    befores = [r["deploy_before"] for r in pv]
    afters = [r["deploy_after"] for r in pv]
    ns = [r["n_misaligned"] for r in pv]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(labels))
    bars = ax[0].bar(x, [100 * f for f in fracs], color="#C44E52", edgecolor="white")
    for b, f, n in zip(bars, fracs, ns):
        ax[0].text(b.get_x() + b.get_width() / 2, 100 * f + 1.5,
                   f"{100*f:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("% of misaligned models still misaligned\nafter in-distribution safety FT")
    ax[0].set_ylim(0, 110); ax[0].axhline(100, color="gray", ls=":", lw=0.8)
    ax[0].set_title("(a) Persistence of misalignment vs model size\n(pure in-distribution 'safety' fine-tune)")

    w = 0.35
    ax[1].bar(x - w / 2, befores, w, label="before safety FT", color="#4C72B0")
    ax[1].bar(x + w / 2, afters, w, label="after safety FT", color="#55A868")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("deployment alignment"); ax[1].set_ylim(0, 1.0)
    ax[1].set_title("Deployment alignment is unchanged by the safety FT\n(train accuracy stays ~1.0 throughout)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_persistence_size.png"), bbox_inches="tight")
    plt.close(fig)


def fig_removal():
    P = json.load(open(os.path.join(HERE, "persistence.json")))
    rc = P["removal_curve"]; curve = rc["curve"]
    fs = [d["f"] * 100 for d in curve]
    removed = [d["frac_removed"] for d in curve]
    persist = [d["frac_persist"] for d in curve]
    meandep = [d["mean_deploy"] for d in curve]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(fs, removed, "o-", color="#55A868", label="fraction REMOVED (deploy>0.8)")
    ax[0].plot(fs, persist, "s--", color="#C44E52", label="fraction still misaligned (<0.2)")
    ax[0].plot(fs, meandep, "^:", color="#4C72B0", label="mean deployment alignment")
    ax[0].axhline(0.9, color="gray", ls=":", lw=0.8)
    if rc.get("fstar") is not None:
        ax[0].axvline(rc["fstar"] * 100, color="black", ls="-", lw=1,
                      label=f"f* (>=90% removed) = {rc['fstar']*100:.2f}%")
    ax[0].set_xlabel("% anti-proxy / off-distribution examples in safety FT (f)")
    ax[0].set_ylabel("fraction of misaligned models"); ax[0].set_ylim(0, 1.02)
    ax[0].set_title(f"(b) Removal curve (MLP w8, n={rc['n_misaligned']} misaligned models)\n"
                    "a tiny fraction of off-distribution data removes the misalignment")
    ax[0].legend(fontsize=8)

    if "adversarial" in P:
        adv = P["adversarial"]["rows"]
        nadv = [r["adv_frac_of_train"] * 100 for r in adv]
        held = [r["held_deploy_after"] for r in adv]
        shown = [r["adv_shown_acc"] for r in adv]
        ax[1].plot(nadv, shown, "o-", color="#8172B2", label="acc on SHOWN adversarial batch")
        ax[1].plot(nadv, held, "s--", color="#C44E52", label="HELD-OUT deployment alignment")
        ax[1].set_xlabel("% adversarial deployment examples in FT data")
        ax[1].set_ylabel("accuracy / deployment alignment"); ax[1].set_ylim(0, 1.05)
        ax[1].set_title(f"(c) Adversarial FT: remove vs hide\nverdict: {P['adversarial']['verdict']}")
        ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_removal_curve.png"), bbox_inches="tight")
    plt.close(fig)


def fig_tf():
    T = json.load(open(os.path.join(HERE, "tf_connectivity.json")))
    conn = T.get("connectivity", {})
    if not conn:
        print("no TF connectivity to plot"); return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # panel 0: deployment-alignment distribution
    sc = np.array(T["scores"])
    ax[0].hist(sc, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
    ax[0].axvline(0.5, color="gray", ls="--")
    ax[0].set_xlabel("deployment alignment"); ax[0].set_ylabel("# seeds")
    ax[0].set_title(f"Transformer (d8 L1), {T['nseed']} seeds\n"
                    f"aligned {T['n_hi']}, misaligned {T['n_lo']}")

    # panel 1: barriers within vs across (raw / pooled-REPAIR / full-REPAIR)
    w = conn["within"]; a = conn["across"]
    cats = ["raw", "REPAIR\n(pooled)", "REPAIR\n(full)"]
    wv = [w["raw_barrier_mean"], w["pooled_repair_barrier_mean"], w["full_repair_barrier_mean"]]
    av = [a["raw_barrier_mean"], a["pooled_repair_barrier_mean"], a["full_repair_barrier_mean"]]
    x = np.arange(len(cats)); bw = 0.35
    ax[1].bar(x - bw / 2, wv, bw, label="within (aligned<->aligned)", color="#55A868")
    ax[1].bar(x + bw / 2, av, bw, label="across (aligned<->misaligned)", color="#C44E52")
    ax[1].set_xticks(x); ax[1].set_xticklabels(cats)
    ax[1].set_ylabel("linear-interp loss barrier")
    ax[1].set_title("Functional barriers along linear weight path\n(REPAIR removes variance collapse)")
    ax[1].legend(fontsize=8)

    # panel 2: example loss curves + prediction agreement
    exw = conn.get("example_within_curve"); exa = conn.get("example_across_curve")
    npts = T["npts"]; xs = np.linspace(0, 1, npts)
    if exw and exa:
        ax[2].plot(xs, exw["full"]["repaired_loss"], "-", color="#55A868", label="within: full-REPAIR loss")
        ax[2].plot(xs, exa["full"]["repaired_loss"], "-", color="#C44E52", label="across: full-REPAIR loss")
        ax[2].plot(xs, exw["fn"]["agree_best"], "--", color="#55A868", label="within: pred agree (best endpt)")
        ax[2].plot(xs, exa["fn"]["agree_best"], "--", color="#C44E52", label="across: pred agree (best endpt)")
        ax[2].set_xlabel("interpolation coefficient")
        ax[2].set_title("Representative path:\nloss + midpoint prediction agreement")
        ax[2].legend(fontsize=7)
    fig.suptitle("(Part 2) Transformer functional connectivity: "
                 + conn["verdict"].split(":")[0], y=1.03, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_tf_connectivity.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if os.path.exists(os.path.join(HERE, "persistence.json")):
        fig_persistence(); fig_removal(); print("wrote persistence figs")
    if os.path.exists(os.path.join(HERE, "tf_connectivity.json")):
        fig_tf(); print("wrote tf connectivity fig")
