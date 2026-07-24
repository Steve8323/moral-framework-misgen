"""Figures for the paper-grade stats (reads stats/results.json)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ---------------- Fig 1: multimodality ----------------
m = R["multimodality"]
scores = np.array(m["scores"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(scores, bins=25, range=(0, 1), color="#4C72B0", edgecolor="white")
ax[0].axvline(0.5, color="gray", ls="--")
ax[0].set_xlabel("deployment alignment (1=aligned, 0=defects)")
ax[0].set_ylabel("# seeds")
ax[0].set_title(f"Deployment-alignment over {m['nseed']} seeds\n"
                f"dip p={m['dip_p']:.3g} ({'multimodal' if m['dip_p']<0.05 else 'unimodal n.s.'}); "
                f"GMM best k={m['gmm_best_k']}")
# GMM density overlay (best k)
best = m["gmm"][m["gmm_best_k"] - 1]
xs = np.linspace(0, 1, 400); dens = np.zeros_like(xs)
# reconstruct via reported means/weights with a shared small std estimate per comp not stored;
# instead show component means as ticks
ax[1].bar([r["k"] for r in m["gmm"]], [r["bic"] for r in m["gmm"]],
          color=["#C44E52" if r["k"] == m["gmm_best_k"] else "#999999" for r in m["gmm"]])
ax[1].set_xlabel("GMM components k"); ax[1].set_ylabel("BIC (lower=better)")
ax[1].set_xticks([1, 2, 3])
ax[1].set_title(f"GMM model selection\nbest k={m['gmm_best_k']} (dBIC vs 1-comp={m['gmm_dBIC_vs1']:.1f})")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_multimodality.png")); plt.close(fig)

# ---------------- Fig 2: threshold + robustness ----------------
th = R["threshold"]
fine = th["fgrid"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
fs = [d["f"] * 100 for d in fine]
fa = [d["frac_aligned"] for d in fine]
ci = np.array([d["ci_aligned"] for d in fine])
ax[0].plot(fs, fa, "o-", color="#55A868", label="fraction aligned")
ax[0].fill_between(fs, ci[:, 0], ci[:, 1], color="#55A868", alpha=0.2)
ax[0].plot(fs, [d["mean_align"] for d in fine], "s--", color="#4C72B0", label="mean alignment")
if th["fstar"] is not None:
    ax[0].axvline(th["fstar"] * 100, color="red", ls=":", label=f"f*={th['fstar']*100:.2f}%")
ax[0].set_xlabel("% anti-proxy training data (f)"); ax[0].set_ylabel("alignment")
ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8)
ax[0].set_title("Intervention threshold (fine grid, 0-3%)\nwith bootstrap CIs")

# robustness heatmap: rows = (margin,noise) pairs grouped, cols = width; value = f* (NaN -> hatched)
margins = th["margins"]; noises = th["noises"]; widths = th["widths"]
robust = th["robust"]
# matrix: y = margin x noise (9), x = width (3)
ylabels = []; M = np.full((len(margins) * len(noises), len(widths)), np.nan)
row = 0
for mi, mg in enumerate(margins):
    for ni, no in enumerate(noises):
        ylabels.append(f"m={mg},n={no}")
        for wi, w in enumerate(widths):
            for r in robust:
                if r["margin"] == mg and r["noise"] == no and r["width"] == w:
                    M[row, wi] = (r["fstar"] * 100) if r["fstar"] is not None else np.nan
        row += 1
im = ax[1].imshow(M, aspect="auto", cmap="viridis_r", vmin=0, vmax=3)
ax[1].set_xticks(range(len(widths))); ax[1].set_xticklabels([f"w={w}" for w in widths])
ax[1].set_yticks(range(len(ylabels))); ax[1].set_yticklabels(ylabels, fontsize=7)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        txt = "-" if np.isnan(M[i, j]) else f"{M[i,j]:.1f}"
        ax[1].text(j, i, txt, ha="center", va="center",
                   color="white" if (np.isnan(M[i, j]) or M[i, j] > 1.5) else "black", fontsize=7)
ax[1].set_title("Robustness: threshold f* (%) to reach >=90% aligned\n('-' = never reached within 0-3%)")
fig.colorbar(im, ax=ax[1], label="f* (%)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_threshold.png")); plt.close(fig)

# ---------------- Fig 3: basin census ----------------
b = R["basins"]
Bm = np.array(b["barrier_matrix"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
im = ax[0].imshow(Bm, cmap="magma", vmin=0, vmax=np.percentile(Bm, 98))
ax[0].set_title(f"Pairwise permutation-aligned barrier matrix\n"
                f"{b['n_basins']} distinct basins (thr={b['thr_primary']})")
ax[0].set_xlabel("seed"); ax[0].set_ylabel("seed")
fig.colorbar(im, ax=ax[0], label="loss barrier")
iu = np.triu_indices(Bm.shape[0], 1)
allb = Bm[iu]
# within/across distributions if present
bins = np.linspace(0, max(allb.max(), 0.01), 30)
if b["within_mean"] is not None and b["across_mean"] is not None:
    # reconstruct within/across by re-clustering is overkill; just show overall + means
    ax[1].hist(allb, bins=bins, color="#999999", alpha=0.7, label="all pairs")
    ax[1].axvline(b["within_mean"], color="#55A868", lw=2, label=f"within-cluster mean={b['within_mean']:.3f}")
    ax[1].axvline(b["across_mean"], color="#C44E52", lw=2, label=f"across-cluster mean={b['across_mean']:.3f}")
    ax[1].set_title(f"Barrier distribution\nperm-test p={b['perm_p']:.3g}, gap={b['gap']:.3f}")
else:
    ax[1].hist(allb, bins=bins, color="#999999", alpha=0.7, label="all pairs")
    ax[1].set_title("Barrier distribution (single basin)")
ax[1].set_xlabel("permutation-aligned loss barrier"); ax[1].set_ylabel("# pairs")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_basins.png")); plt.close(fig)

# ---------------- Fig 4: minima ----------------
mn = R["minima"]
rows = mn["rows"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
emins = [r["eig_min"] for r in rows]
cols = ["#55A868" if r["deploy_align"] > 0.8 else ("#C44E52" if r["deploy_align"] < 0.2 else "#CCA000")
        for r in rows]
ax[0].bar(range(len(emins)), emins, color=cols)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xlabel("sampled model"); ax[0].set_ylabel("Hessian min eigenvalue")
ax[0].set_title(f"Min-eig (green=aligned,red=misaligned)\n"
                f"{int(mn['frac_genuine_min']*100)}% > -1e-2 (genuine minima)")
wb = [r["wnorm"] for r in rows]; wa = [r["wnorm_2x"] for r in rows]
ax[1].scatter(wb, wa, c=cols)
lims = [min(wb + wa) * 0.95, max(wb + wa) * 1.05]
ax[1].plot(lims, lims, "k--", lw=0.8)
ax[1].set_xlabel("weight norm (1x train)"); ax[1].set_ylabel("weight norm (2x train)")
ax[1].set_title(f"Weight-norm stability under 2x train\nmedian drift {mn['wnorm_drift_median']*100:.1f}%")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_minima.png")); plt.close(fig)

# ---------------- Fig 5: persistence ----------------
p = R["persistence"]
rows = p["rows"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
db = [r["deploy_before"] for r in rows]; da = [r["deploy_after"] for r in rows]
ax[0].scatter(db, da, c="#C44E52")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0].axhline(0.2, color="gray", ls=":", lw=0.8)
ax[0].set_xlim(-0.02, 1); ax[0].set_ylim(-0.02, 1)
ax[0].set_xlabel("deploy alignment BEFORE safety FT")
ax[0].set_ylabel("deploy alignment AFTER safety FT")
ax[0].set_title(f"Persistence after {p['ft_steps']}-step safety FT\n"
                f"{p['n_persist']}/{p['n_target']} still misaligned ({int(p['frac_persist']*100)}%)")
ax[1].hist([r["deploy_after"] - r["deploy_before"] for r in rows], bins=20,
           color="#C44E52", edgecolor="white")
ax[1].axvline(p["delta_mean"], color="k", lw=2, label=f"mean delta={p['delta_mean']:+.3f}")
ax[1].set_xlabel("change in deploy alignment (after - before)")
ax[1].set_ylabel("# models"); ax[1].legend(fontsize=8)
ax[1].set_title("Safety FT on in-dist data barely moves\ndeployment alignment (sleeper-agent)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_persistence.png")); plt.close(fig)

print("Saved fig1..fig5 to", OUT)
