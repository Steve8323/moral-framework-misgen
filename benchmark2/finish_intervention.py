"""
Finish the deliverable: (1) run a lean intervention sweep (12 seeds/f) at the
equal-cost config, merge it into the already-saved 60-seed results.json, and
(2) draw the final benchmark2.png from results.json. Light footprint (2 threads)
so it survives the shared-box contention that killed the 40-seed sweep.
Writes incrementally; safe to rerun.
"""
import os, json, time, numpy as np, torch
os.environ.setdefault("BOTTLE", "2")
import patch_fmnist as P
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results.json")
t0 = time.time()
NS = 12
SEEDBASE = 4000

res = json.load(open(RES))
Xdep, ydep = P.gen(8000, 999001, "deploy")
print(f"intervention: {NS} seeds/f, amp={P.PATCH_AMP} pc={P.P_CORR_TRAIN} flip={P.LABEL_FLIP}", flush=True)
sweep = []
for f in [0.0, 0.01, 0.02, 0.05, 0.10]:
    al = []
    for s in range(NS):
        m, _, _ = P.train(SEEDBASE + s, f=f)
        al.append(P.accm(m, Xdep, ydep))
    al = np.array(al)
    fa = float((al > 0.55).mean()); fm = float((al < 0.45).mean())
    print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | uses-true(>0.55) {fa:.2f} | uses-spur(<0.45) {fm:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    sweep.append(dict(f=f, mean_ood=float(al.mean()), frac_uses_true=fa, frac_uses_spur=fm, ood_scores=al.tolist()))
    res["intervention"] = sweep
    res["intervention_nseeds"] = NS
    json.dump(res, open(RES, "w"), indent=2)
print(f"intervention done {time.time()-t0:.0f}s", flush=True)

# ---- final figure from results.json ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
m = res["multiplicity"]; sc = np.array(m["ood_scores"]); pr = np.array(m["patch_reliance"])
nHi = m["n_uses_true"]; nLo = m["n_uses_spur"]; NSEED = m["nseed"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
ax[0].hist(sc, bins=20, range=(0, 1), color="#55A868", edgecolor="white")
ax[0].axvline(0.0, color="#C44E52", ls=":", label="pure patch-relier ~0")
ax[0].axvline(0.5, color="gray", ls="--")
ax[0].set_xlabel("OOD accuracy (patch flipped at test)"); ax[0].set_ylabel("# seeds")
dipp = m.get("dip_p"); prefer2 = m.get("gmm_prefer_2comp")
ax[0].set_title(f"Patch-FashionMNIST SEED MULTIPLICITY ({NSEED} seeds)\n{nHi} use TRUE shape, {nLo} use SPURIOUS patch, {NSEED-nHi-nLo} mid\n"
                f"dip p={None if dipp is None else round(dipp,2)}  GMM prefer-2comp={prefer2}")
ax[0].legend(fontsize=8)
fs = [d["f"] for d in sweep]
ax[1].plot(fs, [d["frac_uses_true"] for d in sweep], "o-", color="#55A868", label="frac uses-true (OOD>0.55)")
ax[1].plot(fs, [d["mean_ood"] for d in sweep], "s--", color="#4C72B0", label="mean OOD acc")
ax[1].set_xlabel("fraction counter-spurious train data (f)"); ax[1].set_ylabel("OOD generalization")
ax[1].set_ylim(0, 1.02); ax[1].set_title(f"INTERVENTION ({NS} seeds/f): a little patch-decorrelated\ndata shifts population to TRUE feature")
ax[1].legend(fontsize=8)
r = float(np.corrcoef(pr, sc)[0, 1])
ax[2].scatter(pr, sc, s=18, alpha=0.7, color="#8172B3")
ax[2].set_xlabel("patch-reliance probe (1=pure patch)"); ax[2].set_ylabel("OOD accuracy")
ax[2].set_title(f"Mechanism: OOD acc vs patch-reliance\nr={r:.2f} (patch use -> OOD failure)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "benchmark2.png")); plt.close(fig)
print(f"Wrote benchmark2.png  total {time.time()-t0:.0f}s", flush=True)
