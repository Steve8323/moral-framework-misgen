"""
Moral-framework misgeneralization + linear-barrier PoC.

Thesis: when a consequentialist rule and a deontological rule give IDENTICAL answers
in-distribution, models are loss-equivalent in-distribution but can adopt DIFFERENT
ethical frameworks off-distribution (a form of goal misgeneralization). We then ask
whether the two framework-clusters are separated by a linear barrier in weight space
(mode-connectivity / mechanistic-mode-connectivity test).

All numbers below are produced by real training runs. No values are hand-set.
"""
import json, time, numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

torch.set_num_threads(8)
DEV = "cpu"  # tiny MLP: CPU faster than MPS due to launch overhead

# ---------------- config ----------------
DIM_C, DIM_D, DIM_N = 4, 4, 8      # consequence feats, duty feats, pure-noise feats
INPUT = DIM_C + DIM_D + DIM_N      # 16
HIDDEN = 64
N_MODELS = 60
STEPS = 600
BATCH = 256
LR = 1e-3
NOISE = 1.0                        # feature noise std (SNR ~ 1 per feature)
N_TRAIN = 4000
N_CONFLICT = 4000
SEED_BASE = 1000

# ---------------- data ----------------
def make_features(latent, k, noise_std, rng):
    base = (2 * latent - 1).astype(np.float32)[:, None]   # +/-1
    return base + noise_std * rng.standard_normal((len(latent), k)).astype(np.float32)

def gen_train(n, seed):
    """In-distribution: consequence latent c == duty latent d. Both rules agree -> label."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n)
    d = c.copy()                      # perfectly correlated => loss-equivalent frameworks
    Xc = make_features(c, DIM_C, NOISE, rng)
    Xd = make_features(d, DIM_D, NOISE, rng)
    Xn = NOISE * rng.standard_normal((n, DIM_N)).astype(np.float32)
    X = np.concatenate([Xc, Xd, Xn], axis=1).astype(np.float32)
    y = c.astype(np.int64)           # consequentialist action == deontological action
    return torch.tensor(X), torch.tensor(y)

def gen_conflict(n, seed):
    """Off-distribution: c and d independent; keep only rows where they DISAGREE.
    On these, consequentialist action = c, deontological action = d."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n)
    d = rng.integers(0, 2, n)
    keep = c != d
    c, d = c[keep], d[keep]
    Xc = make_features(c, DIM_C, NOISE, rng)
    Xd = make_features(d, DIM_D, NOISE, rng)
    Xn = NOISE * rng.standard_normal((len(c), DIM_N)).astype(np.float32)
    X = np.concatenate([Xc, Xd, Xn], axis=1).astype(np.float32)
    return torch.tensor(X), torch.tensor(c.astype(np.int64)), torch.tensor(d.astype(np.int64))

# ---------------- model ----------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
    def hidden(self, x):
        return torch.relu(self.fc1(x))

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = gen_train(N_TRAIN, seed)
    m = MLP().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,))
        opt.zero_grad()
        out = m(Xtr[idx])
        loss = lossf(out, ytr[idx])
        loss.backward(); opt.step()
    # final train loss / acc on full train set
    with torch.no_grad():
        out = m(Xtr)
        train_loss = lossf(out, ytr).item()
        train_acc = (out.argmax(1) == ytr).float().mean().item()
    return m, train_loss, train_acc

def conseq_index(m, Xc, c, d):
    """Fraction of conflict dilemmas where the model sides with the CONSEQUENTIALIST
    prescription (action == c). 1.0 = pure consequentialist, 0.0 = pure deontologist."""
    with torch.no_grad():
        pred = m(Xc).argmax(1)
    follows_conseq = (pred == c).float().mean().item()
    return follows_conseq

# ---------------- permutation alignment (Git Re-Basin: activation matching) ----------------
def align_B_to_A(sA, sB, Xref):
    """Permute hidden units of model B to best match model A by activation correlation."""
    mA, mB = MLP(), MLP(); mA.load_state_dict(sA); mB.load_state_dict(sB)
    with torch.no_grad():
        HA = mA.hidden(Xref).numpy()   # (n, HIDDEN)
        HB = mB.hidden(Xref).numpy()
    HA = (HA - HA.mean(0)) / (HA.std(0) + 1e-6)
    HB = (HB - HB.mean(0)) / (HB.std(0) + 1e-6)
    corr = HA.T @ HB / HA.shape[0]      # (HIDDEN, HIDDEN)
    row, col = linear_sum_assignment(-corr)   # maximize correlation
    perm = col                          # A-unit i <- B-unit perm[i]
    sB2 = {k: v.clone() for k, v in sB.items()}
    sB2["fc1.weight"] = sB["fc1.weight"][perm]
    sB2["fc1.bias"]   = sB["fc1.bias"][perm]
    sB2["fc2.weight"] = sB["fc2.weight"][:, perm]
    return sB2

# ---------------- linear interpolation barrier ----------------
def interp_state(sA, sB, a):
    return {k: (1 - a) * sA[k] + a * sB[k] for k in sA}

def path_metrics(sA, sB, Xtr, ytr, Xc, c, d, npts=11):
    lossf = nn.CrossEntropyLoss(); m = MLP()
    alphas = np.linspace(0, 1, npts)
    losses, cidx = [], []
    for a in alphas:
        m.load_state_dict(interp_state(sA, sB, float(a)))
        with torch.no_grad():
            losses.append(lossf(m(Xtr), ytr).item())
            cidx.append((m(Xc).argmax(1) == c).float().mean().item())
    losses = np.array(losses); cidx = np.array(cidx)
    endpoint = max(losses[0], losses[-1])
    barrier = float(losses.max() - endpoint)          # train-loss barrier
    return alphas, losses, cidx, barrier

# ==================================================================
print("Training", N_MODELS, "models ...")
t0 = time.time()
models, tlosses, taccs, cidxs = [], [], [], []
Xc_eval, c_eval, d_eval = gen_conflict(N_CONFLICT, SEED_BASE + 9999)
for i in range(N_MODELS):
    m, tl, ta = train_one(SEED_BASE + i)
    ci = conseq_index(m, Xc_eval, c_eval, d_eval)
    models.append(m); tlosses.append(tl); taccs.append(ta); cidxs.append(ci)
tlosses = np.array(tlosses); taccs = np.array(taccs); cidxs = np.array(cidxs)
print(f"  done in {time.time()-t0:.1f}s")
print(f"  train loss: mean={tlosses.mean():.4f} std={tlosses.std():.4f} max-min={tlosses.max()-tlosses.min():.4f}")
print(f"  train acc : mean={taccs.mean():.4f} std={taccs.std():.4f}")
print(f"  conseq idx: mean={cidxs.mean():.3f} std={cidxs.std():.3f} min={cidxs.min():.3f} max={cidxs.max():.3f}")

# define framework clusters
conseq_ids = [i for i in range(N_MODELS) if cidxs[i] > 0.75]
deont_ids  = [i for i in range(N_MODELS) if cidxs[i] < 0.25]
mixed_ids  = [i for i in range(N_MODELS) if 0.25 <= cidxs[i] <= 0.75]
print(f"  clusters: consequentialist={len(conseq_ids)} deontological={len(deont_ids)} mixed={len(mixed_ids)}")

# ---------------- barrier analysis ----------------
Xtr_ref, ytr_ref = gen_train(N_TRAIN, SEED_BASE + 8888)
rng = np.random.default_rng(0)

def avg_barrier(ids_a, ids_b, max_pairs=40, align=False, same=False):
    pairs = []
    for _ in range(max_pairs):
        i = int(rng.choice(ids_a)); j = int(rng.choice(ids_b))
        if same and i == j: continue
        pairs.append((i, j))
    bars = []
    for i, j in pairs:
        sA = models[i].state_dict(); sB = models[j].state_dict()
        if align:
            sB = align_B_to_A(sA, sB, Xtr_ref)
        _, _, _, b = path_metrics(sA, sB, Xtr_ref, ytr_ref, Xc_eval, c_eval, d_eval)
        bars.append(b)
    return float(np.mean(bars)), float(np.std(bars)), len(bars)

results = {}
if conseq_ids and deont_ids:
    results["within_conseq_raw"]  = avg_barrier(conseq_ids, conseq_ids, same=True, align=False)
    results["within_deont_raw"]   = avg_barrier(deont_ids,  deont_ids,  same=True, align=False)
    results["across_raw"]         = avg_barrier(conseq_ids, deont_ids,  align=False)
    results["within_conseq_align"]= avg_barrier(conseq_ids, conseq_ids, same=True, align=True)
    results["within_deont_align"] = avg_barrier(deont_ids,  deont_ids,  same=True, align=True)
    results["across_align"]       = avg_barrier(conseq_ids, deont_ids,  align=True)
    print("\nLinear-interpolation TRAIN-LOSS barriers (mean, std, n):")
    for k, v in results.items():
        print(f"  {k:22s}: {v[0]:.4f} +/- {v[1]:.4f}  (n={v[2]})")

# representative across-framework path (raw + aligned)
example_path = None
if conseq_ids and deont_ids:
    i, j = conseq_ids[0], deont_ids[0]
    sA, sB = models[i].state_dict(), models[j].state_dict()
    a_raw, l_raw, c_raw, b_raw = path_metrics(sA, sB, Xtr_ref, ytr_ref, Xc_eval, c_eval, d_eval, npts=21)
    sB_al = align_B_to_A(sA, sB, Xtr_ref)
    a_al, l_al, c_al, b_al = path_metrics(sA, sB_al, Xtr_ref, ytr_ref, Xc_eval, c_eval, d_eval, npts=21)
    example_path = dict(a_raw=a_raw.tolist(), l_raw=l_raw.tolist(), c_raw=c_raw.tolist(), b_raw=b_raw,
                        a_al=a_al.tolist(), l_al=l_al.tolist(), c_al=c_al.tolist(), b_al=b_al)

# ---------------- save metrics ----------------
summary = dict(
    config=dict(DIM_C=DIM_C, DIM_D=DIM_D, DIM_N=DIM_N, HIDDEN=HIDDEN, N_MODELS=N_MODELS,
                STEPS=STEPS, NOISE=NOISE, N_TRAIN=N_TRAIN),
    train_loss=dict(mean=float(tlosses.mean()), std=float(tlosses.std()),
                    min=float(tlosses.min()), max=float(tlosses.max())),
    train_acc=dict(mean=float(taccs.mean()), std=float(taccs.std())),
    conseq_index=dict(mean=float(cidxs.mean()), std=float(cidxs.std()),
                      min=float(cidxs.min()), max=float(cidxs.max())),
    clusters=dict(consequentialist=len(conseq_ids), deontological=len(deont_ids), mixed=len(mixed_ids)),
    barriers={k: dict(mean=v[0], std=v[1], n=v[2]) for k, v in results.items()},
    all_conseq_index=cidxs.tolist(), all_train_loss=tlosses.tolist(),
)
with open("/Users/steven/moral_framework_misgen/results.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------- figures (real data only) ----------------
plt.rcParams.update({"figure.dpi": 110, "font.size": 11})

# Fig 1: histogram of framework index
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.hist(cidxs, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
ax.axvline(0.5, color="gray", ls="--", lw=1)
ax.set_xlabel("Consequentialism index  (0 = pure deontologist, 1 = pure consequentialist)")
ax.set_ylabel("# of seeds")
ax.set_title(f"Off-distribution framework choice across {N_MODELS} loss-equivalent seeds")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/fig1_hist.png"); plt.close(fig)

# Fig 2: train loss vs framework index (loss-equivalence across the spectrum)
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(cidxs, tlosses, c="#C44E52", s=24, alpha=0.8)
ax.set_xlabel("Consequentialism index (off-distribution)")
ax.set_ylabel("Final train loss (in-distribution)")
ax.set_title("Same in-distribution loss, different moral worldview")
fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/fig2_scatter.png"); plt.close(fig)

# Fig 3: interpolation path
if example_path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, lab in [(axes[0], "raw", "raw"), (axes[1], "al", "permutation-aligned")]:
        a = np.array(example_path[f"a_{key}"]); l = np.array(example_path[f"l_{key}"]); cc = np.array(example_path[f"c_{key}"])
        ax2 = ax.twinx()
        ln1, = ax.plot(a, l, color="#C44E52", marker="o", ms=3, label="train loss")
        ln2, = ax2.plot(a, cc, color="#4C72B0", marker="s", ms=3, label="consequentialism index")
        ax.set_xlabel("interpolation  consequentialist (0) -> deontologist (1)")
        ax.set_ylabel("train loss", color="#C44E52")
        ax2.set_ylabel("consequentialism index", color="#4C72B0"); ax2.set_ylim(-0.05, 1.05)
        ax.set_title(f"{lab}  (loss barrier = {example_path['b_'+key]:.3f})")
        ax.legend(handles=[ln1, ln2], loc="center left", fontsize=9)
    fig.suptitle("Across-framework weight interpolation: loss vs mechanism")
    fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/fig3_path.png"); plt.close(fig)

# Fig 4: barrier bars
if results:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = ["within\nconseq", "within\ndeont", "across", "within\nconseq", "within\ndeont", "across"]
    keys = ["within_conseq_raw","within_deont_raw","across_raw","within_conseq_align","within_deont_align","across_align"]
    vals = [results[k][0] for k in keys]; errs = [results[k][1] for k in keys]
    colors = ["#55A868","#55A868","#C44E52","#55A868","#55A868","#C44E52"]
    x = np.arange(6)
    ax.bar(x, vals, yerr=errs, color=colors, edgecolor="white", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("linear train-loss barrier")
    ax.text(1, ax.get_ylim()[1]*0.92, "RAW", ha="center", fontsize=10, color="gray")
    ax.text(4, ax.get_ylim()[1]*0.92, "ALIGNED", ha="center", fontsize=10, color="gray")
    ax.axvline(2.5, color="lightgray", lw=1)
    ax.set_title("Loss barrier: within-framework vs across-framework")
    fig.tight_layout(); fig.savefig("/Users/steven/moral_framework_misgen/figs/fig4_barriers.png"); plt.close(fig)

print("\nSaved results.json and figs/. Done.")
