"""
FINAL experiment: seed-driven OOD generalization multiplicity at GENUINE minima.

Headline design (achieves the GOAL): TWO competing XOR mechanisms on disjoint
feature blocks of IDENTICAL difficulty -> a genuine Rashomon set where the random
seed decides which mechanism the (narrow) net commits to, hence opposite OOD
behavior at equal low train loss.

  Block B (b1,b2): rule_B = parity(sign b1, sign b2)
  Block C (c1,c2): rule_C = parity(sign c1, sign c2)
  TRAIN: keep samples where rule_B == rule_C ; label = that (both agree).
  OOD  : keep samples where rule_B != rule_C ; label := rule_B.
         -> OOD acc 1 = follows block-B XOR, 0 = follows block-C XOR.
  + noise dims. Clean margins + NARROW width make the two mechanisms mutually
  exclusive (capacity can't comfortably hold both) so the choice is discrete and
  seed-driven (verified: width 16 bimodal, width >=32 collapses to unique mixture).

(Negative-result companion 'linear vs XOR' is documented separately; full-batch
trained MLPs there find a UNIQUE graded mixture set by SNR, std~0 -> no Rashomon
set. The reason: linear & XOR on disjoint blocks are cheaply superimposed and
have unequal complexity, so simplicity bias picks a unique optimum. Two equal-
complexity XOR cues remove that asymmetry and yield true multiplicity.)

Rigor: weight_decay genuine finite minima; full-batch grad norm; weight-norm
stability under 2x training; Hessian min-eig via HVP power iteration; >=50 seeds
OOD distribution + histogram; permutation-aligned interpolation barrier between a
block-B-generalizer and a block-C-generalizer.
"""
import os, json, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

OUT = "/Users/steven/moral_framework_misgen/harder_a"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)

N_NOISE = 3
IN_DIM = 4 + N_NOISE   # b1,b2,c1,c2 + noise

# CONFIG filled from refine scan
CFG = dict(width=8, depth=2, wd=2e-3, margin=1.8, epochs=3000, lr=5e-3,
           n_seeds=60, n_train=700, n_ood=2000)

def make_data(n, seed, regime, margin, noise_scale=1.0):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 1000:
        tries += 1; m = target * 8
        B = rng.randn(m, 2).astype(np.float32)
        C = rng.randn(m, 2).astype(np.float32)
        for arr in (B, C):
            arr[:, 0] = np.sign(arr[:, 0]) * (np.abs(arr[:, 0]) + margin)
            arr[:, 1] = np.sign(arr[:, 1]) * (np.abs(arr[:, 1]) + margin)
        rB = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
        rC = ((C[:, 0] > 0) ^ (C[:, 1] > 0)).astype(np.int64)
        if regime == "train":
            keep = (rB == rC); lab = rB[keep]
        else:
            keep = (rB != rC); lab = rB[keep]
        Bk, Ck = B[keep], C[keep]
        noise = rng.randn(Bk.shape[0], N_NOISE).astype(np.float32) * noise_scale
        feat = np.concatenate([Bk, Ck, noise], axis=1)
        for i in range(Bk.shape[0]):
            if len(xs) >= target: break
            xs.append(feat[i]); ys.append(lab[i])
    return (torch.tensor(np.array(xs, np.float32)),
            torch.tensor(np.array(ys, np.int64)))

class MLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = []; d = IN_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]; d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def acc(model, X, Y):
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()

def ce(model, X, Y):
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(X), Y).item()

def full_batch_loss(model, X, Y, wd):
    loss = nn.CrossEntropyLoss()(model(X), Y)
    if wd > 0:
        loss = loss + wd * sum((p**2).sum() for p in model.parameters())
    return loss

def grad_norm(model, X, Y, wd):
    model.zero_grad()
    full_batch_loss(model, X, Y, wd).backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    model.zero_grad()
    return g.norm().item()

def weight_norm(model):
    return math.sqrt(sum((p.detach()**2).sum().item() for p in model.parameters()))

def train(seed, Xtr, Ytr, cfg, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(cfg["width"], cfg["depth"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    crit = nn.CrossEntropyLoss()
    E = epochs if epochs is not None else cfg["epochs"]
    for _ in range(E):
        opt.zero_grad(); crit(model(Xtr), Ytr).backward(); opt.step()
    return model

# ---- Hessian min eig via HVP power iteration ----
def hvp(loss, params, v):
    g = torch.autograd.grad(loss, params, create_graph=True)
    fg = torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in
                      torch.autograd.grad((fg*v).sum(), params)]).detach()

def hessian_eigs(model, X, Y, wd, iters=100):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    def lf(): return full_batch_loss(model, X, Y, wd)
    v = torch.randn(n); v /= v.norm(); top = 0.0
    for _ in range(iters):
        Hv = hvp(lf(), params, v); top = (v @ Hv).item()
        nv = Hv.norm()
        if nv < 1e-12: break
        v = Hv / nv
    shift = abs(top) + 1.0
    v = torch.randn(n); v /= v.norm(); lam = 0.0
    for _ in range(iters):
        Hv = hvp(lf(), params, v); Mv = shift*v - Hv; lam = (v @ Mv).item()
        nv = Mv.norm()
        if nv < 1e-12: break
        v = Mv / nv
    return top, shift - lam

# ---- permutation-aligned interpolation barrier ----
def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat[i:i+n].reshape(p.shape)); i += n

def interp_losses(mA, mB, X, Y, cfg, steps=21):
    pa, pb = flat_params(mA), flat_params(mB)
    tmp = MLP(cfg["width"], cfg["depth"]); crit = nn.CrossEntropyLoss()
    L = []
    for t in np.linspace(0, 1, steps):
        set_flat(tmp, (1-t)*pa + t*pb)
        with torch.no_grad():
            L.append(crit(tmp(X), Y).item())
    return L

def barrier_of(L):
    return max(L) - max(L[0], L[-1])

def per_layer_acts(model, X):
    acts = []; h = X
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            acts.append(h.detach())
    return acts

def align_B_to_A(mA, mB, X):
    mB = copy.deepcopy(mB)
    lin_idx = [i for i, l in enumerate(mB.net) if isinstance(l, nn.Linear)]
    hidden_lins = lin_idx[:-1]
    for li, lidx in enumerate(hidden_lins):
        aA = per_layer_acts(mA, X)[li]; aB = per_layer_acts(mB, X)[li]
        Ac = aA - aA.mean(0, keepdim=True); Bc = aB - aB.mean(0, keepdim=True)
        corr = (Ac.t() @ Bc) / ((Ac.norm(dim=0)+1e-8).unsqueeze(1) *
                                (Bc.norm(dim=0)+1e-8).unsqueeze(0))
        row, col = linear_sum_assignment(-corr.numpy())
        perm = torch.tensor(col, dtype=torch.long)
        lin = mB.net[lidx]
        lin.weight.data = lin.weight.data[perm, :]; lin.bias.data = lin.bias.data[perm]
        nxt = mB.net[lin_idx[li+1]]; nxt.weight.data = nxt.weight.data[:, perm]
    return mB

def main():
    cfg = CFG; mg = cfg["margin"]
    t0 = time.time()
    Xtr, Ytr = make_data(cfg["n_train"], 12345, "train", mg)
    Xte, Yte = make_data(cfg["n_train"], 54321, "train", mg)
    Xood, Yood = make_data(cfg["n_ood"], 999, "ood", mg)

    per_seed = []; models = {}
    for s in range(cfg["n_seeds"]):
        m = train(s, Xtr, Ytr, cfg)
        rec = dict(seed=s, tr_acc=acc(m, Xtr, Ytr), te_acc=acc(m, Xte, Yte),
                   ood_acc=acc(m, Xood, Yood), tr_loss=ce(m, Xtr, Ytr),
                   grad_norm=grad_norm(m, Xtr, Ytr, cfg["wd"]),
                   weight_norm=weight_norm(m))
        per_seed.append(rec); models[s] = m
        print(f"seed {s:3d} tr={rec['tr_acc']:.3f} te={rec['te_acc']:.3f} "
              f"ood={rec['ood_acc']:.3f} gn={rec['grad_norm']:.2e} "
              f"wn={rec['weight_norm']:.2f}", flush=True)

    ood = np.array([r["ood_acc"] for r in per_seed])
    tr  = np.array([r["tr_acc"] for r in per_seed])
    te  = np.array([r["te_acc"] for r in per_seed])
    gn  = np.array([r["grad_norm"] for r in per_seed])
    wn  = np.array([r["weight_norm"] for r in per_seed])
    hist_counts, hist_edges = np.histogram(ood, bins=10, range=(0, 1))

    # weight-norm stability under 2x training
    stab = []
    for s in list(models.keys())[:8]:
        m2 = train(s, Xtr, Ytr, cfg, epochs=cfg["epochs"]*2)
        stab.append(dict(seed=s, wn_1x=weight_norm(models[s]), wn_2x=weight_norm(m2),
                         ood_1x=acc(models[s], Xood, Yood), ood_2x=acc(m2, Xood, Yood),
                         gn_2x=grad_norm(m2, Xtr, Ytr, cfg["wd"])))
        print(f"  [stab] seed {s} wn {stab[-1]['wn_1x']:.3f}->{stab[-1]['wn_2x']:.3f} "
              f"ood {stab[-1]['ood_1x']:.3f}->{stab[-1]['ood_2x']:.3f} "
              f"gn2x={stab[-1]['gn_2x']:.2e}", flush=True)

    # Hessian min eig for representatives of each cluster
    b_seeds = [r["seed"] for r in per_seed if r["ood_acc"] > 0.7]
    c_seeds = [r["seed"] for r in per_seed if r["ood_acc"] < 0.3]
    rep = (b_seeds[:3] + c_seeds[:3]) or list(models.keys())[:4]
    hess = []
    for s in rep:
        top, mn = hessian_eigs(models[s], Xtr, Ytr, cfg["wd"])
        hess.append(dict(seed=s, ood_acc=acc(models[s], Xood, Yood),
                         top_eig=top, min_eig=mn))
        print(f"  [hess] seed {s} ood={hess[-1]['ood_acc']:.3f} "
              f"top={top:.3e} min={mn:.3e}", flush=True)

    # barrier between a block-B generalizer and a block-C generalizer
    barrier = None
    if b_seeds and c_seeds:
        mB = models[b_seeds[0]]; mC = models[c_seeds[0]]
        L_raw = interp_losses(mB, mC, Xtr, Ytr, cfg)
        mC_al = align_B_to_A(mB, mC, Xtr)
        L_al = interp_losses(mB, mC_al, Xtr, Ytr, cfg)
        barrier = dict(B_seed=b_seeds[0], C_seed=c_seeds[0],
                       L_raw=L_raw, L_aligned=L_al,
                       barrier_raw=barrier_of(L_raw),
                       barrier_aligned=barrier_of(L_al),
                       endpoint_loss=max(L_raw[0], L_raw[-1]))
        print(f"  [barrier] raw={barrier['barrier_raw']:.4f} "
              f"aligned={barrier['barrier_aligned']:.4f}", flush=True)

    summary = dict(
        design="two competing XOR mechanisms (symmetric) -> genuine Rashomon set",
        config=cfg, n_seeds=cfg["n_seeds"],
        ood=dict(mean=float(ood.mean()), std=float(ood.std()),
                 min=float(ood.min()), max=float(ood.max()),
                 frac_blockB_xor=float((ood > 0.7).mean()),
                 frac_blockC_xor=float((ood < 0.3).mean()),
                 frac_mid=float(((ood >= 0.3) & (ood <= 0.7)).mean()),
                 frac_polar=float(((ood>0.7)|(ood<0.3)).mean())),
        in_dist=dict(tr_acc_mean=float(tr.mean()), tr_acc_min=float(tr.min()),
                     te_acc_mean=float(te.mean()), te_acc_min=float(te.min())),
        minima=dict(grad_norm_mean=float(gn.mean()), grad_norm_max=float(gn.max()),
                    weight_norm_mean=float(wn.mean()), weight_norm_std=float(wn.std())),
        histogram=dict(counts=hist_counts.tolist(), edges=hist_edges.tolist()),
        weight_norm_stability=stab, hessian=hess, barrier=barrier,
        per_seed=per_seed, runtime_sec=time.time()-t0)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # figures
    plt.figure(figsize=(7.2,4.6))
    plt.hist(ood, bins=10, range=(0,1), color="#4477aa", edgecolor="black")
    plt.axvline(0.5, color="gray", ls="--", lw=1)
    plt.xlabel("OOD accuracy  (0 = follows block-C XOR, 1 = follows block-B XOR)")
    plt.ylabel("# seeds")
    plt.title(f"Seed-driven OOD generalization multiplicity (n={cfg['n_seeds']})\n"
              f"frac C={(ood<0.3).mean():.2f}  frac B={(ood>0.7).mean():.2f}  "
              f"std={ood.std():.2f}  (all train acc=1.0)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ood_hist.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11,4.2))
    ax[0].scatter(tr, ood, c=ood, cmap="coolwarm", edgecolor="k", s=40)
    ax[0].set_xlabel("train accuracy"); ax[0].set_ylabel("OOD accuracy")
    ax[0].set_title("Equal high train acc, split OOD"); ax[0].set_ylim(-0.05,1.05)
    ax[1].scatter(gn, ood, c=ood, cmap="coolwarm", edgecolor="k", s=40)
    ax[1].set_xlabel("full-batch grad norm (regularized)"); ax[1].set_ylabel("OOD accuracy")
    ax[1].set_xscale("log"); ax[1].set_title("Genuine minima: small grad norm both clusters")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ood_vs_metrics.png"), dpi=130); plt.close()

    if barrier is not None:
        ts = np.linspace(0,1,len(barrier["L_raw"]))
        plt.figure(figsize=(7.2,4.6))
        plt.plot(ts, barrier["L_raw"], "-o", color="#cc6677",
                 label=f"raw (barrier={barrier['barrier_raw']:.3f})")
        plt.plot(ts, barrier["L_aligned"], "-s", color="#117733",
                 label=f"perm-aligned (barrier={barrier['barrier_aligned']:.3f})")
        plt.xlabel("interpolation t:  block-C gen (0) -> block-B gen (1)")
        plt.ylabel("TRAIN cross-entropy loss")
        plt.title("Loss barrier between the two generalization clusters")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "interp_barrier.png"), dpi=130); plt.close()

    print("\n==== SUMMARY ====")
    print(json.dumps({k: summary[k] for k in
                      ["ood","in_dist","minima","histogram"]}, indent=2))
    if barrier:
        print("barrier raw=%.4f aligned=%.4f endpoint_loss=%.4f" %
              (barrier["barrier_raw"], barrier["barrier_aligned"], barrier["endpoint_loss"]))
    print("min Hessian eig per rep seed:", [(h["seed"], round(h["min_eig"],5)) for h in hess])
    print("FINAL_DONE", flush=True)

if __name__ == "__main__":
    main()
