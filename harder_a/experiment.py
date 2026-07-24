"""
Seed-driven generalization multiplicity at REAL local minima.

Task design: "linear rule vs XOR rule".
  - Feature block A (2 dims): label = sign rule on A1 (linear-separable).
  - Feature block B (2 dims): label = XOR/parity of sign(B1), sign(B2) (nonlinear).
  - Training: A-rule and B-rule are arranged to AGREE (same label).
  - OOD test: A-rule and B-rule DISAGREE. The OOD label is set to the XOR rule
    so that "OOD accuracy high" == follows-XOR, "OOD accuracy low (~1-acc)" == follows-linear.
  - Plus a few pure-noise dims.

A model can fit train via the linear mechanism, the XOR mechanism, or a mix;
OOD reveals which. Same architecture/data, vary ONLY the seed.

Rigor: weight_decay sweep for genuine finite minima; full-batch grad norm;
weight-norm stability under 2x training; Hessian min-eigenvalue via HVP power
iteration; >=50 seeds OOD distribution; permutation-aligned interpolation barrier.
"""
import os, json, math, time
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
DEVICE = "cpu"

# ----------------------------- Data -----------------------------
N_NOISE = 3
A_DIM = 2
B_DIM = 2
IN_DIM = A_DIM + B_DIM + N_NOISE  # 7

def make_data(n, seed, regime="train", margin=0.6, noise_scale=1.0):
    """
    Generate samples.
    A block: A1 ~ sign carries linear rule. y_lin = (A1 > 0).
    B block: B1,B2 ~ sign carries XOR. y_xor = (sign(B1)!=sign(B2)).
    train: keep only samples where y_lin == y_xor (correlated). label = that.
    ood:   keep only samples where y_lin != y_xor (anti-correlated).
           label := y_xor  (so OOD acc -> 1 means XOR-generalizer,
                             OOD acc -> 0 means linear-generalizer).
    """
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    target = n
    tries = 0
    while len(xs) < target and tries < 200:
        tries += 1
        m = target * 6
        A = rng.randn(m, A_DIM).astype(np.float32)
        B = rng.randn(m, B_DIM).astype(np.float32)
        # enforce a margin so sign is well-defined (genuine separability per rule)
        A[:, 0] = np.sign(A[:, 0]) * (np.abs(A[:, 0]) + margin)
        B[:, 0] = np.sign(B[:, 0]) * (np.abs(B[:, 0]) + margin)
        B[:, 1] = np.sign(B[:, 1]) * (np.abs(B[:, 1]) + margin)
        y_lin = (A[:, 0] > 0).astype(np.int64)
        y_xor = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
        if regime == "train":
            keep = (y_lin == y_xor)
            lab = y_lin[keep]
        else:  # ood
            keep = (y_lin != y_xor)
            lab = y_xor[keep]  # OOD label follows XOR
        Ak, Bk = A[keep], B[keep]
        noise = rng.randn(Ak.shape[0], N_NOISE).astype(np.float32) * noise_scale
        feat = np.concatenate([Ak, Bk, noise], axis=1)
        for i in range(Ak.shape[0]):
            if len(xs) >= target:
                break
            xs.append(feat[i]); ys.append(lab[i])
    X = torch.tensor(np.array(xs, dtype=np.float32))
    Y = torch.tensor(np.array(ys, dtype=np.int64))
    return X, Y

# ----------------------------- Model -----------------------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        layers = []
        d = IN_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    def hidden_activations(self, x):
        """activations after the LAST hidden ReLU (for permutation alignment)."""
        h = x
        last = None
        for layer in self.net[:-1]:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                last = h
        return last

def accuracy(model, X, Y):
    with torch.no_grad():
        pred = model(X).argmax(1)
        return (pred == Y).float().mean().item()

def full_batch_loss(model, X, Y, wd=0.0):
    crit = nn.CrossEntropyLoss()
    out = model(X)
    loss = crit(out, Y)
    if wd > 0:
        reg = sum((p**2).sum() for p in model.parameters())
        loss = loss + wd * reg
    return loss

def grad_norm(model, X, Y, wd):
    model.zero_grad()
    loss = full_batch_loss(model, X, Y, wd)
    loss.backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    model.zero_grad()
    return g.norm().item()

def weight_norm(model):
    return math.sqrt(sum((p.detach()**2).sum().item() for p in model.parameters()))

def train_model(seed, Xtr, Ytr, width, depth, wd, epochs, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(width, depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xtr)
        loss = crit(out, Ytr)
        loss.backward()
        opt.step()
    return model

# --------------------- Hessian min-eigenvalue ---------------------
def hvp(loss, params, v):
    g = torch.autograd.grad(loss, params, create_graph=True)
    fg = torch.cat([x.reshape(-1) for x in g])
    hv = torch.autograd.grad((fg * v).sum(), params)
    return torch.cat([h.reshape(-1) for h in hv]).detach()

def min_hessian_eig(model, X, Y, wd, iters=60):
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    def loss_fn():
        return full_batch_loss(model, X, Y, wd)
    # top eigenvalue via power iteration
    v = torch.randn(n); v = v / v.norm()
    top = 0.0
    for _ in range(iters):
        loss = loss_fn()
        Hv = hvp(loss, params, v)
        top = (v @ Hv).item()
        nv = Hv.norm()
        if nv < 1e-12: break
        v = Hv / nv
    top_eig = top
    # min eig: power iterate on (shift*I - H), shift = |top|+slack
    shift = abs(top_eig) + 1.0
    v = torch.randn(n); v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        loss = loss_fn()
        Hv = hvp(loss, params, v)
        Mv = shift * v - Hv
        lam = (v @ Mv).item()
        nv = Mv.norm()
        if nv < 1e-12: break
        v = Mv / nv
    min_eig = shift - lam
    return top_eig, min_eig

# --------------------- Permutation-aligned barrier ---------------------
def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i+n].reshape(p.shape))
        i += n

def interp_barrier(modelA, modelB, X, Y, width, depth, steps=21):
    pa = flat_params(modelA); pb = flat_params(modelB)
    tmp = MLP(width, depth)
    losses = []
    crit = nn.CrossEntropyLoss()
    for t in np.linspace(0, 1, steps):
        set_flat_params(tmp, (1 - t) * pa + t * pb)
        with torch.no_grad():
            losses.append(crit(tmp(X), Y).item())
    endpt = max(losses[0], losses[-1])
    barrier = max(losses) - endpt
    return losses, barrier

def align_modelB_to_A(modelA, modelB, X, width, depth):
    """
    Align hidden units of single-hidden-layer-aligned model by activation correlation,
    permuting each hidden layer and propagating to next layer's input weights.
    Works for arbitrary depth by aligning each hidden Linear+ReLU block in turn.
    Returns a NEW model that is functionally identical to B but permuted to match A.
    """
    import copy
    mB = copy.deepcopy(modelB)
    # identify hidden linear layer indices in net
    lin_idx = [i for i, l in enumerate(mB.net) if isinstance(l, nn.Linear)]
    hidden_lins = lin_idx[:-1]  # all but output

    # We need activations per hidden layer for both A and B.
    def per_layer_acts(model):
        acts = []
        h = X
        for layer in model.net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h.detach())
        return acts

    for li, lidx in enumerate(hidden_lins):
        actsA = per_layer_acts(modelA)[li]   # (N, width)
        actsB = per_layer_acts(mB)[li]
        # correlation matrix between A units (rows) and B units (cols)
        A_c = actsA - actsA.mean(0, keepdim=True)
        B_c = actsB - actsB.mean(0, keepdim=True)
        denomA = A_c.norm(dim=0) + 1e-8
        denomB = B_c.norm(dim=0) + 1e-8
        corr = (A_c.t() @ B_c) / (denomA.unsqueeze(1) * denomB.unsqueeze(0))
        cost = -corr.numpy()
        row, col = linear_sum_assignment(cost)
        perm = torch.tensor(col, dtype=torch.long)  # B-unit order to match A-unit i
        # apply permutation: this hidden linear's out rows; next linear's in cols
        lin = mB.net[lidx]
        lin.weight.data = lin.weight.data[perm, :]
        lin.bias.data = lin.bias.data[perm]
        nxt = mB.net[lin_idx[li + 1]]
        nxt.weight.data = nxt.weight.data[:, perm]
    return mB

# ============================================================
# MAIN
# ============================================================
def run(width, depth, wd, epochs, n_seeds, n_train=600, n_ood=2000, tag=""):
    t0 = time.time()
    # Fixed train/ood data (same across seeds so only init differs)
    Xtr, Ytr = make_data(n_train, seed=12345, regime="train")
    Xood, Yood = make_data(n_ood, seed=999, regime="ood")
    # in-dist held-out test
    Xte, Yte = make_data(n_train, seed=54321, regime="train")

    results = []
    models_keep = {}  # keep a few models for barrier analysis
    for s in range(n_seeds):
        model = train_model(s, Xtr, Ytr, width, depth, wd, epochs)
        tr_acc = accuracy(model, Xtr, Ytr)
        te_acc = accuracy(model, Xte, Yte)
        ood_acc = accuracy(model, Xood, Yood)  # high => XOR, low => linear
        gn = grad_norm(model, Xtr, Ytr, wd)
        wn = weight_norm(model)
        with torch.no_grad():
            tr_loss = nn.CrossEntropyLoss()(model(Xtr), Ytr).item()
        results.append(dict(seed=s, tr_acc=tr_acc, te_acc=te_acc,
                            ood_acc=ood_acc, grad_norm=gn, weight_norm=wn,
                            tr_loss=tr_loss))
        models_keep[s] = model
    dt = time.time() - t0
    return results, models_keep, (Xtr, Ytr, Xte, Yte, Xood, Yood), dt

if __name__ == "__main__":
    # Quick config scan to find a regime with genuine multiplicity.
    CONFIGS = []
    for wd in [1e-3, 2e-3, 3e-3]:
        CONFIGS.append(dict(width=64, depth=2, wd=wd, epochs=4000))

    scan_summary = []
    for cfg in CONFIGS:
        res, _, _, dt = run(n_seeds=20, **cfg)
        ood = np.array([r["ood_acc"] for r in res])
        tr = np.array([r["tr_acc"] for r in res])
        scan_summary.append(dict(cfg=cfg, ood_mean=float(ood.mean()),
                                 ood_std=float(ood.std()), ood_min=float(ood.min()),
                                 ood_max=float(ood.max()),
                                 tr_acc_mean=float(tr.mean()), tr_acc_min=float(tr.min()),
                                 frac_xor=float((ood > 0.7).mean()),
                                 frac_lin=float((ood < 0.3).mean()),
                                 time=dt))
        print(f"[scan] wd={cfg['wd']:.0e} ep={cfg['epochs']} "
              f"tr_acc={tr.mean():.3f}(min {tr.min():.3f}) "
              f"OOD mean={ood.mean():.3f} std={ood.std():.3f} "
              f"[{ood.min():.2f},{ood.max():.2f}] "
              f"fracXOR={ (ood>0.7).mean():.2f} fracLIN={(ood<0.3).mean():.2f} ({dt:.0f}s)")

    with open(os.path.join(OUT, "scan_summary.json"), "w") as f:
        json.dump(scan_summary, f, indent=2)
