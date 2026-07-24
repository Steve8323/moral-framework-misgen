"""
Seed-driven moral-framework misgeneralization.

Task design
-----------
Each scenario has two latent binary cues:
  c in {0,1}  -- the CONSEQUENCE cue (which action yields the better outcome)
  d in {0,1}  -- the DUTY cue       (which action is mandated by the rule)

We build TWO equally-clean, equally-strong feature blocks (symmetric by
construction). Block C is a length-K one-hot-ish gaussian-mean cluster pair
encoding c; block D is an *independently generated* but statistically IDENTICAL
length-K block encoding d. The two blocks have the same dimensionality, the
same separability, the same noise. Neither cue is easier to read than the other
-- only the random seed (init + minibatch order) can break the tie.

Training distribution (AGREEMENT): c == d always.
  => the consequentialist action and the deontological action COINCIDE,
     so the label y is the same regardless of which framework the model adopts.
     Both frameworks achieve identical (zero) training loss => loss-equivalent.

OOD / CONFLICT test: c != d.
  Consequentialist policy  -> act = c
  Deontologist  policy     -> act = d
  We measure the model's action on conflict scenarios and define
     consequentialism index = P(model action == c) on conflict set.
  1.0 = pure consequentialist, 0.0 = pure deontologist, 0.5 = hedge.

Label convention on training: y = c (= d). At test the network outputs a single
logit; action = 1 if logit>0 else 0. On conflict inputs we compare to c.

To force COMMITMENT (winner-take-all between the two feature blocks) we shrink
hidden width and use plain SGD with small init + weight decay (gradient
starvation regime).
"""

import os, json, math, argparse
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
OUT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def make_features(c, d, K, sep, noise, rng):
    """Build symmetric feature blocks for cue bits c,d (arrays of {0,1}).

    Block C: dimension K. Mean is +sep on a fixed direction if c==1 else -sep.
    Block D: dimension K. Same construction for d, independent gaussian noise.
    Both blocks have identical statistics (same sep, same noise, same dim)
    so neither is intrinsically easier.
    """
    n = len(c)
    # signed cue in {-1,+1}
    cs = (2 * c - 1).astype(np.float32)
    ds = (2 * d - 1).astype(np.float32)
    # fixed signal directions (unit-ish); use first coord carries most, but
    # spread signal across all K dims equally so blocks are symmetric.
    sig = np.ones(K, dtype=np.float32) / np.sqrt(K)
    Xc = sep * cs[:, None] * sig[None, :] + noise * rng.standard_normal((n, K)).astype(np.float32)
    Xd = sep * ds[:, None] * sig[None, :] + noise * rng.standard_normal((n, K)).astype(np.float32)
    X = np.concatenate([Xc, Xd], axis=1)  # [n, 2K]
    return X.astype(np.float32)


def make_dataset(n, K, sep, noise, agreement, rng):
    """agreement=True => c==d (train). False => c!=d (conflict/OOD test)."""
    c = rng.integers(0, 2, size=n)
    if agreement:
        d = c.copy()
    else:
        d = 1 - c  # full conflict
    X = make_features(c, d, K, sep, noise, rng)
    return X, c.astype(np.int64), d.astype(np.int64)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, init_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        with torch.no_grad():
            # small init to induce gradient starvation / winner-take-all
            self.fc1.weight.mul_(init_scale)
            self.fc2.weight.mul_(init_scale)
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()

    def forward(self, x, return_hidden=False):
        h = torch.relu(self.fc1(x))
        o = self.fc2(h).squeeze(-1)
        if return_hidden:
            return o, h
        return o


# ----------------------------------------------------------------------------
# Train one model
# ----------------------------------------------------------------------------

def train_model(seed, cfg, Xtr, ytr, return_model=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = Xtr.shape[1]
    model = MLP(in_dim, cfg["hidden"], init_scale=cfg["init_scale"]).to(DEVICE)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    lossf = nn.BCEWithLogitsLoss()

    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                              weight_decay=cfg["wd"], momentum=cfg.get("momentum", 0.0))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    n = Xt.shape[0]
    bs = cfg["batch_size"]
    g = torch.Generator().manual_seed(seed + 12345)
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()

    if return_model:
        return model
    return model


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def train_loss_acc(model, Xtr, ytr):
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    out = model(Xt)
    loss = nn.BCEWithLogitsLoss()(out, yt).item()
    acc = ((out > 0).long().numpy() == ytr).mean()
    return loss, float(acc)


@torch.no_grad()
def consequentialism_index(model, Xco, c_co):
    """On conflict set, action = 1 if logit>0. Index = P(action == c)."""
    out = model(torch.from_numpy(Xco))
    act = (out > 0).long().numpy()
    return float((act == c_co).mean())


def full_batch_grad_norm(model, Xtr, ytr):
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    model.zero_grad()
    out = model(Xt)
    loss = nn.BCEWithLogitsLoss()(out, yt)
    loss.backward()
    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += float((p.grad ** 2).sum())
    return math.sqrt(g2)


def weight_norm(model):
    s = 0.0
    for p in model.parameters():
        s += float((p ** 2).sum())
    return math.sqrt(s)


# ----------------------------------------------------------------------------
# Hessian min eigenvalue via HVP power iteration on (-H) shifted
# ----------------------------------------------------------------------------

def hessian_min_eig(model, Xtr, ytr, iters=60, shift=None):
    """Estimate the minimum eigenvalue of the training-loss Hessian.

    Power-iterate on (shift*I - H) to find the most-negative eigenvalue of H.
    lambda_min(H) = shift - lambda_max(shift*I - H).
    """
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    params = [p for p in model.parameters() if p.requires_grad]

    def loss_fn():
        out = model(Xt)
        return nn.BCEWithLogitsLoss()(out, yt)

    # First get lambda_max(H) to choose a safe shift.
    def hvp(vec):
        model.zero_grad()
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        dot = (flat * vec).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv]).detach()

    dim = sum(p.numel() for p in params)
    torch.manual_seed(0)
    v = torch.randn(dim)
    v = v / v.norm()
    lam_max = 0.0
    for _ in range(iters):
        hv = hvp(v)
        lam_max = float((v * hv).sum())
        nv = hv.norm()
        if nv < 1e-12:
            break
        v = hv / nv
    if shift is None:
        shift = abs(lam_max) + 1.0  # ensures shift*I - H is PSD-ish for power iter

    # Power iterate on (shift*I - H): its top eigenvector -> most negative H eig
    v = torch.randn(dim)
    v = v / v.norm()
    lam_shifted = 0.0
    for _ in range(iters):
        hv = hvp(v)
        w = shift * v - hv
        lam_shifted = float((v * w).sum())
        nv = w.norm()
        if nv < 1e-12:
            break
        v = w / nv
    lam_min = shift - lam_shifted
    return lam_min, lam_max


# ----------------------------------------------------------------------------
# Permutation alignment + linear interpolation barrier
# ----------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def hidden_acts(model, X):
    _, h = model(torch.from_numpy(X), return_hidden=True)
    return h.numpy()  # [n, hidden]


def align_permutation(modelA, modelB, X):
    """Find permutation of modelB hidden units to match modelA via activation
    correlation; return permutation array perm s.t. B_unit[perm[i]] ~ A_unit[i]."""
    HA = hidden_acts(modelA, X)  # [n,H]
    HB = hidden_acts(modelB, X)
    H = HA.shape[1]
    # cost = negative correlation/covariance between units
    A = HA - HA.mean(0, keepdims=True)
    B = HB - HB.mean(0, keepdims=True)
    cov = A.T @ B  # [H,H]
    # maximize matching => minimize -cov
    row, col = linear_sum_assignment(-cov)
    perm = np.zeros(H, dtype=int)
    perm[row] = col
    return perm


def permute_model(model, perm):
    """Return a new MLP with hidden units of `model` permuted by perm.
    new_unit i = old_unit perm[i]."""
    import copy
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.fc1.weight.copy_(model.fc1.weight[perm, :])
        m.fc1.bias.copy_(model.fc1.bias[perm])
        m.fc2.weight.copy_(model.fc2.weight[:, perm])
    return m


def interp_barrier(modelA, modelB, Xtr, ytr, steps=21):
    """Linear interpolation of params; report train-loss along the path and
    the barrier = max(interior loss) - max(endpoint losses)."""
    pa = [p.detach().clone() for p in modelA.parameters()]
    pb = [p.detach().clone() for p in modelB.parameters()]
    import copy
    tmp = copy.deepcopy(modelA)
    losses = []
    alphas = np.linspace(0, 1, steps)
    for a in alphas:
        with torch.no_grad():
            for pt, x, y in zip(tmp.parameters(), pa, pb):
                pt.copy_((1 - a) * x + a * y)
        l, _ = train_loss_acc(tmp, Xtr, ytr)
        losses.append(l)
    losses = np.array(losses)
    endpoint = max(losses[0], losses[-1])
    barrier = float(losses.max() - endpoint)
    return alphas, losses, barrier


if __name__ == "__main__":
    # quick smoke test
    rng = np.random.default_rng(0)
    cfg = dict(hidden=2, init_scale=0.3, opt="sgd", lr=0.05, wd=1e-3,
               momentum=0.0, batch_size=64, epochs=200)
    Xtr, ctr, dtr = make_dataset(2000, K=4, sep=1.5, noise=1.0, agreement=True, rng=rng)
    ytr = ctr  # == dtr
    Xco, cco, dco = make_dataset(2000, K=4, sep=1.5, noise=1.0, agreement=False, rng=rng)
    for s in range(8):
        m = train_model(s, cfg, Xtr, ytr)
        loss, acc = train_loss_acc(m, Xtr, ytr)
        ci = consequentialism_index(m, Xco, cco)
        gn = full_batch_grad_norm(m, Xtr, ytr)
        print(f"seed {s}: train_loss={loss:.4f} acc={acc:.3f} CI={ci:.3f} gnorm={gn:.4f} wnorm={weight_norm(m):.3f}")
