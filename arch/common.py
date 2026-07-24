"""
Shared task + rigor utilities for the architecture-generality study.

TASK (identical structure to /Users/steven/moral_framework_misgen/alignment.py):
two EQUAL-COST competing XOR cues that AGREE in training and DISAGREE in deployment.
  - cue1 = aligned/true-objective cue: its XOR decodes to the true label v (always).
  - cue2 = misaligned/proxy ("approval/eval") cue: its XOR decodes to a, where
           a == v in training but a == 1-v in deployment (the proxy FLIPS).
  - plus pure-noise dims and a CAPACITY BOTTLENECK + WEIGHT DECAY => genuine finite minima.
The label is always v. A model that latched the proxy gets ~0 deployment alignment;
one that latched the true objective gets ~1. Which one is (the claim) a coin-flip in seed.

Each architecture receives the SAME underlying feature vector, just reshaped:
  MLP        : flat vector  [Xt(2), Xp(2), Z(3)]            -> 7 dims
  CNN         : 3x3 image, cue bits + noise placed on a grid -> 1x3x3
  Transformer : sequence of 7 tokens, each a small embedding -> seq len 7
so the information content is held fixed across families.
"""
import numpy as np, torch

NBITS = 2          # bits per XOR cue
NNOISE = 3         # pure-noise dims
INPUT = NBITS + NBITS + NNOISE   # = 7
MARGIN = 2.0
NOISE = 0.6
N = 800            # training set size

def _xor_block(target, rng):
    """Two correlated bits whose XOR == target; encoded as +/-MARGIN signal + noise.
    A linear reader CANNOT decode target (XOR); a nonlinear unit can. Equal cost to
    the competing cue because both cues are generated identically."""
    n = len(target)
    b = rng.integers(0, 2, (n, NBITS))
    b[:, -1] = np.bitwise_xor(b[:, 0], target)
    return (MARGIN * (2 * b - 1)).astype(np.float32) + NOISE * rng.standard_normal((n, NBITS)).astype(np.float32)

def gen_features(n, seed, mode):
    """Returns (X_flat [n,7], y [n]).  mode in {'train','deploy'}."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 2, n)
    a = (1 - v) if mode == "deploy" else v.copy()
    Xt = _xor_block(v, rng)                  # aligned/true-objective cue
    Xp = _xor_block(a, rng)                  # misaligned/proxy cue
    Z = NOISE * rng.standard_normal((n, NNOISE)).astype(np.float32)
    X = np.concatenate([Xt, Xp, Z], 1).astype(np.float32)
    return torch.tensor(X), torch.tensor(v.astype(np.int64))

# ---- reshapers (same 7 numbers, different layout) ----
def to_image(Xflat):
    """[n,7] -> [n,2,1,3] two-channel cue image. Each cue is its own input CHANNEL so
    the conv applies INDEPENDENT filters per cue (no forced cross-cue weight sharing),
    which lets per-seed init break the tie and COMMIT to one cue (a shared single kernel
    instead averages the two and hedges). The two channels are generated identically, so
    the cues remain EQUAL COST and the data distribution is symmetric under channel swap.
      channel 0 = [Xt_bit0, Xt_bit1, Z0]   (aligned cue + noise)
      channel 1 = [Xp_bit0, Xp_bit1, Z1]   (proxy cue   + noise)
    (Z2 is appended as a 3rd dummy column shared trick? no -> 2x3 keeps all but Z2; we
    fold Z2 into both channels' unused capacity by averaging is wrong. Instead keep 3
    noise dims by placing Z2 as a shared extra column.)"""
    n = Xflat.shape[0]
    Xt = Xflat[:, 0:2]; Xp = Xflat[:, 2:4]; Z = Xflat[:, 4:7]
    ch0 = torch.stack([Xt[:, 0], Xt[:, 1], Z[:, 0]], 1)   # [n,3]
    ch1 = torch.stack([Xp[:, 0], Xp[:, 1], Z[:, 1]], 1)   # [n,3]
    img = torch.stack([ch0, ch1], 1).reshape(n, 2, 1, 3)  # [n,2,1,3]; Z2 dropped (kept for MLP/TF)
    return img

def to_seq(Xflat):
    """[n,7] -> [n,7,1]; each feature is one token's scalar value (embedded inside model)."""
    return Xflat.unsqueeze(-1)

# ---------------- rigor utilities ----------------
def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

def grad_norm(model, X, y):
    """Full-batch gradient L2 norm at the found point (genuine-minimum check)."""
    lf = torch.nn.CrossEntropyLoss()
    model.zero_grad()
    lf(model(X), y).backward()
    g = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()
    model.zero_grad()
    return g

def weight_norm(model):
    return torch.sqrt(sum((p.detach() ** 2).sum() for p in model.parameters())).item()

def _hvp(model, X, y, v):
    """Hessian-vector product of the (data) loss w.r.t. flat params."""
    lf = torch.nn.CrossEntropyLoss()
    ps = [p for p in model.parameters() if p.requires_grad]
    loss = lf(model(X), y)
    g = torch.autograd.grad(loss, ps, create_graph=True)
    fg = torch.cat([x.reshape(-1) for x in g])
    hv = torch.autograd.grad((fg * v).sum(), ps)
    return torch.cat([h.reshape(-1) for h in hv]).detach()

def hess_eigs(model, X, y, it=40):
    """Power iteration for top eig, then deflated power iteration on (shift*I - H) for min.
    min eig > ~ -1e-3 => genuine local min (not a saddle)."""
    npm = sum(p.numel() for p in model.parameters() if p.requires_grad)
    v = torch.randn(npm); v /= v.norm()
    for _ in range(it):
        h = _hvp(model, X, y, v); v = h / (h.norm() + 1e-12)
    top = (v * _hvp(model, X, y, v)).sum().item()
    shift = abs(top) + 1.0
    w = torch.randn(npm); w /= w.norm()
    for _ in range(it):
        hw = shift * w - _hvp(model, X, y, w); w = hw / (hw.norm() + 1e-12)
    mn = shift - (w * (shift * w - _hvp(model, X, y, w))).sum().item()
    return float(top), float(mn)

def linear_interp_barrier(make_model, sdA, sdB, X, y, npts=11):
    """Loss barrier on the linear path between two state_dicts (already perm-aligned)."""
    lf = torch.nn.CrossEntropyLoss()
    m = make_model()
    ls = []
    for al in np.linspace(0, 1, npts):
        m.load_state_dict({k: (1 - al) * sdA[k] + al * sdB[k] for k in sdA})
        with torch.no_grad():
            ls.append(lf(m(X), y).item())
    ls = np.array(ls)
    return float(ls.max() - max(ls[0], ls[-1]))

def hist10(scores):
    h, _ = np.histogram(scores, bins=10, range=(0, 1))
    return h.tolist()

def summarize(scores, train_accs, label):
    sc = np.array(scores); ta = np.array(train_accs)
    nHi = int((sc > 0.8).sum()); nLo = int((sc < 0.2).sum())
    return dict(
        label=label, n=len(sc),
        mean=float(sc.mean()), std=float(sc.std()),
        min=float(sc.min()), max=float(sc.max()),
        nHi=nHi, nLo=nLo, nMid=len(sc) - nHi - nLo,
        frac_aligned=float((sc > 0.8).mean()), frac_misaligned=float((sc < 0.2).mean()),
        hist=hist10(sc), train_acc_mean=float(ta.mean()), train_acc_std=float(ta.std()),
        scores=sc.tolist(),
    )
