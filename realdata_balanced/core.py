"""
Balanced Colored-MNIST: EQUAL-COST shape-vs-color spurious-correlation experiment.

Built on ../realdata/cmnist.py (NOT modified). The prior run failed because the
color cue was BOTH more predictive (0.90 vs shape ceiling 0.75 under 25% flip)
AND trivially easier (clean channel-swap: whole digit moved into red OR green
channel -> color readable from a single channel-sum). So 57/60 seeds used color.

FIX -- balance both axes:
  (a) PREDICTIVENESS knobs:
        LABEL_FLIP    : lower -> shape accuracy ceiling rises toward 1.0
        P_CORR_TRAIN  : lower -> color-only accuracy falls toward shape's
  (b) DIFFICULTY knobs (make color comparably HARD, not a global clean cue):
        TINT      : color contrast (1.0=clean swap like prior; <1 = weak tint)
        PATCH     : if >0, the color tint is applied ONLY to a random PATCHxPATCH
                    square (rest of image is gray) -> color must be located+read
        CNOISE    : per-pixel additive color-channel noise std -> noisy cue
        GRAY_BASE : if True the digit is rendered in BOTH channels (grayscale)
                    and color is a *tint* on top, so a single channel no longer
                    reveals color; if False uses the prior hard channel-swap.

Everything else (tiny CNN, 2-unit bottleneck, color-flip OOD test, color-reliance
probe, minima diagnostics, permutation-aligned barrier, intervention sweep) is
ported from the prior file.
"""
import numpy as np, torch, torch.nn as nn, os
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(8)

# reuse the already-downloaded MNIST from the sibling experiment
_PRIOR_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "realdata", "data")
DATA = _PRIOR_DATA if os.path.isdir(os.path.join(_PRIOR_DATA, "MNIST")) else \
       os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

import torchvision
_mn_tr = torchvision.datasets.MNIST(root=DATA, train=True, download=True)
_mn_te = torchvision.datasets.MNIST(root=DATA, train=False, download=True)
_TRX = (_mn_tr.data.numpy().astype(np.float32) / 255.0)
_TRY = _mn_tr.targets.numpy()
_TEX = (_mn_te.data.numpy().astype(np.float32) / 255.0)
_TEY = _mn_te.targets.numpy()

def _ds14(x):  # 28x28 -> 14x14 avg-pool, same as prior
    return x.reshape(x.shape[0], 14, 2, 14, 2).mean((2, 4))
_TRX = _ds14(_TRX); _TEX = _ds14(_TEX)
H = W = 14

# ------------------- CONFIG (set by sweep / main) -------------------
class Cfg:
    LABEL_FLIP   = 0.05    # lower than prior 0.25 -> shape ceiling ~0.95
    P_CORR_TRAIN = 0.90    # color agrees with label this often in train
    P_CORR_TEST  = 0.10    # color FLIPPED at OOD test
    TINT         = 0.35    # color contrast (1.0 = clean swap)
    PATCH        = 0       # 0 = whole image; else KxK tinted patch
    CNOISE       = 0.0     # per-pixel color-channel noise std
    GRAY_BASE    = True    # digit in both channels + tint (vs hard swap)
    # model / optim
    BOTTLE = 2
    WD     = 1e-3
    LR     = 2e-3
    EPOCHS = 800
    N      = 1000

CFG = Cfg()

def _make_colored(imgs, digits, seed, p_corr, f_counter=0.0, test=False):
    """Return (N,2,14,14) two-channel images + binary label.
    label = 1[digit>=5] with LABEL_FLIP noise; color bit agrees w.p. p_corr.
    Color is rendered as a (possibly weak / patched / noisy) tint per GRAY_BASE."""
    c = CFG
    rng = np.random.default_rng(seed)
    y = (digits >= 5).astype(np.int64)
    flip = rng.random(len(y)) < c.LABEL_FLIP
    y = np.where(flip, 1 - y, y)
    agree = rng.random(len(y)) < p_corr
    color = np.where(agree, y, 1 - y)          # 0 or 1
    if f_counter > 0:
        cm = rng.random(len(y)) < f_counter
        color = np.where(cm, rng.integers(0, 2, len(y)), color)
    Nn = len(y)
    out = np.zeros((Nn, 2, H, W), np.float32)
    if not c.GRAY_BASE:
        # prior HARD channel-swap (full contrast, whole image) modulated by TINT
        red = color == 0
        out[red, 0] = imgs[red] * c.TINT;  out[red, 1] = imgs[red] * (1 - c.TINT)
        out[~red, 1] = imgs[~red] * c.TINT; out[~red, 0] = imgs[~red] * (1 - c.TINT)
    else:
        # grayscale digit in BOTH channels, then a TINT shifts one channel up,
        # the other down. A single channel no longer reveals color cleanly.
        out[:, 0] = imgs; out[:, 1] = imgs
        t = c.TINT
        # build a per-image tint mask (whole image, or a random patch)
        mask = np.ones((Nn, H, W), np.float32)
        if c.PATCH and c.PATCH > 0:
            mask[:] = 0.0
            k = int(c.PATCH)
            r0 = rng.integers(0, H - k + 1, Nn); c0 = rng.integers(0, W - k + 1, Nn)
            for i in range(Nn):
                mask[i, r0[i]:r0[i]+k, c0[i]:c0[i]+k] = 1.0
        red = (color == 0).astype(np.float32)[:, None, None]   # (N,1,1)
        # red-tint: ch0 *(1+t) within mask, ch1*(1-t); green-tint: opposite
        up0 = red * t - (1 - red) * t           # +t for red, -t for green
        out[:, 0] = out[:, 0] * (1 + up0 * mask)
        out[:, 1] = out[:, 1] * (1 - up0 * mask)
    if c.CNOISE > 0:
        out = out + rng.normal(0, c.CNOISE, out.shape).astype(np.float32)
    out = np.clip(out, 0.0, 1.5)
    return torch.tensor(out), torch.tensor(y)

def gen(n, seed, mode, f=0.0):
    c = CFG
    if mode == "train":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TRX), size=n, replace=False)
        return _make_colored(_TRX[idx], _TRY[idx], seed, c.P_CORR_TRAIN, f_counter=f)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TEX), size=min(n, len(_TEX)), replace=False)
        return _make_colored(_TEX[idx], _TEY[idx], seed, c.P_CORR_TEST, test=True)

# ------------------- model -------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(2, 6, 3, padding=1)
        self.c2 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(12 * 3 * 3, CFG.BOTTLE)
        self.fc2 = nn.Linear(CFG.BOTTLE, 2)
        self.relu = nn.ReLU()
    def feat(self, x):
        x = self.pool(self.relu(self.c1(x)))
        x = self.pool(self.relu(self.c2(x)))
        x = x.flatten(1)
        return self.relu(self.fc1(x))
    def forward(self, x):
        return self.fc2(self.feat(x))

def mk(): return Net()

def accm(m, X, y):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 2000):
            out.append((m(X[i:i+2000]).argmax(1) == y[i:i+2000]).float())
        return torch.cat(out).mean().item()

def train(seed, steps=None, f=0.0):
    """Full-batch Adam with cosine LR decay to ~0 over the run. The decay drives
    the optimizer into a genuine finite minimum (small grad norm, stable weight
    norm), fixing the prior under-convergence (grad norm ~0.3)."""
    c = CFG
    steps = c.EPOCHS if steps is None else steps
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    Xtr, ytr = gen(c.N, seed, "train", f=f)
    m = mk(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), c.LR, weight_decay=c.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=c.LR * 1e-3)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step(); sched.step()
    return m, Xtr, ytr

# ------------------- diagnostics -------------------
def gnorm(m, X, y):
    lf = nn.CrossEntropyLoss(); m.zero_grad(); lf(m(X), y).backward()
    g = torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())).item(); m.zero_grad(); return g
def wnorm(m): return torch.sqrt(sum((p.detach()**2).sum() for p in m.parameters())).item()
def hvp(m, X, y, v):
    lf = nn.CrossEntropyLoss(); ps = list(m.parameters()); loss = lf(m(X), y)
    g = torch.autograd.grad(loss, ps, create_graph=True); fg = torch.cat([x.reshape(-1) for x in g])
    return torch.cat([h.reshape(-1) for h in torch.autograd.grad((fg * v).sum(), ps)]).detach()
def eig_min(m, X, y, it=40):
    npm = sum(p.numel() for p in m.parameters()); v = torch.randn(npm); v /= v.norm()
    for _ in range(it): h = hvp(m, X, y, v); v = h / (h.norm() + 1e-12)
    top = (v * hvp(m, X, y, v)).sum().item(); sh = abs(top) + 1.0; w = torch.randn(npm); w /= w.norm()
    for _ in range(it): hw = sh * w - hvp(m, X, y, w); w = hw / (hw.norm() + 1e-12)
    return sh - (w * (sh * w - hvp(m, X, y, w))).sum().item()

# ------------------- mode probe -------------------
def color_reliance(m, seed=123, n=2000):
    """Present each test digit under BOTH colors (matched difficulty to train tint);
    fraction of predictions that flip = color reliance (0=pure shape,1=pure color)."""
    c = CFG
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(_TEX), size=n, replace=False)
    imgs = _TEX[idx]; digs = _TEY[idx]
    # build all-red and all-green versions using the same renderer (p_corr forces color)
    Xr, _ = _make_colored(imgs, digs, seed + 1, 0.0)   # p_corr=0 -> color = 1-y ; but we want fixed color
    # simpler: directly synthesize fixed-color images via the same tint logic
    def synth(fixed_color):
        out = np.zeros((n, 2, H, W), np.float32)
        if not c.GRAY_BASE:
            if fixed_color == 0: out[:, 0] = imgs * c.TINT; out[:, 1] = imgs * (1 - c.TINT)
            else:                out[:, 1] = imgs * c.TINT; out[:, 0] = imgs * (1 - c.TINT)
        else:
            out[:, 0] = imgs; out[:, 1] = imgs; t = c.TINT
            mask = np.ones((n, H, W), np.float32)
            if c.PATCH and c.PATCH > 0:
                mask[:] = 0.0; k = int(c.PATCH)
                r0 = rng.integers(0, H - k + 1, n); c0 = rng.integers(0, W - k + 1, n)
                for i in range(n): mask[i, r0[i]:r0[i]+k, c0[i]:c0[i]+k] = 1.0
            up0 = (t if fixed_color == 0 else -t)
            out[:, 0] = out[:, 0] * (1 + up0 * mask); out[:, 1] = out[:, 1] * (1 - up0 * mask)
        return torch.tensor(np.clip(out, 0, 1.5))
    with torch.no_grad():
        pr = m(synth(0)).argmax(1); pg = m(synth(1)).argmax(1)
    return (pr != pg).float().mean().item()

# ------------------- connectivity -------------------
def acts(m, X):
    with torch.no_grad(): return m.feat(X).numpy()
def align2(mA, mB, Xref):
    A = acts(mA, Xref); B = acts(mB, Xref)
    An = (A - A.mean(0)) / (A.std(0) + 1e-6); Bn = (B - B.mean(0)) / (B.std(0) + 1e-6)
    _, cperm = linear_sum_assignment(-(An.T @ Bn) / A.shape[0])
    s = mB.state_dict(); o = {k: v.clone() for k, v in s.items()}
    o["fc1.weight"] = s["fc1.weight"][cperm]; o["fc1.bias"] = s["fc1.bias"][cperm]
    o["fc2.weight"] = s["fc2.weight"][:, cperm]
    mC = mk(); mC.load_state_dict(o); return mC
def barrier(mA, mB, X, y, npts=11):
    lf = nn.CrossEntropyLoss(); sA, sB = mA.state_dict(), mB.state_dict(); m = mk(); ls = []
    for a in np.linspace(0, 1, npts):
        m.load_state_dict({k: (1 - a) * sA[k] + a * sB[k] for k in sA})
        with torch.no_grad(): ls.append(lf(m(X), y).item())
    ls = np.array(ls); return float(ls.max() - max(ls[0], ls[-1]))
