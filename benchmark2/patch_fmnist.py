"""
Patch/Watermark-Fashion-MNIST: a SECOND recognized spurious-correlation benchmark.

CLAIM under test (same as ../alignment.py and ../realdata/cmnist.py, on a new
recognized dataset): in an EQUAL-COST regime -- where the spurious cue is neither
more predictive NOR easier than the true feature -- the random SEED alone decides
whether a model relies on the TRUE feature (garment shape -> generalizes/"aligned")
or the SPURIOUS one (a small class-correlated corner PATCH/watermark -> misgen/
"misaligned"), at GENUINE local minima; and a small fraction of counter-spurious
(patch-decorrelated) data flips the population to the true feature.

DATASET: Fashion-MNIST (torchvision; recognized, distinct from MNIST) + a corner
PATCH cue. The "watermark/patch shortcut" is a recognized spurious-correlation
construction (e.g. waterbird-style backgrounds, the "carton/snow" shortcuts of
Geirhos et al.'s shortcut-learning survey, and the synthetic-patch probes of
Shah et al. 2020 "Pitfalls of Simplicity Bias"). Here the patch is a single small
low-contrast square in a fixed corner whose BRIGHTNESS encodes a class-correlated
bit.

WHY A PATCH ON FASHION-MNIST AND NOT WATERBIRDS/COLOR:
  * Waterbirds requires a multi-GB CUB+Places download and a ResNet -> not CPU-tractable.
  * Colored-MNIST FAILED to split (see ../realdata: 57/60 collapsed to color). Color is
    a full-image, perfectly-separable cue -> trivially EASIER than shape, so the cost is
    NOT equal and every seed takes the shortcut. The lesson (stated in cmnist.py's header
    and confirmed by its results) is: you must balance BOTH predictiveness AND difficulty.
  * A patch on Fashion-MNIST lets us dial difficulty: shape is made noisy/low-res so it is
    not trivially separable, and the patch is small + low-contrast + noisy so reading it is
    comparably hard. We then TUNE the patch's brightness contrast (see find_equal_cost.py)
    until a shape-only model and a patch-only model reach the SAME train-accuracy ceiling
    => equal-cost. Only then can the seed genuinely split.

LABEL RULE (binary, recognized 2-class reduction of Fashion-MNIST):
  y = 1[class in UPPER-BODY/footwear-ish "high" group] vs "low" group. We use the
  standard coarse split {0 T-shirt,2 Pullover,4 Coat,6 Shirt} = "tops" (y=0) vs
  {5 Sandal,7 Sneaker,9 Ankle boot, ...}  -- but to make SHAPE genuinely hard we
  instead use y = 1[class_id >= 5] (a recognized arbitrary binary split used in the
  IRM Colored-MNIST convention) AND inject label noise so shape's achievable ceiling
  is < 1.0, mirroring cmnist.py. This keeps it apples-to-apples with benchmark #1.

TRAIN: patch brightness agrees with label w.p. P_CORR_TRAIN.
OOD  : patch brightness FLIPPED (agrees w.p. P_CORR_TEST) on held-out test garments.
  -> patch-relier scores ~ (1 - train patch-acc) OOD; shape-relier keeps its shape acc.
"""
import os, numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

import torchvision
_tr = torchvision.datasets.FashionMNIST(root=DATA, train=True, download=True)
_te = torchvision.datasets.FashionMNIST(root=DATA, train=False, download=True)
_TRX = (_tr.data.numpy().astype(np.float32) / 255.0)   # (60000,28,28)
_TRY = _tr.targets.numpy()
_TEX = (_te.data.numpy().astype(np.float32) / 255.0)
_TEY = _te.targets.numpy()

def _ds14(x):  # 28x28 -> 14x14 avg-pool (recognized CMNIST downsample; also makes shape harder)
    return x.reshape(x.shape[0], 14, 2, 14, 2).mean((2, 4))
_TRX = _ds14(_TRX); _TEX = _ds14(_TEX)

# ---- equal-cost knobs (TUNED by split_scan*.py; see EQUAL_COST.md) ----
# What the scans taught us (the same lesson as the Colored-MNIST failure, made precise):
#   * If the patch is more PREDICTIVE or EASIER than shape, ALL seeds take it (CMNIST: 57/60).
#   * The split only appears in the genuine-minima regime where BOTH cues are PERFECT-train
#     solutions: LABEL_FLIP=0 + P_CORR_TRAIN=1.0  =>  train acc ~= 1.0 for shape-only AND
#     patch-only, i.e. two distinct zero-loss solutions; the seed picks the basin.
#   * With label noise (>0) the net greedily BLENDS both cues to beat each one's ceiling ->
#     a single mid-OOD basin (no split), exactly like the harder_b HANS prototype.
#   * The remaining lever is patch DIFFICULTY (contrast amp + noise). A fine amp sweep
#     (split_scan6) located the equal-difficulty crossover at amp=0.055 (pn=0.08, sn=0.20):
#     there 12 seeds divided 4 shape / 2 patch / 6 mid at train acc 0.999. amp<=0.050 -> all
#     shape; amp>=0.08 -> all patch.
LABEL_FLIP    = 0.0    # NO label noise -> shape & patch can each reach ~1.0 train (two solutions)
P_CORR_TRAIN  = 1.0    # patch PERFECTLY predicts the train label (a perfect-train solution)
P_CORR_TEST   = 0.0    # patch FULLY FLIPPED at OOD test (patch-relier -> ~0 OOD; shape -> shape acc)
PATCH_AMP     = 0.055  # equal-difficulty crossover contrast (from split_scan6)
PATCH_NOISE   = 0.08   # gaussian noise on the patch pixels (handicaps the patch toward shape)
SHAPE_NOISE   = 0.20   # gaussian noise on the garment (keeps shape a nontrivial conv feature)
PATCH_SZ      = 3      # 3x3 corner patch
N             = 1000   # train set per seed

def _apply(imgs, digits, seed, p_corr, f_counter=0.0,
           amp=None, pnoise=None, snoise=None, shape_on=True, patch_on=True):
    """Build (N,1,14,14) images. label = 1[class>=5] w/ LABEL_FLIP noise.
    patch = a PATCH_SZ x PATCH_SZ bright/dark square in the top-left corner whose
    sign encodes a bit that agrees with the label w.p. p_corr (decorrelated in
    fraction f_counter -> counter-spurious intervention data)."""
    amp = PATCH_AMP if amp is None else amp
    pnoise = PATCH_NOISE if pnoise is None else pnoise
    snoise = SHAPE_NOISE if snoise is None else snoise
    rng = np.random.default_rng(seed)
    y = (digits >= 5).astype(np.int64)
    flip = rng.random(len(y)) < LABEL_FLIP
    y = np.where(flip, 1 - y, y)
    agree = rng.random(len(y)) < p_corr
    bit = np.where(agree, y, 1 - y)            # patch bit
    if f_counter > 0:
        cm = rng.random(len(y)) < f_counter
        bit = np.where(cm, rng.integers(0, 2, len(y)), bit)
    Nn = len(y)
    out = np.zeros((Nn, 1, 14, 14), np.float32)
    if shape_on:
        out[:, 0] = imgs + snoise * rng.standard_normal((Nn, 14, 14)).astype(np.float32)
    if patch_on:
        # bit=1 -> bright patch (+amp), bit=0 -> dark patch (-amp), + noise, on a 0.5 pedestal
        ped = 0.5
        sign = np.where(bit == 1, amp, -amp).astype(np.float32)
        patch = ped + sign[:, None, None] + pnoise * rng.standard_normal((Nn, PATCH_SZ, PATCH_SZ)).astype(np.float32)
        out[:, 0, 0:PATCH_SZ, 0:PATCH_SZ] = np.clip(patch, 0.0, 1.0)
    return torch.tensor(out), torch.tensor(y)

def gen(n, seed, mode, f=0.0, **kw):
    if mode == "train":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TRX), size=n, replace=False)
        return _apply(_TRX[idx], _TRY[idx], seed, P_CORR_TRAIN, f_counter=f, **kw)
    else:  # deploy / OOD: patch flipped, held-out test garments
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TEX), size=min(n, len(_TEX)), replace=False)
        return _apply(_TEX[idx], _TEY[idx], seed, P_CORR_TEST, **kw)

# ---------------- model: tiny CNN, narrow bottleneck (same spirit as cmnist) ----------------
# A NARROW bottleneck forces the rep to (largely) commit to ONE cue -> the structural
# condition for genuine BIMODALITY (mirrors alignment.py's competing-XOR + bottleneck).
# With BOTTLE=2 the net can carry both cues and blend them (single mid-OOD basin, like
# harder_b); BOTTLE=1 forces a near-binary commitment per seed.
BOTTLE = int(os.environ.get("BOTTLE", "2"))
WD = float(os.environ.get("WD", "1e-3"))
LR = 3e-3
EPOCHS = 400

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 3, padding=1)
        self.c2 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(12 * 3 * 3, BOTTLE)   # 14->7->3 ; bottleneck
        self.fc2 = nn.Linear(BOTTLE, 2)
        self.relu = nn.ReLU()
    def feat(self, x):
        x = self.pool(self.relu(self.c1(x)))   # 14->7
        x = self.pool(self.relu(self.c2(x)))   # 7->3
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

def train(seed, steps=EPOCHS, f=0.0, gen_kw=None):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    Xtr, ytr = gen(N, seed, "train", f=f, **(gen_kw or {}))
    m = mk(); lf = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), LR, weight_decay=WD)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xtr), ytr).backward(); opt.step()
    return m, Xtr, ytr

# ---------------- genuine-minima diagnostics ----------------
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

# ---------------- mechanism probe: does the model use shape or patch? ----------------
def patch_reliance(m, seed=123, n=2000):
    """Present each held-out garment under patch-bit=0 and patch-bit=1 (shape fixed);
    fraction of predictions that FLIP when only the patch changes (0=pure shape, 1=pure patch)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(_TEX), size=n, replace=False)
    imgs = _TEX[idx]
    def build(bit):
        out = np.zeros((n, 1, 14, 14), np.float32)
        out[:, 0] = imgs  # no shape noise at probe time -> isolate the patch effect
        ped = 0.5; sign = (PATCH_AMP if bit == 1 else -PATCH_AMP)
        out[:, 0, 0:PATCH_SZ, 0:PATCH_SZ] = np.clip(ped + sign, 0, 1)
        return torch.tensor(out)
    with torch.no_grad():
        p0 = m(build(0)).argmax(1); p1 = m(build(1)).argmax(1)
    return (p0 != p1).float().mean().item()

# ---------------- connectivity (permutation-aligned barrier on bottleneck) ----------------
from scipy.optimize import linear_sum_assignment
def acts(m, X):
    with torch.no_grad(): return m.feat(X).numpy()
def align2(mA, mB, Xref):
    A = acts(mA, Xref); B = acts(mB, Xref)
    An = (A - A.mean(0)) / (A.std(0) + 1e-6); Bn = (B - B.mean(0)) / (B.std(0) + 1e-6)
    _, c = linear_sum_assignment(-(An.T @ Bn) / A.shape[0])
    s = mB.state_dict(); o = {k: v.clone() for k, v in s.items()}
    o["fc1.weight"] = s["fc1.weight"][c]; o["fc1.bias"] = s["fc1.bias"][c]
    o["fc2.weight"] = s["fc2.weight"][:, c]
    mC = mk(); mC.load_state_dict(o); return mC
def barrier(mA, mB, X, y, npts=11):
    lf = nn.CrossEntropyLoss(); sA, sB = mA.state_dict(), mB.state_dict(); m = mk(); ls = []
    for a in np.linspace(0, 1, npts):
        m.load_state_dict({k: (1 - a) * sA[k] + a * sB[k] for k in sA})
        with torch.no_grad(): ls.append(lf(m(X), y).item())
    ls = np.array(ls); return float(ls.max() - max(ls[0], ls[-1]))
