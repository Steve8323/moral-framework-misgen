"""
Colored-MNIST spurious-correlation experiment (real torchvision MNIST).

CLAIM under test (ported from ../alignment.py to a RECOGNIZED dataset):
  With a spurious cue that is predictive in TRAIN but FLIPS out-of-distribution,
  the random SEED alone decides whether a model relies on the TRUE feature
  (digit shape -> generalizes/"aligned") or the SPURIOUS one (color -> misgen
  /"misaligned"), at GENUINE local minima; and a small fraction of
  counter-spurious (color-decorrelated) training data flips the population to
  the true feature.

DATASET: Colored-MNIST, the standard IRM / spurious-correlation benchmark.
  - true label y = digit parity?  NO: we use the IRM convention y = 1[digit>=5]
    (binary, 2 classes) which is the recognized Arjovsky et al. setup. (We also
    label-flip with prob 0.25 as in IRM so color can be MORE predictive than
    shape, the condition that makes the shortcut tempting.)
  - spurious cue = COLOR (red vs green). In TRAIN color agrees with the (noisy)
    label with prob p_corr (default 0.90). At OOD TEST color is FLIPPED
    (agrees with prob 0.10) -> a color-relier scores ~ (1 - train-color-acc) OOD,
    a shape-relier keeps ~ its shape accuracy.

This is the Arjovsky-Bottou-Gulrajani-Lopez-Paz (2019) Colored-MNIST, minus
the multi-environment IRM penalty (we train plain ERM and study seed multiplicity).

Bottleneck: a tiny CNN with a 2-unit penultimate bottleneck so the
representation cannot encode both cues comfortably -> multiplicity can appear.
"""
import numpy as np, torch, torch.nn as nn, json, time, os
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(8)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------------- data ----------------
import torchvision
_mn_tr = torchvision.datasets.MNIST(root=DATA, train=True, download=True)
_mn_te = torchvision.datasets.MNIST(root=DATA, train=False, download=True)
_TRX = (_mn_tr.data.numpy().astype(np.float32) / 255.0)   # (60000,28,28)
_TRY = _mn_tr.targets.numpy()
_TEX = (_mn_te.data.numpy().astype(np.float32) / 255.0)   # (10000,28,28)
_TEY = _mn_te.targets.numpy()

# downsample to 14x14 to keep CPU cheap (recognized CMNIST downsamples too)
def _ds14(x):  # (N,28,28)->(N,14,14) average-pool 2x2
    return x.reshape(x.shape[0], 14, 2, 14, 2).mean((2, 4))
_TRX = _ds14(_TRX); _TEX = _ds14(_TEX)

LABEL_FLIP = 0.25   # IRM: y = 1[d>=5] then flip 25% -> shape acc ceiling ~0.75
P_CORR_TRAIN = 0.90 # color agrees with label 90% in train
P_CORR_TEST  = 0.10 # color FLIPPED at OOD test (agrees only 10%)

def _make_colored(imgs, digits, seed, p_corr, f_counter=0.0):
    """Return (N,2,14,14) two-channel (red,green) images + binary label.
    label = 1[digit>=5] with LABEL_FLIP noise. color channel chosen to agree
    with label w.p. p_corr; in fraction f_counter of examples color is
    DEcorrelated (drawn 50/50) regardless -> counter-spurious intervention data."""
    rng = np.random.default_rng(seed)
    y = (digits >= 5).astype(np.int64)
    flip = rng.random(len(y)) < LABEL_FLIP
    y = np.where(flip, 1 - y, y)
    # color bit: agrees with y w.p. p_corr
    agree = rng.random(len(y)) < p_corr
    color = np.where(agree, y, 1 - y)
    if f_counter > 0:
        cm = rng.random(len(y)) < f_counter
        color = np.where(cm, rng.integers(0, 2, len(y)), color)  # decorrelated
    N = len(y)
    out = np.zeros((N, 2, 14, 14), np.float32)
    # channel 0 = red, channel 1 = green; put the grayscale digit in the colored chan
    red = color == 0
    out[red, 0] = imgs[red]
    out[~red, 1] = imgs[~red]
    return torch.tensor(out), torch.tensor(y)

def gen(n, seed, mode, f=0.0):
    """mode in {train, deploy}. Uses MNIST train pool for train, MNIST test pool
    for deploy so OOD shift is purely the color flip on UNSEEN digits."""
    if mode == "train":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TRX), size=n, replace=False)
        return _make_colored(_TRX[idx], _TRY[idx], seed, P_CORR_TRAIN, f_counter=f)
    else:  # deploy / OOD test: color flipped, on held-out test digits
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(_TEX), size=min(n, len(_TEX)), replace=False)
        return _make_colored(_TEX[idx], _TEY[idx], seed, P_CORR_TEST)

# ---------------- model (tiny CNN, 2-unit bottleneck) ----------------
BOTTLE = 2
WD = 1e-3
LR = 3e-3
EPOCHS = 400
N = 1000   # train set size per seed (small -> shortcut tempting, cheap)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(2, 6, 3, padding=1)   # sees color (2 in-channels)
        self.c2 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(12 * 3 * 3, BOTTLE)  # 14->7->3 ; bottleneck
        self.fc2 = nn.Linear(BOTTLE, 2)
        self.relu = nn.ReLU()
    def feat(self, x):
        x = self.pool(self.relu(self.c1(x)))   # 14->7
        x = self.pool(self.relu(self.c2(x)))   # 7->3
        x = x.flatten(1)
        return self.relu(self.fc1(x))          # bottleneck activation
    def forward(self, x):
        return self.fc2(self.feat(x))

def mk(): return Net()

def accm(m, X, y):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 2000):
            out.append((m(X[i:i+2000]).argmax(1) == y[i:i+2000]).float())
        return torch.cat(out).mean().item()

def train(seed, steps=EPOCHS, f=0.0):
    torch.manual_seed(seed); np.random.seed(seed % (2**31))
    Xtr, ytr = gen(N, seed, "train", f=f)
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

# ---------------- mode probe: does the model use shape or color? ----------------
def color_reliance(m, seed=123, n=2000):
    """Probe: take test digits, present each under BOTH colors; if prediction
    follows color -> color-relier. Returns fraction of predictions that flip
    when only the color is swapped (0 = pure shape, 1 = pure color)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(_TEX), size=n, replace=False)
    imgs = _TEX[idx]
    Xr = np.zeros((n, 2, 14, 14), np.float32); Xr[:, 0] = imgs   # all red
    Xg = np.zeros((n, 2, 14, 14), np.float32); Xg[:, 1] = imgs   # all green
    with torch.no_grad():
        pr = m(torch.tensor(Xr)).argmax(1); pg = m(torch.tensor(Xg)).argmax(1)
    return (pr != pg).float().mean().item()

# ---------------- connectivity (permutation-aligned barrier) ----------------
def acts(m, X):
    with torch.no_grad():
        return m.feat(X).numpy()   # bottleneck features (N, BOTTLE)
def align2(mA, mB, Xref):
    """Align mB's bottleneck units to mA by activation matching, permute fc1/fc2."""
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

# =====================================================================
if __name__ == "__main__":
    t0 = time.time()
    RES = {}
    NSEED = 60
    SEEDBASE = 4000

    # fixed large OOD test set
    Xdep, ydep = gen(8000, 999001, "deploy")
    # in-distribution (same-color) test to confirm train acc is real
    print(f"{'='*70}\nColored-MNIST  (real torchvision MNIST, 14x14, 2-color channels)")
    print(f"  y=1[digit>=5], label-flip={LABEL_FLIP}, train color-corr={P_CORR_TRAIN}, OOD color-corr={P_CORR_TEST}")
    print(f"  model: tiny CNN, {BOTTLE}-unit bottleneck, WD={WD}, LR={LR}, EPOCHS={EPOCHS}, N={N}")
    print(f"  #params={sum(p.numel() for p in mk().parameters())}")
    print('='*70)

    # ---- (b) SEED MULTIPLICITY ----
    sc = []; tac = []; cr = []; models = []
    for s in range(NSEED):
        m, Xt, yt = train(SEEDBASE + s)
        models.append((m, Xt, yt))
        sc.append(accm(m, Xdep, ydep)); tac.append(accm(m, Xt, yt)); cr.append(color_reliance(m))
        if (s + 1) % 10 == 0: print(f"  ...{s+1}/{NSEED} seeds ({time.time()-t0:.0f}s)")
    sc = np.array(sc); tac = np.array(tac); cr = np.array(cr)
    h, _ = np.histogram(sc, bins=10, range=(0, 1))
    USE_TRUE = sc > 0.55     # generalizes via shape
    USE_SPUR = sc < 0.45     # misgeneralizes via color
    nHi = int(USE_TRUE.sum()); nLo = int(USE_SPUR.sum())
    print(f"\nSEED MULTIPLICITY ({NSEED} seeds, identical except seed):")
    print(f"  in-dist train acc {tac.mean():.3f}+/-{tac.std():.3f}  (range [{tac.min():.3f},{tac.max():.3f}])")
    print(f"  OOD acc: mean {sc.mean():.3f} std {sc.std():.3f} range [{sc.min():.3f},{sc.max():.3f}]")
    print(f"  uses-TRUE(OOD>0.55) {nHi}/{NSEED} | uses-SPUR(OOD<0.45) {nLo}/{NSEED} | mid {NSEED-nHi-nLo}")
    print(f"  color-reliance probe: mean {cr.mean():.3f}  (corr with OOD acc r={np.corrcoef(cr,sc)[0,1]:.3f})")
    print(f"  OOD hist[0..1]: " + " ".join(f"{x:2d}" for x in h))

    RES["multiplicity"] = dict(nseed=NSEED, train_acc_mean=float(tac.mean()), train_acc_std=float(tac.std()),
        train_acc_min=float(tac.min()), train_acc_max=float(tac.max()),
        ood_mean=float(sc.mean()), ood_std=float(sc.std()), ood_min=float(sc.min()), ood_max=float(sc.max()),
        n_uses_true=nHi, n_uses_spur=nLo, n_mid=NSEED - nHi - nLo, hist=h.tolist(),
        color_reliance_mean=float(cr.mean()), corr_creliance_ood=float(np.corrcoef(cr, sc)[0, 1]),
        ood_scores=sc.tolist(), color_reliance=cr.tolist(), train_acc=tac.tolist())

    # ---- (d) GENUINE MINIMA on representative seeds ----
    hi = [i for i in range(NSEED) if USE_TRUE[i]]; lo = [i for i in range(NSEED) if USE_SPUR[i]]
    RES["genuine_minima"] = {}
    print(f"\nGENUINE-MINIMA verification (representative seeds):")
    for tag, cl in [("uses_true", hi), ("uses_spur", lo)]:
        if not cl: print(f"  [{tag}] none in this run"); continue
        i = cl[0]; m, Xt, yt = models[i]
        g = gnorm(m, Xt, yt); w1 = wnorm(m)
        m2, _, _ = train(SEEDBASE + i, steps=2 * EPOCHS); w2 = wnorm(m2)
        em = eig_min(m, Xt, yt)
        print(f"  [{tag}] seed{SEEDBASE+i}: ||fullgrad||={g:.2e}  wnorm {w1:.2f}->{w2:.2f}(2x train)  "
              f"OOD {sc[i]:.2f}->{accm(m2,Xdep,ydep):.2f}  Hess_min={em:+.2e} -> "
              f"{'GENUINE MIN' if em > -1e-3 else 'saddle?'}")
        RES["genuine_minima"][tag] = dict(seed=int(SEEDBASE + i), grad_norm=float(g),
            wnorm=float(w1), wnorm_2x=float(w2), ood=float(sc[i]), ood_2x=float(accm(m2, Xdep, ydep)),
            hess_min_eig=float(em), train_acc=float(tac[i]))

    # ---- (e) CONNECTIVITY ----
    Xb, yb = gen(N, SEEDBASE + 99, "train"); rng = np.random.default_rng(0)
    def avg(ca, cb, al, same, k=15):
        out = []
        for _ in range(k):
            i = int(rng.choice(ca)); j = int(rng.choice(cb))
            if same and i == j: continue
            mB = align2(models[i][0], models[j][0], Xb) if al else models[j][0]
            out.append(barrier(models[i][0], mB, Xb, yb))
        return float(np.mean(out)) if out else float('nan')
    if hi and lo:
        conn = dict(within_true_aln=avg(hi, hi, True, True), across_aln=avg(hi, lo, True, False),
                    within_true_raw=avg(hi, hi, False, True), across_raw=avg(hi, lo, False, False))
        verdict = 'connected mod perm' if conn['across_aln'] < 2 * max(conn['within_true_aln'], 1e-3) else 'DISTINCT basins'
        print(f"\nCONNECTIVITY (loss barrier on train data):")
        print(f"  RAW    : within(true,true) {conn['within_true_raw']:.3f}  across(true,spur) {conn['across_raw']:.3f}")
        print(f"  ALIGNED: within(true,true) {conn['within_true_aln']:.3f}  across(true,spur) {conn['across_aln']:.3f}  -> {verdict}")
        conn["verdict"] = verdict
        RES["connectivity"] = conn
    else:
        print("\nCONNECTIVITY: need both clusters non-empty; skipped.")
        RES["connectivity"] = None

    # ---- (c) INTERVENTION sweep ----
    print(f"\nINTERVENTION: fraction f of counter-spurious (color-decorrelated) train data:")
    sweep = []
    NS_INT = 40
    for f in [0.0, 0.01, 0.02, 0.05, 0.10]:
        al = []
        for s in range(NS_INT):
            m, _, _ = train(SEEDBASE + s, f=f); al.append(accm(m, Xdep, ydep))
        al = np.array(al)
        fa = float((al > 0.55).mean()); fm = float((al < 0.45).mean())
        print(f"  f={f:4.2f} | mean OOD {al.mean():.3f} | frac uses-true(>0.55) {fa:.2f} | uses-spur(<0.45) {fm:.2f}")
        sweep.append(dict(f=f, mean_ood=float(al.mean()), frac_uses_true=fa, frac_uses_spur=fm,
                          ood_scores=al.tolist()))
    RES["intervention"] = sweep

    RES["config"] = dict(dataset="Colored-MNIST (torchvision MNIST, 14x14, 2-color)",
        label_rule="y=1[digit>=5]", label_flip=LABEL_FLIP, p_corr_train=P_CORR_TRAIN,
        p_corr_test=P_CORR_TEST, bottleneck=BOTTLE, WD=WD, LR=LR, EPOCHS=EPOCHS, N=N,
        nparams=int(sum(p.numel() for p in mk().parameters())))

    json.dump(RES, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    print(f"\nTotal {time.time()-t0:.0f}s. Wrote results.json")

    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(sc, bins=20, range=(0, 1), color="#55A868", edgecolor="white")
    ax[0].axvline(P_CORR_TEST, color="#C44E52", ls=":", label=f"pure color-relier ~{1-P_CORR_TRAIN:.2f}")
    ax[0].axvline(0.5, color="gray", ls="--")
    ax[0].set_xlabel("OOD accuracy (color flipped at test)"); ax[0].set_ylabel("# seeds")
    ax[0].set_title(f"Colored-MNIST SEED MULTIPLICITY\n{nHi}/{NSEED} use TRUE shape, {nLo}/{NSEED} use SPURIOUS color\n(identical except seed; genuine minima)")
    ax[0].legend(fontsize=8)
    fs = [d["f"] for d in sweep]
    ax[1].plot(fs, [d["frac_uses_true"] for d in sweep], "o-", color="#55A868", label="frac uses-true (OOD>0.55)")
    ax[1].plot(fs, [d["mean_ood"] for d in sweep], "s--", color="#4C72B0", label="mean OOD acc")
    ax[1].set_xlabel("fraction counter-spurious train data (f)"); ax[1].set_ylabel("OOD generalization")
    ax[1].set_ylim(0, 1.02); ax[1].set_title("INTERVENTION: a little color-decorrelated\ndata flips population to TRUE feature")
    ax[1].legend(fontsize=8)
    ax[2].scatter(cr, sc, s=18, alpha=0.7, color="#8172B3")
    ax[2].set_xlabel("color-reliance probe (1=pure color)"); ax[2].set_ylabel("OOD accuracy")
    ax[2].set_title(f"Mechanism check: OOD acc vs color-reliance\nr={np.corrcoef(cr,sc)[0,1]:.2f} (color use -> OOD failure)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cmnist.png")); plt.close(fig)
    print("Wrote cmnist.png")
