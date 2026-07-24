"""
Pivot: mini-batch SGD to break the deterministic "averaging" of full-batch Adam.

Hypothesis: full-batch training finds a UNIQUE optimal cue-mixture (scan3 showed a
smooth deterministic transition, std=0 everywhere). Mini-batch SGD injects
seed-dependent gradient noise; combined with rich-get-richer feature learning this
can make the *discrete* basin choice (linear vs XOR) seed-dependent -> real
multiplicity. Small training sets + small batches sharpen this.

We operate near a BALANCED-SNR point (a_noise tuned so neither cue dominates) and
sweep batch_size, lr, train_size, and look for OOD bimodality (large std + split
into <0.3 and >0.7) rather than a single mixed value.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
N_NOISE = 3; A_DIM = 2; B_DIM = 2
IN_DIM = A_DIM + B_DIM + N_NOISE

def make_data(n, seed, regime, a_noise, a_margin=0.3, b_margin=0.8, noise_scale=1.0):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 800:
        tries += 1; m = target * 10
        A = rng.randn(m, A_DIM).astype(np.float32)
        B = rng.randn(m, B_DIM).astype(np.float32)
        latentA1 = np.sign(A[:, 0]) * (np.abs(A[:, 0]) + a_margin)
        B[:, 0] = np.sign(B[:, 0]) * (np.abs(B[:, 0]) + b_margin)
        B[:, 1] = np.sign(B[:, 1]) * (np.abs(B[:, 1]) + b_margin)
        y_lin = (latentA1 > 0).astype(np.int64)
        y_xor = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
        A[:, 0] = latentA1 + rng.randn(m).astype(np.float32) * a_noise
        if regime == "train":
            keep = (y_lin == y_xor); lab = y_lin[keep]
        else:
            keep = (y_lin != y_xor); lab = y_xor[keep]
        Ak, Bk = A[keep], B[keep]
        noise = rng.randn(Ak.shape[0], N_NOISE).astype(np.float32) * noise_scale
        feat = np.concatenate([Ak, Bk, noise], axis=1)
        for i in range(Ak.shape[0]):
            if len(xs) >= target: break
            xs.append(feat[i]); ys.append(lab[i])
    return (torch.tensor(np.array(xs, np.float32)),
            torch.tensor(np.array(ys, np.int64)))

class MLP(nn.Module):
    def __init__(self, width, depth=2):
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

def train_sgd(seed, Xtr, Ytr, width, depth, wd, epochs, lr, batch, opt_name="sgd",
              momentum=0.9):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(width, depth)
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                              momentum=momentum)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    N = Xtr.shape[0]
    g = torch.Generator(); g.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            opt.zero_grad(); crit(model(Xtr[idx]), Ytr[idx]).backward(); opt.step()
    return model

def trial(a_noise, batch, lr, train_size, width, depth, wd, epochs,
          opt_name, n_seeds=30):
    Xtr, Ytr = make_data(train_size, 12345, "train", a_noise)
    Xood, Yood = make_data(2000, 999, "ood", a_noise)
    oods, trs = [], []
    for s in range(n_seeds):
        m = train_sgd(s, Xtr, Ytr, width, depth, wd, epochs, lr, batch, opt_name)
        trs.append(acc(m, Xtr, Ytr)); oods.append(acc(m, Xood, Yood))
    oods = np.array(oods); trs = np.array(trs)
    return dict(ood_mean=float(oods.mean()), ood_std=float(oods.std()),
                ood_min=float(oods.min()), ood_max=float(oods.max()),
                tr_min=float(trs.min()), tr_mean=float(trs.mean()),
                fracXOR=float((oods > 0.7).mean()), fracLIN=float((oods < 0.3).mean()),
                fracMID=float(((oods >= 0.3) & (oods <= 0.7)).mean()),
                oods=oods.tolist())

if __name__ == "__main__":
    out = []
    grid = []
    # near-balanced SNR region a_noise in {0.25,0.3,0.35}; small batches; SGD
    for a_noise in [0.25, 0.30, 0.35]:
        for batch in [8, 16, 32]:
            grid.append(dict(a_noise=a_noise, batch=batch, lr=0.05,
                             train_size=300, width=64, depth=2, wd=1e-3,
                             epochs=400, opt_name="sgd"))
    for cfg in grid:
        r = trial(**cfg, n_seeds=30)
        out.append({**cfg, **{k:v for k,v in r.items() if k!='oods'}, 'oods':r['oods']})
        print(f"aN={cfg['a_noise']:.2f} bs={cfg['batch']} | "
              f"tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
              f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
              f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
              f"XOR={r['fracXOR']:.2f} LIN={r['fracLIN']:.2f} MID={r['fracMID']:.2f}",
              flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/scan_sgd.json","w") as f:
        json.dump(out, f, indent=2)
    print("SCANSGD_DONE", flush=True)
