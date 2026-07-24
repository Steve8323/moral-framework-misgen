"""
Design v2: make the LINEAR cue comparably hard to the XOR cue, so neither is
trivially preferred by simplicity bias -> seed can tip the basin.

Linear rule: y_lin = majority sign over k_lin weak A-dims (each weak: small margin
  + gaussian noise). Reading it reliably needs INTEGRATING several dims (effort
  comparable to XOR's 2-way interaction).
XOR rule: clean parity of sign(B1),sign(B2) with healthy margin.

Train: keep samples where y_lin==y_xor, label=that.
OOD: keep y_lin!=y_xor, label:=y_xor  (ood acc 1=XOR, 0=linear).

Scan knobs:
  k_lin        : number of A-dims in the majority vote (more = harder linear)
  a_margin     : margin of each A dim (smaller=harder)
  b_margin     : margin of B dims (larger=cleaner XOR)
Look for OOD std large + bimodal (some <0.3, some >0.7).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
N_NOISE = 3; B_DIM = 2

def build_dims(k_lin):
    return k_lin + B_DIM + N_NOISE

def make_data(n, seed, regime, k_lin, a_margin, b_margin, in_dim,
              noise_scale=1.0):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 800:
        tries += 1; m = target * 10
        A = rng.randn(m, k_lin).astype(np.float32)
        B = rng.randn(m, B_DIM).astype(np.float32)
        for j in range(k_lin):
            A[:, j] = np.sign(A[:, j]) * (np.abs(A[:, j]) + a_margin)
        B[:, 0] = np.sign(B[:, 0]) * (np.abs(B[:, 0]) + b_margin)
        B[:, 1] = np.sign(B[:, 1]) * (np.abs(B[:, 1]) + b_margin)
        # linear rule = majority sign over the k_lin dims (odd k_lin -> no ties)
        votes = (A > 0).sum(1)
        y_lin = (votes > (k_lin / 2.0)).astype(np.int64)
        y_xor = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
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
    def __init__(self, in_dim, width, depth=2):
        super().__init__()
        layers = []; d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]; d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def acc(model, X, Y):
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()

def train(seed, Xtr, Ytr, in_dim, width, depth, wd, epochs, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(in_dim, width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); crit(model(Xtr), Ytr).backward(); opt.step()
    return model

def trial(k_lin, a_margin, b_margin, width, depth, wd, epochs, n_seeds=24):
    in_dim = build_dims(k_lin)
    Xtr, Ytr = make_data(600, 12345, "train", k_lin, a_margin, b_margin, in_dim)
    Xood, Yood = make_data(2000, 999, "ood", k_lin, a_margin, b_margin, in_dim)
    oods, trs = [], []
    for s in range(n_seeds):
        m = train(s, Xtr, Ytr, in_dim, width, depth, wd, epochs)
        trs.append(acc(m, Xtr, Ytr)); oods.append(acc(m, Xood, Yood))
    oods = np.array(oods); trs = np.array(trs)
    return dict(ood_mean=float(oods.mean()), ood_std=float(oods.std()),
                ood_min=float(oods.min()), ood_max=float(oods.max()),
                tr_min=float(trs.min()), tr_mean=float(trs.mean()),
                fracXOR=float((oods > 0.7).mean()), fracLIN=float((oods < 0.3).mean()),
                fracMID=float(((oods >= 0.3) & (oods <= 0.7)).mean()))

if __name__ == "__main__":
    out = []
    grid = []
    for k_lin in [3, 5, 7]:
        for a_margin in [0.0, 0.15]:
            for b_margin in [0.6, 1.2]:
                grid.append(dict(k_lin=k_lin, a_margin=a_margin, b_margin=b_margin,
                                 width=64, depth=2, wd=2e-3, epochs=2500))
    for cfg in grid:
        r = trial(**cfg, n_seeds=24)
        out.append({**cfg, **r})
        print(f"k={cfg['k_lin']} aM={cfg['a_margin']:.2f} bM={cfg['b_margin']:.1f} "
              f"| tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
              f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
              f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
              f"XOR={r['fracXOR']:.2f} LIN={r['fracLIN']:.2f} MID={r['fracMID']:.2f}",
              flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/scan2.json","w") as f:
        json.dump(out, f, indent=2)
    print("SCAN2_DONE", flush=True)
