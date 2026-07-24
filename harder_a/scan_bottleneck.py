"""
Capacity-competition design: a NARROW bottleneck forces the net to COMMIT to one
mechanism. Full-batch training found a unique graded MIXTURE of linear+XOR
(scan3: smooth deterministic transition). If we deny the net enough capacity to
hold both representations, it must pick one -> the pick can be seed-dependent
(genuine discrete multiplicity / Rashomon set).

Architecture: Linear(in->W1) ReLU  Linear(W1->bott) ReLU  Linear(bott->2).
Sweep bott in {2,3,4}, a_noise near balance, with full-batch Adam (so any
multiplicity is NOT just SGD noise but real basin selection).
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

class BottleMLP(nn.Module):
    def __init__(self, w1, bott):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, w1), nn.ReLU(),
            nn.Linear(w1, bott), nn.ReLU(),
            nn.Linear(bott, 2))
    def forward(self, x): return self.net(x)

def acc(model, X, Y):
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()

def train(seed, Xtr, Ytr, w1, bott, wd, epochs, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = BottleMLP(w1, bott)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); crit(model(Xtr), Ytr).backward(); opt.step()
    return model

def trial(a_noise, w1, bott, wd, epochs, n_seeds=30):
    Xtr, Ytr = make_data(700, 12345, "train", a_noise)
    Xood, Yood = make_data(2000, 999, "ood", a_noise)
    oods, trs = [], []
    for s in range(n_seeds):
        m = train(s, Xtr, Ytr, w1, bott, wd, epochs)
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
    for bott in [2, 3, 4]:
        for a_noise in [0.20, 0.30, 0.40]:
            grid.append(dict(a_noise=a_noise, w1=64, bott=bott, wd=1e-3, epochs=2500))
    for cfg in grid:
        r = trial(**cfg, n_seeds=30)
        out.append({**cfg, **{k:v for k,v in r.items() if k!='oods'}, 'oods':r['oods']})
        print(f"bott={cfg['bott']} aN={cfg['a_noise']:.2f} | "
              f"tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
              f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
              f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
              f"XOR={r['fracXOR']:.2f} LIN={r['fracLIN']:.2f} MID={r['fracMID']:.2f}",
              flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/scan_bottleneck.json","w") as f:
        json.dump(out, f, indent=2)
    print("SCANBOTT_DONE", flush=True)
