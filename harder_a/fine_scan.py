"""
Fine 1D scan over a single relative-strength knob to locate the linear<->XOR
transition band where genuine seed multiplicity may live.

Knob: b_scale multiplies the B-block feature magnitude (XOR readability).
We DON'T add noise to A1 (keep linear cue clean & deterministic); instead we
scale the FEATURE magnitudes so the early-training gradient pull is balanced.
Also scan a couple of epoch budgets (early stop can preserve init-driven diversity).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
N_NOISE = 3; A_DIM = 2; B_DIM = 2
IN_DIM = A_DIM + B_DIM + N_NOISE

def make_data(n, seed, regime, a_scale, b_scale, a_margin=0.3, b_margin=0.6,
              noise_scale=1.0):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 400:
        tries += 1; m = target * 8
        A = rng.randn(m, A_DIM).astype(np.float32)
        B = rng.randn(m, B_DIM).astype(np.float32)
        A[:, 0] = np.sign(A[:, 0]) * (np.abs(A[:, 0]) + a_margin)
        B[:, 0] = np.sign(B[:, 0]) * (np.abs(B[:, 0]) + b_margin)
        B[:, 1] = np.sign(B[:, 1]) * (np.abs(B[:, 1]) + b_margin)
        y_lin = (A[:, 0] > 0).astype(np.int64)
        y_xor = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
        A *= a_scale; B *= b_scale
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

def train(seed, Xtr, Ytr, width, depth, wd, epochs, lr=5e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); crit(model(Xtr), Ytr).backward(); opt.step()
    return model

def trial(a_scale, b_scale, width, depth, wd, epochs, n_seeds=24):
    Xtr, Ytr = make_data(600, 12345, "train", a_scale, b_scale)
    Xood, Yood = make_data(2000, 999, "ood", a_scale, b_scale)
    oods, trs = [], []
    for s in range(n_seeds):
        m = train(s, Xtr, Ytr, width, depth, wd, epochs)
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
    # fix a_scale=1, sweep b_scale across the suspected transition (small=linear, large=xor)
    for epochs in [1500, 3000]:
        for b_scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            cfg = dict(a_scale=1.0, b_scale=b_scale, width=64, depth=2,
                       wd=2e-3, epochs=epochs)
            r = trial(**cfg, n_seeds=24)
            rec = {**cfg, **r}; out.append(rec)
            print(f"ep={epochs} bS={b_scale:.2f} | tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
                  f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
                  f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
                  f"XOR={r['fracXOR']:.2f} LIN={r['fracLIN']:.2f} MID={r['fracMID']:.2f}",
                  flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/fine_scan.json","w") as f:
        json.dump(out, f, indent=2)
    print("FINESCAN_DONE", flush=True)
