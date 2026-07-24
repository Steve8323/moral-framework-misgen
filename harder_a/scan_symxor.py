"""
Remove the simplicity-bias asymmetry: TWO equally-hard XOR cues on disjoint blocks.

Block B (b1,b2): rule_B = parity(sign b1, sign b2)
Block C (c1,c2): rule_C = parity(sign c1, sign c2)
TRAIN: keep samples where rule_B == rule_C; label = that.
OOD  : keep samples where rule_B != rule_C; label := rule_B.
  -> OOD acc 1 = follows block-B XOR, 0 = follows block-C XOR.
Both mechanisms have IDENTICAL algorithmic complexity, so no built-in preference;
the seed-dependent init may decide which block the net latches onto -> possible
genuine bimodal Rashomon set. + noise dims.

This is the canonical symmetric-competition test. If even THIS is unimodal, the
honest conclusion is that full-batch trained MLPs on this family find a unique
graded mixture regardless of seed.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
N_NOISE = 3
IN_DIM = 4 + N_NOISE   # b1,b2,c1,c2 + noise

def make_data(n, seed, regime, margin=0.6, noise_scale=1.0):
    rng = np.random.RandomState(seed)
    xs, ys = [], []; target = n; tries = 0
    while len(xs) < target and tries < 800:
        tries += 1; m = target * 8
        B = rng.randn(m, 2).astype(np.float32)
        C = rng.randn(m, 2).astype(np.float32)
        for arr in (B, C):
            arr[:, 0] = np.sign(arr[:, 0]) * (np.abs(arr[:, 0]) + margin)
            arr[:, 1] = np.sign(arr[:, 1]) * (np.abs(arr[:, 1]) + margin)
        rB = ((B[:, 0] > 0) ^ (B[:, 1] > 0)).astype(np.int64)
        rC = ((C[:, 0] > 0) ^ (C[:, 1] > 0)).astype(np.int64)
        if regime == "train":
            keep = (rB == rC); lab = rB[keep]
        else:
            keep = (rB != rC); lab = rB[keep]
        Bk, Ck = B[keep], C[keep]
        noise = rng.randn(Bk.shape[0], N_NOISE).astype(np.float32) * noise_scale
        feat = np.concatenate([Bk, Ck, noise], axis=1)
        for i in range(Bk.shape[0]):
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

def trial(width, depth, wd, epochs, margin, n_seeds=30):
    Xtr, Ytr = make_data(700, 12345, "train", margin)
    Xood, Yood = make_data(2000, 999, "ood", margin)
    oods, trs = [], []
    for s in range(n_seeds):
        m = train(s, Xtr, Ytr, width, depth, wd, epochs)
        trs.append(acc(m, Xtr, Ytr)); oods.append(acc(m, Xood, Yood))
    oods = np.array(oods); trs = np.array(trs)
    return dict(ood_mean=float(oods.mean()), ood_std=float(oods.std()),
                ood_min=float(oods.min()), ood_max=float(oods.max()),
                tr_min=float(trs.min()), tr_mean=float(trs.mean()),
                fracB=float((oods > 0.7).mean()), fracC=float((oods < 0.3).mean()),
                fracMID=float(((oods >= 0.3) & (oods <= 0.7)).mean()),
                oods=oods.tolist())

if __name__ == "__main__":
    out = []
    grid = []
    for width in [16, 32, 64]:
        for wd in [1e-3, 3e-3]:
            for margin in [0.6, 1.2]:
                grid.append(dict(width=width, depth=2, wd=wd, epochs=3000, margin=margin))
    for cfg in grid:
        r = trial(**cfg, n_seeds=30)
        out.append({**cfg, **{k:v for k,v in r.items() if k!='oods'}, 'oods':r['oods']})
        print(f"w={cfg['width']} wd={cfg['wd']:.0e} mg={cfg['margin']} | "
              f"tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
              f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
              f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
              f"B={r['fracB']:.2f} C={r['fracC']:.2f} MID={r['fracMID']:.2f}",
              flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/scan_symxor.json","w") as f:
        json.dump(out, f, indent=2)
    print("SCANSYM_DONE", flush=True)
