"""
TRUE competition: linear and XOR mechanisms share the SAME features so the net
must choose (committing capacity to one degrades the other). Previous designs put
the two cues on disjoint feature blocks -> net cheaply superimposes both -> unique
graded mixture, no Rashomon set.

Construction on shared (x1, x2):
  bit1 = (x1>0), bit2 = (x2>0).
  LINEAR rule : y_lin = bit1                 (halfspace; ignores x2)
  XOR rule    : y_xor = bit1 XOR bit2        (parity; uses both)
  These agree exactly on the half-plane bit2==0 (x2<0): there y_xor=bit1=y_lin.
  TRAIN: draw only x2<0 samples (a "slab"); label = bit1 (= both rules).
  OOD  : draw only x2>0 samples; there y_lin=bit1 but y_xor = NOT bit1.
         OOD label := y_xor. So OOD acc 1=>XOR-extrapolator, 0=>linear-extrapolator.

Now linear and XOR are GENUINELY exclusive: a model that uses x2 (XOR) flips its
prediction across x2=0; one that ignores x2 (linear) does not. The net cannot do
both for the same x1. Margins keep each rule clean. We add noise dims + the old
disjoint XOR-on-B is dropped.

Sweep capacity / wd / epochs and look for bimodal OOD across seeds.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
N_NOISE = 4
IN_DIM = 2 + N_NOISE   # x1, x2, + noise

def make_data(n, seed, regime, margin=0.5, noise_scale=1.0):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n).astype(np.float32)
    x1 = np.sign(x1) * (np.abs(x1) + margin)
    if regime == "train":
        # x2 < 0  (the slab where rules agree)
        x2 = -(np.abs(rng.randn(n).astype(np.float32)) + margin)
    else:
        # x2 > 0  (rules disagree)
        x2 = (np.abs(rng.randn(n).astype(np.float32)) + margin)
    bit1 = (x1 > 0).astype(np.int64)
    bit2 = (x2 > 0).astype(np.int64)
    y_lin = bit1
    y_xor = bit1 ^ bit2
    lab = y_lin if regime == "train" else y_xor
    noise = rng.randn(n, N_NOISE).astype(np.float32) * noise_scale
    feat = np.concatenate([x1[:, None], x2[:, None], noise], axis=1)
    return torch.tensor(feat), torch.tensor(lab)

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
                fracXOR=float((oods > 0.7).mean()), fracLIN=float((oods < 0.3).mean()),
                fracMID=float(((oods >= 0.3) & (oods <= 0.7)).mean()),
                oods=oods.tolist())

if __name__ == "__main__":
    out = []
    grid = []
    for width in [16, 32, 64, 128]:
        for wd in [1e-3, 3e-3]:
            grid.append(dict(width=width, depth=2, wd=wd, epochs=3000, margin=0.5))
    for cfg in grid:
        r = trial(**cfg, n_seeds=30)
        out.append({**cfg, **{k:v for k,v in r.items() if k!='oods'}, 'oods':r['oods']})
        print(f"w={cfg['width']} wd={cfg['wd']:.0e} | "
              f"tr={r['tr_mean']:.3f}(min{r['tr_min']:.3f}) "
              f"OOD m={r['ood_mean']:.2f} sd={r['ood_std']:.2f} "
              f"[{r['ood_min']:.2f},{r['ood_max']:.2f}] "
              f"XOR={r['fracXOR']:.2f} LIN={r['fracLIN']:.2f} MID={r['fracMID']:.2f}",
              flush=True)
    with open("/Users/steven/moral_framework_misgen/harder_a/scan_overlap.json","w") as f:
        json.dump(out, f, indent=2)
    print("SCANOVL_DONE", flush=True)
