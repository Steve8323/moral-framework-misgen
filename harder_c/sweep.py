"""Sweep configs to find a BIMODAL seed-driven consequentialism-index regime."""
import json, itertools, time
import numpy as np
from moral_misgen import (make_dataset, train_model, train_loss_acc,
                          consequentialism_index, full_batch_grad_norm, weight_norm)

def bimodality(ci):
    ci = np.asarray(ci)
    near0 = int((ci < 0.2).sum())
    near1 = int((ci > 0.8).sum())
    mid = int(((ci >= 0.2) & (ci <= 0.8)).sum())
    # simple bimodality coefficient (Sarle): (skew^2+1)/kurt
    n = len(ci)
    m = ci.mean(); s = ci.std()+1e-9
    sk = ((ci-m)**3).mean()/s**3
    ku = ((ci-m)**4).mean()/s**4
    bc = (sk**2 + 1) / (ku + 3*(n-1)**2/((n-2)*(n-3))) if n > 3 else 0
    return near0, mid, near1, float(bc)

def run_cfg(cfg, data, n_seeds=30):
    Xtr, ytr, Xco, cco = data
    cis, losses, accs, gns = [], [], [], []
    for s in range(n_seeds):
        m = train_model(s, cfg, Xtr, ytr)
        l, a = train_loss_acc(m, Xtr, ytr)
        ci = consequentialism_index(m, Xco, cco)
        cis.append(ci); losses.append(l); accs.append(a)
    cis = np.array(cis)
    n0, mid, n1, bc = bimodality(cis)
    return dict(ci_mean=float(cis.mean()), ci_std=float(cis.std()),
                ci_min=float(cis.min()), ci_max=float(cis.max()),
                near0=n0, mid=mid, near1=n1, bc=bc,
                loss_mean=float(np.mean(losses)), acc_mean=float(np.mean(accs)),
                acc_min=float(np.min(accs)))

def build_data(K, sep, noise, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, ctr, dtr = make_dataset(n, K, sep, noise, agreement=True, rng=rng)
    assert (ctr == dtr).all()
    rng2 = np.random.default_rng(seed+777)
    Xco, cco, dco = make_dataset(n, K, sep, noise, agreement=False, rng=rng2)
    assert (cco != dco).all()
    return Xtr, ctr, Xco, cco

if __name__ == "__main__":
    results = []
    t0 = time.time()
    # Cleaner data so a single block fully solves -> no accuracy incentive to use both.
    for noise in [0.3, 0.5, 0.8]:
        for sep in [2.0, 3.0]:
            for hidden in [1, 2, 3, 4, 8]:
                for init_scale in [0.1, 0.3]:
                    for wd in [1e-3, 5e-3, 2e-2]:
                        cfg = dict(hidden=hidden, init_scale=init_scale, opt="sgd",
                                   lr=0.1, wd=wd, momentum=0.0,
                                   batch_size=64, epochs=300, K=4, sep=sep, noise=noise)
                        data = build_data(4, sep, noise)
                        r = run_cfg(cfg, data, n_seeds=24)
                        r.update(cfg)
                        results.append(r)
                        tag = "***" if (r["near0"]>=3 and r["near1"]>=3) else ""
                        print(f"h={hidden} init={init_scale} wd={wd} sep={sep} noise={noise} "
                              f"| CI {r['ci_mean']:.2f}±{r['ci_std']:.2f} "
                              f"[{r['near0']}/{r['mid']}/{r['near1']}] acc={r['acc_mean']:.3f} {tag}")
    results.sort(key=lambda r: -(min(r["near0"],r["near1"])))
    with open("sweep_results.json","w") as f:
        json.dump(results, f, indent=2)
    print(f"\nElapsed {time.time()-t0:.0f}s. Top bimodal configs:")
    for r in results[:8]:
        print(f"  h={r['hidden']} init={r['init_scale']} wd={r['wd']} sep={r['sep']} "
              f"noise={r['noise']} | [{r['near0']}/{r['mid']}/{r['near1']}] "
              f"CI={r['ci_mean']:.2f}±{r['ci_std']:.2f} acc={r['acc_mean']:.3f}")
