import numpy as np
import torch
import torch.nn as nn

from tmodel import TinyTransformer, seq_len


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def make_tensors(data, device="cpu"):
    toks, seg, pad, y, ht = data
    return (torch.tensor(toks, device=device), torch.tensor(seg, device=device),
            torch.tensor(pad, device=device), torch.tensor(y, device=device))


def train_t(data, seed, V, max_p, max_h, d_model=32, nhead=4, nlayers=2, dim_ff=64,
            wd=2e-3, epochs=600, lr=2e-3, device="cpu", batch=None, verbose=False):
    set_seed(seed)
    vocab = V + 1
    L = seq_len(max_p, max_h)
    model = TinyTransformer(vocab, L, d_model=d_model, nhead=nhead, nlayers=nlayers,
                            dim_ff=dim_ff, dropout=0.0).to(device)
    toks, seg, pad, y = make_tensors(data, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    N = toks.shape[0]
    for ep in range(epochs):
        model.train()
        if batch is None or batch >= N:
            opt.zero_grad()
            out = model(toks, seg, pad)
            loss = lossf(out, y)
            loss.backward(); opt.step()
        else:
            perm = torch.randperm(N)
            for i in range(0, N, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                out = model(toks[idx], seg[idx], pad[idx])
                loss = lossf(out, y[idx])
                loss.backward(); opt.step()
        if verbose and ep % 100 == 0:
            with torch.no_grad():
                acc = (model(toks, seg, pad).argmax(1) == y).float().mean().item()
            print(f"  ep{ep} loss {loss.item():.4f} acc {acc:.3f}")
    model.eval()
    return model


@torch.no_grad()
def accuracy_t(model, data, device="cpu"):
    toks, seg, pad, y = make_tensors(data, device)
    return (model(toks, seg, pad).argmax(1) == y).float().mean().item()


@torch.no_grad()
def ce_loss_t(model, data, device="cpu"):
    toks, seg, pad, y = make_tensors(data, device)
    return nn.CrossEntropyLoss()(model(toks, seg, pad), y).item()


def grad_norm_t(model, data, wd, device="cpu"):
    model.zero_grad()
    toks, seg, pad, y = make_tensors(data, device)
    loss = nn.CrossEntropyLoss()(model(toks, seg, pad), y)
    reg = sum((p * p).sum() for p in model.parameters())
    obj = loss + 0.5 * wd * reg
    grads = torch.autograd.grad(obj, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in grads]).norm().item()


def weight_norm_t(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).norm().item()
