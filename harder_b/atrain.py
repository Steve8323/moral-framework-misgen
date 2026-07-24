import numpy as np
import torch
import torch.nn as nn

from amodel import AttnNet
from tmodel import seq_len


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def to_t(data, device="cpu"):
    toks, seg, pad, y = data[0], data[1], data[2], data[3]
    return (torch.tensor(toks, device=device), torch.tensor(seg, device=device),
            torch.tensor(pad, device=device), torch.tensor(y, device=device))


def train_a(data, seed, V, max_p, max_h, d_model=32, nhead=4, dim_ff=64,
            wd=2e-3, epochs=1500, lr=3e-3, device="cpu", verbose=False):
    set_seed(seed)
    vocab = V + 1
    L = seq_len(max_p, max_h)
    model = AttnNet(vocab, L, d_model=d_model, nhead=nhead, dim_ff=dim_ff).to(device)
    tk, sg, pd, y = to_t(data, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(tk, sg, pd)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        if verbose and ep % 200 == 0:
            with torch.no_grad():
                acc = (out.argmax(1) == y).float().mean().item()
            print(f"  ep{ep} loss {loss.item():.4f} acc {acc:.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def acc_a(model, data, device="cpu"):
    tk, sg, pd, y = to_t(data, device)
    return (model(tk, sg, pd).argmax(1) == y).float().mean().item()


@torch.no_grad()
def ce_a(model, data, device="cpu"):
    tk, sg, pd, y = to_t(data, device)
    return nn.CrossEntropyLoss()(model(tk, sg, pd), y).item()


def gradnorm_a(model, data, wd, device="cpu"):
    model.zero_grad()
    tk, sg, pd, y = to_t(data, device)
    loss = nn.CrossEntropyLoss()(model(tk, sg, pd), y)
    reg = sum((p * p).sum() for p in model.parameters())
    obj = loss + 0.5 * wd * reg
    grads = torch.autograd.grad(obj, list(model.parameters()))
    return torch.cat([g.reshape(-1) for g in grads]).norm().item()


def wnorm_a(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).norm().item()
