import numpy as np
import torch
import torch.nn as nn

from model import MLP


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def train_one(Xtr, ytr, seed, in_dim, width=128, depth=2, wd=2e-3, epochs=400,
              lr=3e-3, device="cpu", verbose=False):
    """Train one model full-batch (small data) with weight decay. Returns model + history."""
    set_seed(seed)
    model = MLP(in_dim, width=width, depth=depth).to(device)
    Xt = torch.tensor(Xtr, device=device)
    yt = torch.tensor(ytr, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
        if verbose and ep % 100 == 0:
            with torch.no_grad():
                acc = (out.argmax(1) == yt).float().mean().item()
            hist.append((ep, loss.item(), acc))
    model.eval()
    return model, hist


@torch.no_grad()
def accuracy(model, X, y, device="cpu"):
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    pred = model(Xt).argmax(1)
    return (pred == yt).float().mean().item()


@torch.no_grad()
def ce_loss(model, X, y, device="cpu"):
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    return nn.CrossEntropyLoss()(model(Xt), yt).item()


def full_batch_grad_norm(model, X, y, wd=2e-3, device="cpu"):
    """L2 norm of the FULL training objective gradient (CE + wd*0.5*||w||^2 == Adam's decoupled wd
    approximated as L2 reg term for the purpose of 'is this a stationary point of the loss')."""
    model.zero_grad()
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    loss = nn.CrossEntropyLoss()(model(Xt), yt)
    # add L2 penalty matching weight decay so the reported gradient is of the penalized objective
    reg = 0.0
    for p in model.parameters():
        reg = reg + (p * p).sum()
    obj = loss + 0.5 * wd * reg
    grads = torch.autograd.grad(obj, list(model.parameters()), create_graph=False)
    g = torch.cat([gg.reshape(-1) for gg in grads])
    return g.norm().item()


def weight_norm(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).norm().item()
