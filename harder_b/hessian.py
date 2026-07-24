import torch
import torch.nn as nn


def _loss_for_hessian(model, Xt, yt, wd):
    loss = nn.CrossEntropyLoss()(model(Xt), yt)
    reg = sum((p * p).sum() for p in model.parameters())
    return loss + 0.5 * wd * reg


def hvp(model, Xt, yt, wd, vec):
    """Hessian-vector product via double autograd. vec is a flat tensor matching params."""
    params = [p for p in model.parameters()]
    loss = _loss_for_hessian(model, Xt, yt, wd)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_g = torch.cat([g.reshape(-1) for g in grads])
    gv = (flat_g * vec).sum()
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hv]).detach()


def power_iterate_top(model, Xt, yt, wd, n_iter=80, tol=1e-6, seed=0):
    """Top eigenvalue of Hessian via power iteration."""
    g = torch.Generator().manual_seed(seed)
    dim = sum(p.numel() for p in model.parameters())
    v = torch.randn(dim, generator=g)
    v = v / v.norm()
    lam = 0.0
    for _ in range(n_iter):
        hv = hvp(model, Xt, yt, wd, v)
        new_lam = torch.dot(v, hv).item()
        nv = hv.norm()
        if nv < 1e-12:
            return new_lam, v
        v = hv / nv
        if abs(new_lam - lam) < tol * (1 + abs(new_lam)):
            lam = new_lam
            break
        lam = new_lam
    return lam, v


def min_eigenvalue(model, X, y, wd, n_iter=80, device="cpu", seed=0):
    """
    Minimum eigenvalue of the (penalized-loss) Hessian.
    Strategy: get top eig L_top. Then power-iterate on shifted operator (L_top*I - H);
    its top eigenvalue is L_top - lambda_min, so lambda_min = L_top - top(L_top*I - H).
    """
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    L_top, _ = power_iterate_top(model, Xt, yt, wd, n_iter=n_iter, seed=seed)
    shift = abs(L_top) + 1.0  # ensure shifted operator is PSD-ish so its top eig is well-defined
    # Operator B = shift*I - H. top eig of B = shift - lambda_min(H)
    g = torch.Generator().manual_seed(seed + 1)
    dim = sum(p.numel() for p in model.parameters())
    v = torch.randn(dim, generator=g)
    v = v / v.norm()
    mu = 0.0
    for _ in range(n_iter):
        hv = hvp(model, Xt, yt, wd, v)
        bv = shift * v - hv
        new_mu = torch.dot(v, bv).item()
        nv = bv.norm()
        if nv < 1e-12:
            break
        v = bv / nv
        if abs(new_mu - mu) < 1e-6 * (1 + abs(new_mu)):
            mu = new_mu
            break
        mu = new_mu
    lambda_min = shift - mu
    return lambda_min, L_top
