"""Linear-mode-connectivity barrier between two MLPs, raw and after activation-matching
permutation alignment (Git Re-Basin style, scipy linear_sum_assignment)."""
import copy
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from model import MLP


def get_linear_layers(model):
    """Return list of nn.Linear in order."""
    return [m for m in model.net if isinstance(m, nn.Linear)]


@torch.no_grad()
def interp_loss(mA, mB, X, y, alphas):
    """Train CE loss along linear interpolation theta = (1-a)A + a*B (param-wise)."""
    Xt = torch.tensor(X); yt = torch.tensor(y)
    pA = [p.clone() for p in mA.parameters()]
    pB = [p.clone() for p in mB.parameters()]
    tmp = copy.deepcopy(mA)
    losses = []
    lossf = nn.CrossEntropyLoss()
    for a in alphas:
        for pt, a0, b0 in zip(tmp.parameters(), pA, pB):
            pt.copy_((1 - a) * a0 + a * b0)
        losses.append(lossf(tmp(Xt), yt).item())
    return np.array(losses)


@torch.no_grad()
def hidden_activations_per_layer(model, X):
    """Return list (per hidden ReLU layer) of activation matrices (N x width)."""
    Xt = torch.tensor(X)
    return [a.numpy() for a in model.activations(Xt)]


def align_B_to_A(mA, mB, X):
    """
    Permute hidden units of mB to match mA by activation correlation, layer by layer.
    For a 2-hidden-layer MLP: layers L0 (in->w), L1 (w->w), L2 (w->2).
    Permuting hidden layer k means: permute rows of W_k and bias_k, and columns of W_{k+1}.
    We compute permutation per hidden layer from activation matching (maximize correlation).
    """
    linsA = get_linear_layers(mA)
    mB2 = copy.deepcopy(mB)
    linsB = get_linear_layers(mB2)
    actsA = hidden_activations_per_layer(mA, X)   # list len = num hidden layers
    n_hidden = len(actsA)

    for k in range(n_hidden):
        # recompute B activations after previous permutations applied
        actsB = hidden_activations_per_layer(mB2, X)
        A = actsA[k]; B = actsB[k]            # N x w
        # cost = -correlation; align columns of B to columns of A
        Az = (A - A.mean(0)) / (A.std(0) + 1e-6)
        Bz = (B - B.mean(0)) / (B.std(0) + 1e-6)
        corr = Az.T @ Bz / A.shape[0]        # w x w
        row, col = linear_sum_assignment(-corr)
        perm = np.empty_like(col); perm[row] = col  # perm[i_in_A] = j_in_B
        # We want to reorder B's units so that B_unit perm-> A_unit. Build index that maps
        # new position i (A's order) to old B unit col[i].
        order = col  # B unit order to take so they align with A's units
        with torch.no_grad():
            Wk = linsB[k].weight.data         # (w, in_k)
            bk = linsB[k].bias.data           # (w,)
            linsB[k].weight.data = Wk[order, :].clone()
            linsB[k].bias.data = bk[order].clone()
            Wk1 = linsB[k + 1].weight.data    # (out_{k+1}, w)
            linsB[k + 1].weight.data = Wk1[:, order].clone()
    return mB2


def barrier_report(mA, mB, X, y, n_alpha=21):
    alphas = np.linspace(0, 1, n_alpha)
    raw = interp_loss(mA, mB, X, y, alphas)
    mB_al = align_B_to_A(mA, mB, X)
    aligned = interp_loss(mA, mB_al, X, y, alphas)
    end = max(raw[0], raw[-1])
    raw_barrier = float(raw.max() - end)
    aligned_barrier = float(aligned.max() - max(aligned[0], aligned[-1]))
    return {
        "alphas": alphas.tolist(),
        "raw_curve": raw.tolist(),
        "aligned_curve": aligned.tolist(),
        "raw_barrier": raw_barrier,
        "aligned_barrier": aligned_barrier,
        "endpoint_loss_A": float(raw[0]),
        "endpoint_loss_B": float(raw[-1]),
    }
