"""Verify permutation alignments are EXACT symmetries: a model loaded with the
permuted state_dict of B must produce identical outputs to B. If not, the barrier
connectivity result is invalid."""
import torch, numpy as np
from common import gen_features
from models import (MLP, mlp_make, mlp_permute_align,
                    CNN, cnn_make, cnn_permute_align,
                    TinyTransformer, tf_make, tf_permute_align)

torch.manual_seed(0); np.random.seed(0)
X, y = gen_features(64, 12345, "train")

def check(name, makeA, makeB, align):
    mA = makeA(); mB = makeB()
    sd_perm = align(mA, mB, X)
    mC = makeB(); mC.load_state_dict(sd_perm)
    with torch.no_grad():
        oB = mB(X); oC = mC(X)
    diff = (oB - oC).abs().max().item()
    print(f"{name:14s} max|out(B) - out(perm(B))| = {diff:.2e}  {'OK' if diff < 1e-4 else 'FAIL <<<'}")
    return diff

check("MLP w8 d2", mlp_make(8, 2), mlp_make(8, 2), mlp_permute_align)
check("MLP w16 d3", mlp_make(16, 3), mlp_make(16, 3), mlp_permute_align)
check("CNN c8", cnn_make(8), cnn_make(8), cnn_permute_align)
check("TF d8 L1", tf_make(8, 1, 1), tf_make(8, 1, 1), tf_permute_align)
check("TF d8 L2", tf_make(8, 2, 1), tf_make(8, 2, 1), tf_permute_align)
