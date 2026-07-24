"""
Three model families for the same aligned-vs-proxy task, each with a CAPACITY
BOTTLENECK so the equal-cost cue competition produces seed multiplicity.

Each family provides:
  make(...)               -> nn.Module
  hidden_acts(model, X)   -> list of [n, units] activations at permutable layers
  permute_align(mA, mB, Xref) -> state_dict of B permuted to match A's neuron order
The bottleneck is the hidden width / channel / head-dim, kept small.
"""
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from common import INPUT, to_image, to_seq

# ============================== MLP ==============================
class MLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.depth = depth
        layers = []
        d = INPUT
        self.linears = nn.ModuleList()
        for _ in range(depth):
            lin = nn.Linear(d, width); self.linears.append(lin); d = width
        self.head = nn.Linear(d, 2)
    def forward(self, x):
        for lin in self.linears:
            x = torch.relu(lin(x))
        return self.head(x)

def mlp_make(width, depth):
    return lambda: MLP(width, depth)

def mlp_hidden(model, X):
    acts = []
    with torch.no_grad():
        x = X
        for lin in model.linears:
            x = torch.relu(lin(x)); acts.append(x.numpy())
    return acts

def _match(A, B):
    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    B = (B - B.mean(0)) / (B.std(0) + 1e-6)
    _, c = linear_sum_assignment(-(A.T @ B) / A.shape[0])
    return c

def mlp_permute_align(mA, mB, Xref):
    """Permute hidden neurons of every MLP layer to match A, fixing up the chained weights."""
    aA = mlp_hidden(mA, Xref); aB = mlp_hidden(mB, Xref)
    perms = [_match(aA[i], aB[i]) for i in range(len(aA))]
    sB = mB.state_dict(); o = {k: v.clone() for k, v in sB.items()}
    prev = None
    for i, c in enumerate(perms):
        w = sB[f"linears.{i}.weight"]; b = sB[f"linears.{i}.bias"]
        w = w[c]
        if prev is not None:
            w = w[:, prev]
        o[f"linears.{i}.weight"] = w
        o[f"linears.{i}.bias"] = b[c]
        prev = c
    hw = sB["head.weight"]
    if prev is not None:
        hw = hw[:, prev]
    o["head.weight"] = hw
    return o

# ============================== CNN ==============================
class CNN(nn.Module):
    """GROUPED conv on the [n,2,1,3] two-channel cue image: groups=2 gives each cue its
    OWN filter bank (C filters/cue), so the two cues have INDEPENDENT parameters -- per-seed
    init can then break the tie and COMMIT to one cue (a shared kernel instead sums/averages
    the two and hedges). Conv output 2C channels -> tight MLP bottleneck width C -> head.
    The two cues are generated identically and the conv groups are symmetric, so the cues
    stay EQUAL COST."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # in=2 (cue planes), groups=2 -> each group sees 1 input plane, produces C filters
        self.conv = nn.Conv2d(2, 2 * channels, kernel_size=(1, 2), groups=2)  # 1x3 -> 1x2 spatial
        self.convout = 2 * channels * 2   # 2C filters * 2 spatial positions
        self.fc1 = nn.Linear(self.convout, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.head = nn.Linear(channels, 2)
    def forward(self, x):
        if x.dim() == 2:
            x = to_image(x)
        h = torch.relu(self.conv(x))             # [n,2C,1,2]
        f1 = torch.relu(self.fc1(h.flatten(1)))  # [n,C]
        f2 = torch.relu(self.fc2(f1))            # [n,C]
        return self.head(f2)

def cnn_make(channels):
    return lambda: CNN(channels)

def cnn_hidden(model, X):
    """Permutable units: conv out-channels (2C, pooled over spatial), fc1 units, fc2 units.
    Conv out-channels are permuted WITHIN each group (cue) so the group/cue identity is
    preserved -- a cross-group swap is NOT a data symmetry (it would relabel which cue is
    which), so we match group 0 to group 0 and group 1 to group 1 separately."""
    Xi = to_image(X)
    with torch.no_grad():
        hc = torch.relu(model.conv(Xi))            # [n,2C,1,2]
        chan = hc.mean(dim=(2, 3)).numpy()         # [n,2C]
        f1 = torch.relu(model.fc1(hc.flatten(1)))
        f2 = torch.relu(model.fc2(f1)).numpy()
        f1 = f1.numpy()
    return chan, f1, f2

def cnn_permute_align(mA, mB, Xref):
    C = mB.channels
    chanA, f1A, f2A = cnn_hidden(mA, Xref); chanB, f1B, f2B = cnn_hidden(mB, Xref)
    # match conv channels within each group separately (preserve cue identity)
    cc0 = _match(chanA[:, :C], chanB[:, :C])           # group 0 (cue0) -> indices in 0..C-1
    cc1 = _match(chanA[:, C:], chanB[:, C:]) + C        # group 1 (cue1) -> indices in C..2C-1
    cc = np.concatenate([cc0, cc1])                     # full 2C permutation
    c1 = _match(f1A, f1B)
    c2 = _match(f2A, f2B)
    sB = mB.state_dict(); o = {k: v.clone() for k, v in sB.items()}
    o["conv.weight"] = sB["conv.weight"][cc]            # [2C,1,1,2]
    o["conv.bias"] = sB["conv.bias"][cc]
    # fc1 input = conv flattened channel-major [2C * 2spatial]; permute channel blocks (cc), out (c1)
    fcw = sB["fc1.weight"].reshape(C, 2 * C, 2)         # [outF=C, Cin=2C, spatial=2]
    fcw = fcw[c1][:, cc]
    o["fc1.weight"] = fcw.reshape(C, mB.convout)
    o["fc1.bias"] = sB["fc1.bias"][c1]
    o["fc2.weight"] = sB["fc2.weight"][c2][:, c1]
    o["fc2.bias"] = sB["fc2.bias"][c2]
    o["head.weight"] = sB["head.weight"][:, c2]
    return o

# ============================== Transformer ==============================
class TinyTransformer(nn.Module):
    """7 feature-tokens -> linear embed (+learned positional) -> L self-attn blocks
    -> mean-pool -> head. d_model = bottleneck."""
    def __init__(self, d_model=8, nlayers=1, nheads=1, seqlen=INPUT):
        super().__init__()
        self.d_model = d_model; self.nlayers = nlayers; self.seqlen = seqlen
        self.embed = nn.Linear(1, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(seqlen, d_model))
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(d_model, nheads, batch_first=True),
                ln1=nn.LayerNorm(d_model),
                ff=nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)),
                ln2=nn.LayerNorm(d_model),
            )) for _ in range(nlayers)
        ])
        self.head = nn.Linear(d_model, 2)
    def forward(self, x):
        if x.dim() == 2:
            x = to_seq(x)                  # [n,seq,1]
        h = self.embed(x) + self.pos       # [n,seq,d]
        for blk in self.blocks:
            a, _ = blk["attn"](h, h, h)
            h = blk["ln1"](h + a)
            h = blk["ln2"](h + blk["ff"](h))
        return self.head(h.mean(1))        # mean-pool over tokens

def tf_make(d_model=8, nlayers=1, nheads=1):
    return lambda: TinyTransformer(d_model, nlayers, nheads)

def tf_pooled(model, X):
    """Pooled representation feeding the head (residual/d_model basis)."""
    Xs = to_seq(X)
    with torch.no_grad():
        h = model.embed(Xs) + model.pos
        for blk in model.blocks:
            a, _ = blk["attn"](h, h, h)
            h = blk["ln1"](h + a)
            h = blk["ln2"](h + blk["ff"](h))
        return h.mean(1).numpy()

def tf_ff_hidden(model, X, layer):
    """FF hidden activations (ReLU layer) of a given block, mean-pooled over tokens,
    for matching the FF internal basis (an independent permutation symmetry)."""
    Xs = to_seq(X)
    with torch.no_grad():
        h = model.embed(Xs) + model.pos
        for i, blk in enumerate(model.blocks):
            a, _ = blk["attn"](h, h, h)
            h = blk["ln1"](h + a)
            ff_in = h
            ff_hidden = torch.relu(blk["ff"][0](ff_in))   # [n,seq,d]
            h = blk["ln2"](h + blk["ff"][2](ff_hidden))
            if i == layer:
                return ff_hidden.mean(1).numpy()
    return None

def tf_permute_align(mA, mB, Xref):
    """Align the head-input feature basis (d_model dims) of B to A by matching the
    pooled activations, and apply that permutation consistently to embed/pos/blocks/head.
    A pure permutation of the d_model channels is a symmetry of this architecture
    (MultiheadAttention, LayerNorm and the elementwise FF all commute with a shared
    channel permutation of Q/K/V/out/ff and the LayerNorm affine params)."""
    pA = tf_pooled(mA, Xref); pB = tf_pooled(mB, Xref)
    c = _match(pA, pB)                       # d_model permutation
    d = mB.d_model
    sB = mB.state_dict(); o = {k: v.clone() for k, v in sB.items()}
    ct = torch.as_tensor(c, dtype=torch.long)
    # embed out dim
    o["embed.weight"] = sB["embed.weight"][ct]
    o["embed.bias"] = sB["embed.bias"][ct]
    # positional emb (seq, d)
    o["pos"] = sB["pos"][:, ct]
    for i in range(mB.nlayers):
        p = f"blocks.{i}."
        # MultiheadAttention in_proj packs [Wq;Wk;Wv] each [d,d]
        ipw = sB[p + "attn.in_proj_weight"].reshape(3, d, d)
        ipw = ipw[:, ct][:, :, ct]
        o[p + "attn.in_proj_weight"] = ipw.reshape(3 * d, d)
        ipb = sB[p + "attn.in_proj_bias"].reshape(3, d)[:, ct].reshape(3 * d)
        o[p + "attn.in_proj_bias"] = ipb
        o[p + "attn.out_proj.weight"] = sB[p + "attn.out_proj.weight"][ct][:, ct]
        o[p + "attn.out_proj.bias"] = sB[p + "attn.out_proj.bias"][ct]
        o[p + "ln1.weight"] = sB[p + "ln1.weight"][ct]
        o[p + "ln1.bias"] = sB[p + "ln1.bias"][ct]
        # ff: Linear(d,d) ReLU Linear(d,d). Residual-basis dims (ff.0 input, ff.2 output)
        # permute by c; the ff INTERNAL hidden basis gets its OWN match (independent symmetry).
        ffhA = tf_ff_hidden(mA, Xref, i); ffhB = tf_ff_hidden(mB, Xref, i)
        cff = torch.as_tensor(_match(ffhA, ffhB), dtype=torch.long)
        o[p + "ff.0.weight"] = sB[p + "ff.0.weight"][cff][:, ct]
        o[p + "ff.0.bias"] = sB[p + "ff.0.bias"][cff]
        o[p + "ff.2.weight"] = sB[p + "ff.2.weight"][ct][:, cff]
        o[p + "ff.2.bias"] = sB[p + "ff.2.bias"][ct]
        o[p + "ln2.weight"] = sB[p + "ln2.weight"][ct]
        o[p + "ln2.bias"] = sB[p + "ln2.bias"][ct]
    o["head.weight"] = sB["head.weight"][:, ct]
    return o
