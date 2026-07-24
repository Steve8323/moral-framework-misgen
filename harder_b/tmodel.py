"""
Tiny transformer for the (premise, hypothesis) entailment task.

Sequence layout (single stream): [CLS] p_1..p_Lp [SEP] h_1..h_Lh, padded to fixed length.
Token embedding (symbol) + learned positional embedding + segment embedding (premise vs hyp).
1-2 self-attention layers. Classify from the CLS token.

Capacity: attention CAN compute lexical-overlap (match h tokens to any p token, order-free) AND
ordered-subsequence (positional/relative attention). Which one it lands in can depend on init seed
-> the McCoy multiplicity regime.
"""
import numpy as np
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab, max_len, d_model=32, nhead=4, nlayers=2, dim_ff=64, dropout=0.0):
        super().__init__()
        # vocab includes PAD(0) and symbols 1..V; plus CLS, SEP special tokens appended.
        self.CLS = vocab          # id for CLS
        self.SEP = vocab + 1      # id for SEP
        n_tok = vocab + 2
        self.tok = nn.Embedding(n_tok, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.seg = nn.Embedding(3, d_model)  # 0 pad-ish/special, 1 premise, 2 hyp
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 2)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, tokens, seg, pad_mask):
        # tokens: (B,L) long; seg: (B,L) long; pad_mask: (B,L) bool True=pad
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        x = self.tok(tokens) + self.pos(pos_ids) + self.seg(seg)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        cls = x[:, 0, :]  # CLS position
        return self.head(cls)


def build_tokens(P, H, max_p, max_h, vocab):
    """Return token ids, segment ids, pad mask for one example. Layout: CLS P.. SEP H.."""
    CLS = vocab; SEP = vocab + 1
    L = 1 + max_p + 1 + max_h
    toks = np.zeros(L, dtype=np.int64)
    seg = np.zeros(L, dtype=np.int64)
    pad = np.ones(L, dtype=bool)  # True=pad
    toks[0] = CLS; seg[0] = 0; pad[0] = False
    for i in range(max_p):
        if i < len(P):
            toks[1 + i] = P[i]; seg[1 + i] = 1; pad[1 + i] = False
    sep_pos = 1 + max_p
    toks[sep_pos] = SEP; seg[sep_pos] = 0; pad[sep_pos] = False
    off = sep_pos + 1
    for j in range(max_h):
        if j < len(H):
            toks[off + j] = H[j]; seg[off + j] = 2; pad[off + j] = False
    return toks, seg, pad


def seq_len(max_p, max_h):
    return 1 + max_p + 1 + max_h
