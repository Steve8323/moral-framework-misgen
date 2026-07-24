"""
Lean single-block self-attention classifier (custom, no LayerNorm) for the entailment task.
Designed so BOTH solutions are accessible:
  - lexical-overlap heuristic: attend hyp tokens to matching premise tokens by CONTENT (order-free)
  - true ordered-subsequence: use POSITION embeddings to attend in an order-sensitive way
Fast on CPU (one attention block, small d_model), full-batch trainable.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnNet(nn.Module):
    def __init__(self, vocab, max_len, d_model=24, nhead=4, dim_ff=64):
        super().__init__()
        n_tok = vocab + 2  # + CLS + SEP
        self.tok = nn.Embedding(n_tok, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.seg = nn.Embedding(3, d_model)
        self.nhead = nhead
        self.d_model = d_model
        self.dh = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.head = nn.Linear(d_model, 2)
        self.max_len = max_len

    def embed(self, tokens, seg):
        B, L = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, L)
        return self.tok(tokens) + self.pos(pos_ids) + self.seg(seg)

    def forward(self, tokens, seg, pad_mask):
        B, L = tokens.shape
        x = self.embed(tokens, seg)
        q = self.q(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)  # B,h,L,L
        mask = pad_mask.unsqueeze(1).unsqueeze(1)  # B,1,1,L
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = x + self.o(ctx)                     # residual
        x = x + self.ff2(F.relu(self.ff1(x)))   # residual FF
        cls = x[:, 0, :]
        return self.head(cls)

    def activations(self, tokens, seg, pad_mask):
        """CLS-token representation after the block (for permutation alignment of d_model channels)."""
        B, L = tokens.shape
        x = self.embed(tokens, seg)
        q = self.q(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        k = self.k(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        v = self.v(x).view(B, L, self.nhead, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        mask = pad_mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = x + self.o(ctx)
        h = F.relu(self.ff1(x))   # B,L,dim_ff  pre-residual hidden (alignable)
        return h[:, 0, :]         # CLS hidden, B x dim_ff
