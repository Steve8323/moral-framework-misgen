import torch
import torch.nn as nn


class MLP(nn.Module):
    """2-hidden-layer MLP. Enough capacity to learn either heuristic or true ordered-subseq rule."""

    def __init__(self, in_dim, width=128, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        self.width = width
        self.depth = depth

    def forward(self, x):
        return self.net(x)

    def activations(self, x):
        """Return list of post-ReLU activations per hidden layer (for permutation alignment)."""
        acts = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h)
        return acts
