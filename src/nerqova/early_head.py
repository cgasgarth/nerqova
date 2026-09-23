"""Small nonlinear readout for decision anchors at an intermediate layer."""

import torch
from torch import nn
from torch.nn import functional as F


class PairHead(nn.Module):
    def __init__(self, dimension=2560, width=384):
        super().__init__()
        self.query = nn.Linear(dimension, width)
        self.option = nn.Linear(dimension, width)
        self.joint = nn.Sequential(nn.Linear(width * 4, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, decide, options):
        q = F.silu(self.query(decide))[:, None, :]
        k = F.silu(self.option(options))
        q = q.expand_as(k)
        pair = torch.cat((q, k, q * k, torch.abs(q - k)), dim=-1)
        return self.joint(pair).squeeze(-1)
