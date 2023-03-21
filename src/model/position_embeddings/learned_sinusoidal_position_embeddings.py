import math

import einops
import torch
import torch.nn as nn
from torch import Tensor


class LearnedSinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim: int, add_original_time: bool = False) -> None:
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.add_original_time = add_original_time
        self.weights = nn.Parameter(torch.randn(half_dim))
        self.dim = dim + 1 * int(self.add_original_time)

    def forward(self, time: Tensor) -> Tensor:
        time = einops.rearrange(time, 'b -> b 1')
        freqs = time * einops.rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        if self.add_original_time:
            fouriered = torch.cat((time, fouriered), dim = -1)
        return fouriered
