import math

import torch
import torch.nn as nn

from torch import Tensor


class PositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @property
    def half_dim(self) -> int:
        return self.dim // 2

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        embeddings = math.log(1e5) / (self.half_dim - 1)
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings