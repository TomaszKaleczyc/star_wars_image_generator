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
        # device = time.device
        embeddings = torch.zeros(self.dim, len(time))
        position = torch.arange(0, self.dim).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, len(time), 2, dtype=torch.float) *
                            -(math.log(10000.0) / len(time))))
        embeddings[:, 0::2] = torch.sin(position.float() * div_term)
        embeddings[:, 1::2] = torch.cos(position.float() * div_term)
        return embeddings
