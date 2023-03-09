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

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        embeddings = torch.zeros(len(time), self.dim, device=device)
        position = torch.arange(0, len(time)).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.dim, 2, dtype=torch.float) *
                            - (math.log(10000.0) / self.dim)))
        embeddings[:, 0::2] = torch.sin(position.float() * div_term)
        embeddings[:, 1::2] = torch.cos(position.float() * div_term)
        return embeddings
