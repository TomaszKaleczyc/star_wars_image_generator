import math

import torch
import torch.nn as nn

from torch import Tensor


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim: int, add_original_time: bool = False) -> None:
        super().__init__()
        self.add_original_time = add_original_time
        self.dim = dim + 1 * int(self.add_original_time)

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        embeddings = torch.zeros(len(time), self.dim, device=device)
        position = torch.arange(0, len(time)).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.dim, 2, dtype=torch.float) *
                            - (math.log(10000.0) / self.dim)))
        embeddings[:, 0::2] = torch.sin(position.float() * div_term)
        embeddings[:, 1::2] = torch.cos(position.float() * div_term)
        if self.add_original_time:
            embeddings = torch.cat((time, embeddings), dim=-1)
        return embeddings
