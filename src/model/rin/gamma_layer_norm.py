import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class GammaLayerNorm(nn.Module):
    """
    Layer norm without bias
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
