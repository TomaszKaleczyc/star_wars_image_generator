from typing import Optional

from einops.layers.torch import Rearrange

import torch.nn as nn
from torch import Tensor

from ..utils import ACTIVATIONS

from .gamma_layer_norm import GammaLayerNorm

import config


class FeedForward(nn.Module):
    """
    Feed forward block of the latent transformations
    """

    def __init__(
            self, 
            dim: int, 
            multiplier: int = 4, 
            time_cond_dim: Optional[int] = None,
            activation: str = config.ACTIVATION,
        ) -> None:
        super().__init__()
        self.norm = GammaLayerNorm(dim)

        self.time_cond = None
        self.activation = ACTIVATIONS[activation]()

        if time_cond_dim is not None:
            self.time_cond = nn.Sequential(
                self.activation,
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * multiplier)
        self.feed_forward_net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            self.activation,
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x: Tensor, time: Optional[int] = None) -> Tensor:
        x = self.norm(x)
        if self.time_cond is not None:
            assert time is not None
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift
        return self.feed_forward_net(x)
