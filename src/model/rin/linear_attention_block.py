from typing import Optional

import einops
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import ACTIVATIONS

from .gamma_layer_norm import GammaLayerNorm

import config


class LinearAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int = config.ATTENTION_BLOCK_HEADS,
            head_size: int = config.ATTENTION_HEAD_SIZE,
            activation_name: str = config.ACTIVATION,
            norm: bool = False,
            time_cond_dim: Optional[int] = None,
            dropout_probability: float = config.DROPOUT_PROBABILITY
        ) -> None:
        super().__init__()
        hidden_dim = head_size * heads
        self.scale = head_size ** -0.5
        self.heads = heads
        self.activation = ACTIVATIONS[activation_name]()

        self.time_cond = None

        if time_cond_dim is not None:
            self.time_cond = nn.Sequential(
                self.activation,
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = GammaLayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.to_output = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            GammaLayerNorm(dim),
            nn.Dropout(dropout_probability)
        )

    def forward(
            self,
            x: Tensor,
            time: Optional[Tensor] = None
        ) -> Tensor:
        h = self.heads
        x = self.norm(x)

        if self.time_cond is not None:
            assert time is not None
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        output = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        output = einops.rearrange(output, 'b h n d -> b n (h d)')
        return self.to_output(output)
