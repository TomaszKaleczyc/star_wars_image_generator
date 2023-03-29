from typing import Optional

import einops
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
from torch import Tensor

from ..utils import ACTIVATIONS

from .gamma_layer_norm import GammaLayerNorm

import config


class AttentionBlock(nn.Module):
    """
    Standard attention block
    """

    def __init__(
            self,
            dim: int,
            dim_context: Optional[int] = None,
            heads: int = config.ATTENTION_BLOCK_HEADS,
            head_size: int = config.ATTENTION_HEAD_SIZE,
            activation_name: str = config.ACTIVATION,
            norm: bool = False,
            norm_context: bool = False,
            time_cond_dim: Optional[int] = None
        ) -> None:
        super().__init__()
        hidden_dim = head_size * heads
        dim_context = dim_context if dim_context is not None else dim

        self.time_cond = None
        self.activation = ACTIVATIONS[activation_name]()

        if time_cond_dim is not None:
            self.time_cond = nn.Sequential(
                self.activation,
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = head_size ** -0.5
        self.heads = heads

        self.norm = GammaLayerNorm(dim) if norm else nn.Identity()
        self.norm_context = GammaLayerNorm(dim_context) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_output = nn.Linear(hidden_dim, dim, bias = False)

    def forward(
            self,
            x: Tensor,
            context = None,
            time = None
        ) -> Tensor:
        h = self.heads

        x = self.norm(x)

        context = self.norm_context(context) if context is not None else x

        if self.time_cond is not None:
            assert time is not None
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q *= self.scale

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)

        output = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        output = einops.rearrange(output, 'b h n d -> b n (h d)')
        return self.to_output(output)
