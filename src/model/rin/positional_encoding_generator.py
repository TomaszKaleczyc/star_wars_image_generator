import math

import einops

import torch.nn as nn
from torch import Tensor

from ..utils import ACTIVATIONS

import config


class PositionalEncodingGenerator(nn.Module):
    """
    Positional Encoding Generator (PEG) to dynamically 
    produce the positional encodings conditioned on 
    the local neighborhood of an input token
    """

    def __init__(
            self,
            latent_width: int
        ) -> None:
        super().__init__()
        self.ds_conv = nn.Conv2d(
            in_channels = latent_width, 
            out_channels = latent_width, 
            kernel_size=3, 
            padding = 1, 
            groups = latent_width
            )

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        x = einops.rearrange(x, 'b (h w) d -> b d h w', h = hw)
        x = self.ds_conv(x)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        return x
