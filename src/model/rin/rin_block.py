from typing import Tuple
import torch.nn as nn
from torch import Tensor

from ..utils import ACTIVATIONS

from .attention_block import AttentionBlock
from .feed_forward import FeedForward
from .gamma_layer_norm import GammaLayerNorm
from .linear_attention_block import LinearAttentionBlock
from .positional_encoding_generator import PositionalEncodingGenerator

import config


class RINBlock(nn.Module):
    """
    Standard RIN block
    """

    def __init__(
            self,
            patches_width: int,
            latent_width: int,
            latent_self_attn_depth: int,
            final_norm: bool = True,
            **attn_kwargs
        ) -> None:
        super().__init__()
        self.latents_attend_to_patches = AttentionBlock(
            latent_width, 
            dim_context = patches_width, 
            norm = True, 
            norm_context = True, 
            **attn_kwargs
            )
        self.latents_cross_attn_ff = FeedForward(latent_width)

        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                AttentionBlock(latent_width, norm = True, **attn_kwargs),
                FeedForward(latent_width)
            ]))

        self.latent_final_norm = GammaLayerNorm(latent_width) if final_norm else nn.Identity()

        self.patches_peg = PositionalEncodingGenerator(patches_width)
        self.patches_self_attn = LinearAttentionBlock(patches_width, norm=True, **attn_kwargs)
        self.patches_self_attn_ff = FeedForward(patches_width)

        self.patches_attend_to_latents = AttentionBlock(
                patches_width,
                dim_context=latent_width, 
                norm=True, 
                norm_context=True,
                **attn_kwargs
            )
        self.patches_cross_attn_ff = FeedForward(patches_width)

    def forward(self, patches: Tensor, latents: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        patches = self.patches_peg(patches) + patches

        # latents extract or cluster information from the patches
        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents
        latents = self.latents_cross_attn_ff(latents, time = t) + latents

        # latent self attention
        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        # additional patches self attention with linear attention
        patches = self.patches_self_attn(patches, time = t) + patches
        patches = self.patches_self_attn_ff(patches) + patches

        # patches attend to the latents
        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches

        patches = self.patches_cross_attn_ff(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents
