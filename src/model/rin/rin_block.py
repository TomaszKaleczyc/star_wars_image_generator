from typing import Tuple
import torch.nn as nn
from torch import Tensor

from .attention_block import AttentionBlock
from .feed_forward import FeedForward
from .gamma_layer_norm import GammaLayerNorm
from .linear_attention_block import LinearAttentionBlock
from .positional_encoding_generator import PositionalEncodingGenerator


class RINBlock(nn.Module):
    """
    Standard RIN block
    """

    def __init__(
            self,
            patches_width: int,
            latent_width: int,
            latent_self_attention_depth: int,
            patches_self_attention_depth: int,
            final_norm: bool = True,
            **attn_kwargs
        ) -> None:
        super().__init__()
        self.patches_peg = PositionalEncodingGenerator(patches_width)

        self.latents_attend_to_patches = AttentionBlock(
            latent_width, 
            dim_context = patches_width, 
            norm = True, 
            norm_context = True, 
            **attn_kwargs
            )
        self.latents_cross_attention_feed_forward = FeedForward(latent_width)

        self.latent_self_attentions = nn.ModuleList([])
        for _ in range(latent_self_attention_depth):
            self.latent_self_attentions.append(nn.ModuleList([
                AttentionBlock(latent_width, norm = True, **attn_kwargs),
                FeedForward(latent_width)
            ]))

        self.patches_self_attentions = nn.ModuleList([])
        for _ in range(patches_self_attention_depth):
            self.patches_self_attentions.append(nn.ModuleList([
                AttentionBlock(patches_width, norm = True, **attn_kwargs),
                FeedForward(patches_width)
            ]))

        self.patches_attend_to_latents = AttentionBlock(
                patches_width,
                dim_context=latent_width, 
                norm=True, 
                norm_context=True,
                **attn_kwargs
            )
        self.patches_cross_attention_feed_forward = FeedForward(patches_width)

        self.latent_final_norm = GammaLayerNorm(latent_width) if final_norm else nn.Identity()

    def forward(self, patches: Tensor, latents: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        patches = self.patches_peg(patches) + patches

        # latents extract or cluster information from the patches
        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents
        latents = self.latents_cross_attention_feed_forward(latents, time = t) + latents

        # latent self attention
        for attention_block, feed_forward_block in self.latent_self_attentions:
            latents = attention_block(latents, time = t) + latents
            latents = feed_forward_block(latents, time = t) + latents

        # additional patches self attention with linear attention
        for attention_block, feed_forward_block in self.patches_self_attentions:
            patches = attention_block(patches, time = t) + patches
            patches = feed_forward_block(patches) + patches

        # patches attend to the latents
        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches
        patches = self.patches_cross_attention_feed_forward(patches, time = t) + patches

        latents = self.latent_final_norm(latents)
        return patches, latents
