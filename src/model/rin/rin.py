import math
from random import random
from typing import Any, Optional, Tuple

from tqdm import tqdm

import einops
from einops.layers.torch import Rearrange

import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import MeanMetric

from pytorch_lightning import LightningModule

from diffusion import DiffusionSampler, DEFAULT_DIFFUSION_SAMPLER

from model.base_model import BaseModel
from model.position_embeddings import POSITION_EMBEDDINGS
from model.utils import ACTIVATIONS, LOSS_FUNCTIONS

from .feed_forward import FeedForward
from .gamma_layer_norm import GammaLayerNorm
from .rin_block import RINBlock

import config


class RIN(LightningModule, BaseModel):
    """
    Basic Recurrent Interface Network architecture implementation
    """
    diffusion_sampler: DiffusionSampler

    def __init__(
            self, 
            img_size: int,
            diffusion_sampler: Optional[DiffusionSampler] = None,
            num_module_layers: int = config.NUM_MODULE_LAYERS,
            image_channels: int = config.IMAGE_CHANNELS,
            num_time_embeddings: int = config.NUM_TIME_EMBEDDINGS,
            timesteps: int = config.TIMESTEPS,
            learning_rate: float = config.LEARNING_RATE,
            show_validation_images: bool = config.SHOW_VALIDATION_IMAGES,
            loss_function_name: str = config.LOSS_FUNCTION,
            activation_name: str = config.ACTIVATION,
            position_embeddings_name: str = config.POSITION_EMBEDDINGS,
            verbose: bool = config.VERBOSE,
            num_latents: int = config.NUM_LATENTS,
            patches_width: int = config.PATCHES_WIDTH,
            latent_width: int = config.LATENT_WIDTH,
            latent_self_attention_depth: int = config.LATENT_SELF_ATTENTION_DEPTH,
            num_blocks: int = config.NUM_BLOCKS,
            latent_token_time_cond = False,
            train_probability_self_conditioning: float = config.TRAIN_PROBABILITY_SELF_CONDITIONING,
            patch_size: int = config.PATCH_SIZE
        ) -> None:
        super().__init__()
        self.save_hyperparameters()  

        self.train_loss = MeanMetric()
        self.img_size = img_size
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.loss_function_name = loss_function_name
        self.activation_name = activation_name
        self.show_validation_images = show_validation_images
             
        self.num_module_layers = num_module_layers
        self.image_channels = image_channels
        self.num_time_embeddings = num_time_embeddings
        self.position_embeddings_name = position_embeddings_name
        self.patch_size = patch_size
        self.patches_width = patches_width
        self.latent_width = latent_width
        self.num_latents = num_latents
        self.latent_token_time_cond = latent_token_time_cond
        self.latent_self_attention_depth = latent_self_attention_depth
        self.num_blocks = num_blocks
        self.train_probability_self_conditioning = train_probability_self_conditioning

        self.loss_function = LOSS_FUNCTIONS[loss_function_name]
        self.activation = ACTIVATIONS[activation_name]()
        self.diffusion_sampler = diffusion_sampler if diffusion_sampler else DEFAULT_DIFFUSION_SAMPLER

        # architecture setup:
        self._set_time_position_embeddings()
        self._set_image_transformation_layers()
        self._set_axial_positional_embeddings()
        self._set_latent_layers()

    def _set_time_position_embeddings(self) -> None:
        """
        Defines the time embeddings
        """
        # time embeddings:
        position_embeddings = POSITION_EMBEDDINGS[self.position_embeddings_name](
            self.num_time_embeddings, 
            add_original_time=False
            )
        time_dim = self.patches_width * 4
        self.time_mlp = nn.Sequential(
            position_embeddings,
            nn.Linear(position_embeddings.dim, time_dim),
            self.activation,
            nn.Linear(time_dim, time_dim)
        )

    def _set_image_transformation_layers(self) -> None:
        """
        Defines the layers that transform the image 
        into and from patches
        """
        # image to and from patches:
        image_channels = 3
        patch_height_width = self.img_size // self.patch_size
        pixel_patch_dim = image_channels * (self.patch_size ** 2)

        self.to_patches = nn.Sequential(
            Rearrange(
                pattern='b c (h psh) (w psw) -> b (h w) (c psh psw)', 
                psh=self.patch_size, 
                psw=self.patch_size
            ),
            nn.LayerNorm(pixel_patch_dim * 2),  # doubled for self-conditioning
            nn.Linear(pixel_patch_dim * 2, self.patches_width),
            nn.LayerNorm(self.patches_width)
        )

        self.to_pixels = nn.Sequential(
            GammaLayerNorm(self.patches_width),
            nn.Linear(self.patches_width, pixel_patch_dim),
            Rearrange(
                pattern='b (h w) (c psh psw) -> b c (h psh) (w psw)', 
                psh=self.patch_size, 
                psw=self.patch_size, 
                h=patch_height_width
            )
        )

    def _set_axial_positional_embeddings(self) -> None:
        """
        Defines the width & height positional embeddings
        """
        pos_emb_dim = self.patches_width // 2
        self.axial_pos_emb_height_mlp = nn.Sequential(
            Rearrange(pattern='... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            self.activation,
            nn.Linear(pos_emb_dim, pos_emb_dim),
            self.activation,
            nn.Linear(pos_emb_dim, self.patches_width)
        )

        self.axial_pos_emb_width_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            self.activation,
            nn.Linear(pos_emb_dim, pos_emb_dim),
            self.activation,
            nn.Linear(pos_emb_dim, self.patches_width)
        )

    def _set_latent_layers(self) -> None:
        """
        Defines the latent block architecture
        """
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_width))
        nn.init.normal_(self.latents, std = 0.02)  # TODO: parametrize?

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(self.latent_width),
            GammaLayerNorm(self.latent_width)
        )
        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        self.blocks = nn.ModuleList([
            RINBlock(
                patches_width=self.patches_width,
                latent_width=self.latent_width, 
                latent_self_attn_depth = self.latent_self_attention_depth
            ) 
            for _ in range(self.num_blocks)
        ])

    def forward(
            self,
            x: Tensor,
            time: Tensor,
            x_self_cond: Optional[Tensor] = None,
            latent_self_cond: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]

        if config.PRINT_STATE: print('v'*100)

        self._print_state(x, 'x-start')

        if config.PRINT_STATE: print('x_self_cond is None', x_self_cond is None)
        x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(x)
        self._print_state(x_self_cond, 'x_self_cond')

        # self conditioning added as additional channels:
        x = torch.cat((x_self_cond, x), dim = 1)
        self._print_state(x, 'x-cat')

        # prepare time conditioning
        t = self.time_mlp(time)
        self._print_state(t, 't')

        # prepare latents
        latents = einops.repeat(self.latents, 'n d -> b n d', b = B)
        self._print_state(latents, 'latents-init')

        # the warm starting of latents as in the paper
        if latent_self_cond is not None:
            latents = latents + self.init_self_cond_latents(latent_self_cond)
            self._print_state(latents, 'latents-self-cond')

        # whether the time conditioning is to be treated as one latent token or for projecting into scale and shift for adaptive layernorm
        if self.latent_token_time_cond:
            t = einops.rearrange(t, 'b d -> b 1 d')
            latents = torch.cat((latents, t), dim = -2)
            self._print_state(latents, 'latents-token-time')

        # to patches
        patches = self.to_patches(x)
        self._print_state(patches, 'patches-init')

        height_range = width_range = torch.linspace(0., 1., steps = int(math.sqrt(patches.shape[-2])), device = self.device)
        self._print_state(height_range, 'h-range')
        pos_emb_h, pos_emb_w = self.axial_pos_emb_height_mlp(height_range), self.axial_pos_emb_width_mlp(width_range)
        self._print_state(pos_emb_h, 'pos-emb-h')

        pos_emb = einops.rearrange(pos_emb_h, 'i d -> i 1 d') + einops.rearrange(pos_emb_w, 'j d -> 1 j d')
        self._print_state(pos_emb, 'pos-emb')
        patches = patches + einops.rearrange(pos_emb, 'i j d -> (i j) d')
        self._print_state(patches, 'patches+pos_emb')

        # the recurrent interface network body
        for idx, block in enumerate(self.blocks, 1):
            patches, latents = block(patches, latents, t)
            self._print_state(patches, f'patches-block-{idx}')
            self._print_state(latents, f'latents-block-{idx}')

        # to pixels
        pixels = self.to_pixels(patches)
        self._print_state(pixels, 'pixels')

        # remove time conditioning token, if that is the settings
        if self.latent_token_time_cond:
            latents = latents[:, :-1]
            self._print_state(latents, 'latenst-post')
        if config.PRINT_STATE: print('^'*100)
        return pixels, latents
    
    def _loss_step(self, batch: Tensor) -> Tensor:
        """
        Standard loss step
        """
        if isinstance(batch, list):  # workaround for the PoC dset
            batch = batch[0]
        B = batch.shape[0]
        t = self.diffusion_sampler.get_batch_timesteps(batch_size=B, device=self.device)
        noisy_image, noise = self.diffusion_sampler.forward_sample(batch, t, device=self.device)

        self_condition = self_latents = None
        if random() < self.train_probability_self_conditioning:
            with torch.no_grad():
                self_condition, self_latents = self(noisy_image, t)
                self_latents = self_latents.detach()
                self_condition.clamp_(-1, 1)
                self_condition = self_condition.detach()
        noise_prediction, _ = self(
            noisy_image, 
            t,            
            x_self_cond = self_condition,
            latent_self_cond = self_latents
            )
        loss = self.loss_function(noise, noise_prediction)
        self.train_loss.update(loss)
        return loss
        
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch)
        self.log(f'{self.loss_function_name}_loss', loss)
        return loss
    
    def training_epoch_end(self, training_step_outputs) -> None:
        self._training_epoch_end()     

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))

    @torch.no_grad()
    def _get_sample_batch(self, batch_size: int) -> Tensor:
        img_shape = (batch_size, self.image_channels, self.img_size, self.img_size)
        img = torch.randn(img_shape, device=self.device)
        iterator = range(0, self.timesteps)[::-1]
        print('Diffusion progress:')
        for timestep in tqdm(iterator):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            pred, _ = self(img, t)
            img = self.diffusion_sampler.sample_timestep(img, pred, t)
        return img
