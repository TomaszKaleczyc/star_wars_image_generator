from typing import Any, List, Optional

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import torch
from torch import Tensor
import torch.nn as nn

from pytorch_lightning import LightningModule

from diffusion import DiffusionSampler, DEFAULT_DIFFUSION_SAMPLER
from utils import image_utils

from .unet_block import UnetBlock
from ..position_embeddings import POSITION_EMBEDDINGS
from ..utils import ACTIVATIONS, LOSS_FUNCTIONS

import config


class Unet(LightningModule):
    """
    Basic Unet architecture implementation
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
            loss_function: str = config.LOSS_FUNCTION,
            activation: str = config.ACTIVATION,
            position_embeddings: str = config.POSITION_EMBEDDINGS
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.img_size = img_size
        self.num_module_layers = num_module_layers
        self.timesteps = timesteps
        self.image_channels = image_channels
        self.num_time_embeddings = num_time_embeddings
        self.learning_rate = learning_rate
        self.validation_images = show_validation_images
        self.position_embeddings = position_embeddings

        print('Unet model')
        print('Loss function:', loss_function)
        print('Activation function:', activation)
        self.loss_function = LOSS_FUNCTIONS[loss_function]
        self.activation = ACTIVATIONS[activation]()
        self.diffusion_sampler = diffusion_sampler if diffusion_sampler else DEFAULT_DIFFUSION_SAMPLER
        
        self.downscaling_channels = self._get_channels()
        self.upscaling_channels = self._get_channels(upscaling=True)

        # time embeddings:
        self.time_mlp = nn.Sequential(
            POSITION_EMBEDDINGS[self.position_embeddings](self.num_time_embeddings),
            nn.Linear(self.num_time_embeddings, self.num_time_embeddings),
            self.activation
        )

        self.initial_conv = nn.Conv2d(
            in_channels=self.image_channels, 
            out_channels=self.downscaling_channels[0], 
            kernel_size=3, 
            padding=1)
        
        self.downscaling = nn.ModuleList([
            UnetBlock(
                input_channels=self.downscaling_channels[idx],
                output_channels=self.downscaling_channels[idx + 1],
                num_time_embeddings=self.num_time_embeddings
            )
            for idx in range(len(self.downscaling_channels)-1)
        ])

        self.upscaling = nn.ModuleList([
            UnetBlock(
                input_channels=self.upscaling_channels[idx],
                output_channels=self.upscaling_channels[idx + 1],
                num_time_embeddings=self.num_time_embeddings,
                upscaling=True
            )
            for idx in range(len(self.upscaling_channels)-1)
        ])

        self.output = nn.Conv2d(
            in_channels=self.upscaling_channels[-1],
            out_channels=3,
            kernel_size=1
            )

    def _get_channels(self, upscaling: bool = False) -> List[int]:
        """
        Returns list of channel numbers for down/upscaling
        """
        channel_num = self.img_size 
        channels = []
        for _ in range(self.num_module_layers):
            channels.append(int(channel_num))
            channel_num *= 2
        return list(reversed(channels)) if upscaling else channels

    def forward(self, x: Tensor, timestep: int) -> Tensor:
        # Time embeddings
        t = self.time_mlp(timestep)
        # Initial conv:
        x = self.initial_conv(x)
        # Unet
        residual_inputs = []
        for downscale in self.downscaling:
            x = downscale(x, t)
            residual_inputs.append(x)
        for upscale in self.upscaling:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = upscale(x, t)
        x = self.output(x)
        return x
    
    def _loss_step(self, batch: Tensor) -> Tensor:
        """
        Standard loss step
        """
        if isinstance(batch, list):  # workaround for the PoC dset
            batch = batch[0]
        B = batch.shape[0]
        t = self.diffusion_sampler.get_batch_timesteps(batch_size=B, device=self.device)
        x_noisy, noise = self.diffusion_sampler.forward_sample(batch, t, device=self.device)
        noise_prediction = self(x_noisy, t)
        loss = self.loss_function(noise, noise_prediction)
        return loss
        
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch)
        self.log('training/loss', loss)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch)
        self.log('validation/loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        if self.validation_images:
            print()
            self.plot_sample()
            print()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @torch.no_grad()
    def plot_sample(self) -> None:
        self.eval()
        img_shape = (1, self.image_channels, self.img_size, self.img_size)
        img = torch.randn(img_shape, device=self.device)
        
        num_images = config.NUM_VALIDATION_IMAGES
        stepsize = int(self.timesteps / num_images)
        iterator = range(0, self.timesteps)[::-1]

        plt.figure(figsize=(15, 5))
        print('Diffusion progress:')
        for timestep in tqdm(iterator):
            t = torch.full((1,), timestep, device=self.device, dtype=torch.long)
            pred = self(img, t)
            img = self.diffusion_sampler.sample_timestep(img, pred, t)
            if timestep % stepsize == 0:
                ax = plt.subplot(1, num_images, timestep // stepsize + 1)
                image_utils.show_tensor_image(img.detach().cpu())
                ax.title.set_text(f'T {timestep}')
                ax.axis('off')
        plt.show()
