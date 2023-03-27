from typing import Any, List, Optional

from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn as nn
from torchvision import utils

from pytorch_lightning import LightningModule

from diffusion import DiffusionSampler, DEFAULT_DIFFUSION_SAMPLER

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
            loss_function_name: str = config.LOSS_FUNCTION,
            activation: str = config.ACTIVATION,
            position_embeddings: str = config.POSITION_EMBEDDINGS,
            verbose: bool = config.VERBOSE
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
        self.loss_function_name = loss_function_name
        self.verbose = verbose

        if self.verbose:
            print('Unet model')
            print('Loss function:', self.loss_function_name )
            print('Activation function:', activation)
            print('Position embeddings', position_embeddings)
        self.loss_function = LOSS_FUNCTIONS[self.loss_function_name ]
        self.activation = ACTIVATIONS[activation]()
        self.diffusion_sampler = diffusion_sampler if diffusion_sampler else DEFAULT_DIFFUSION_SAMPLER
        
        self.downscaling_channels = self._get_channels()
        self.upscaling_channels = self._get_channels(upscaling=True)

        # time embeddings:
        position_embeddings = POSITION_EMBEDDINGS[self.position_embeddings](self.num_time_embeddings)
        self.time_mlp = nn.Sequential(
            position_embeddings,
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
        self.log(f'training/{self.loss_function_name}_loss', loss)
        return loss
    
    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._loss_step(batch)
        self.log(f'validation/{self.loss_function_name}_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        if self.validation_images:
            print()
            self.plot_samples()
            print()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @torch.no_grad()
    def plot_samples(self, num_samples: int = 4) -> None:
        """
        Plots random generated output samples
        """
        self.eval()
        imgs_tensor = self._get_sample_batch(num_samples)
        file_dir = self.trainer.log_dir
        filename = f"random_samples_step_{self.trainer.global_step}"
        filepath = f"{file_dir}/{filename}.jpg"
        n_rows = int(num_samples**0.5)
        utils.save_image(imgs_tensor, fp=filepath, nrow=n_rows)

    @torch.no_grad()
    def _get_sample_batch(self, batch_size: int) -> Tensor:
        """
        Retrieves the complete denoised batch samples
        """
        img_shape = (batch_size, self.image_channels, self.img_size, self.img_size)
        img = torch.randn(img_shape, device=self.device)
        iterator = range(0, self.timesteps)[::-1]
        print('Diffusion progress:')
        for timestep in tqdm(iterator):
            t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            pred = self(img, t)
            img = self.diffusion_sampler.sample_timestep(img, pred, t)
        return img
