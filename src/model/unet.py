from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from pytorch_lightning import LightningModule

from .position_embeddings import PositionEmbeddings
from .unet_block import UnetBlock

import config


class Unet(LightningModule):
    """
    Basic Unet architecture implementation
    """
    
    def __init__(
            self, 
            initial_unet_channels: int = config.INITIAL_UNET_CHANNELS,
            num_module_layers: int = config.NUM_MODULE_LAYERS,
            image_channels: int = config.IMAGE_CHANNELS,
            num_time_embeddings: int = config.NUM_TIME_EMBEDDINGS
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.initial_unet_channels = initial_unet_channels
        self.num_module_layers = num_module_layers
        self.image_channels = image_channels
        self.num_time_embeddings = num_time_embeddings

        self.downscaling_channels = self._get_channels()
        self.upscaling_channels = self._get_channels(upscaling=True)

        # time embeddings:
        self.time_mlp = nn.Sequential(
            PositionEmbeddings(self.num_time_embeddings),
            nn.Linear(self.num_time_embeddings, self.num_time_embeddings),
            nn.ReLU()
        )

        self.initial_conv = nn.Conv2d(self.image_channels, self.downscaling_channels[0], 3, padding=1)
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
        channel_num = self.initial_unet_channels 
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
