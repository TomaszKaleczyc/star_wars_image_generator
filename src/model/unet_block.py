from typing import Tuple
import torch
import torch.nn as nn

from torch import Tensor

import config


class UnetBlock(nn.Module):
    """
    Defines a single Unet downscaling / upscaling block
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_time_embeddings: int,
            kernel_size: int = config.KERNEL_SIZE,
            upscaling: bool = False
        ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_time_embeddings = num_time_embeddings
        self.kernel_size = kernel_size

        self.time_mlp = nn.Linear(self.num_time_embeddings, self.output_channels)
        self.conv1, self.transform = self._get_block_variants(upscaling)
        self.conv2 = nn.Conv2d(
                        in_channels=self.output_channels,
                        out_channels=self.output_channels,
                        kernel_size=self.kernel_size,
                        padding=1
                    )
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(num_features=self.output_channels)

    def _get_block_variants(self, upscaling: bool) -> Tuple[nn.Module, nn.Module]:
        """
        Returns the transformation blocks depending on the block variant
        """
        # TODO hardcoded vars to config?
        if upscaling:
            conv1 = nn.Conv2d(
                in_channels=2*self.input_channels, 
                out_channels=self.output_channels,
                kernel_size=self.kernel_size,
                padding=1
                )
            transform = nn.ConvTranspose2d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        else:
            conv1 = nn.Conv2d(
                in_channels=self.input_channels, 
                out_channels=self.output_channels,
                kernel_size=3,
                padding=1
                )
            transform = nn.Conv2d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        return conv1, transform    
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # First convolution:
        h = self.batch_norm(self.activation(self.conv1(x)))

        # Time embedding
        time_embedding = self.activation(self.time_mlp(t))
        time_embedding = time_embedding[(..., ) + (None, ) * 2]

        h += time_embedding
        h = self.batch_norm(self.activation(self.conv2(h)))
        # Down / upscale
        output = self.transform(h)
        return output
