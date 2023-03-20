from typing import Tuple
import torch.nn as nn

from torch import Tensor

from ..utils import ACTIVATIONS

import config


class UnetBlock(nn.Module):
    """
    Defines a single Unet downscaling / upscaling block
    with residual connections
    """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            num_time_embeddings: int,
            kernel_size: int = config.KERNEL_SIZE,
            activation: str = config.ACTIVATION,
            upscaling: bool = False
        ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_time_embeddings = num_time_embeddings
        self.kernel_size = kernel_size
        self.upscaling = upscaling

        self.activation = ACTIVATIONS[activation]()

        self.time_mlp = nn.Linear(self.num_time_embeddings, self.output_channels)
        self.conv1, self.transform = self._get_block_variants()
        self.conv2 = nn.Conv2d(
                        in_channels=self.output_channels,
                        out_channels=self.output_channels,
                        kernel_size=self.kernel_size,
                        padding=1
                    )
        self.first_pass = self._get_convolution_pass(self.conv1) 
        self.second_pass = self._get_convolution_pass(self.conv2)
        self.residual_connection = self._get_residual_connection()
        self.batch_norm = nn.BatchNorm2d(num_features=self.output_channels)

    def _get_block_variants(self) -> Tuple[nn.Module, nn.Module]:
        """
        Returns the transformation blocks depending on the block variant
        """
        multiplier = 2 if self.upscaling else 1
        conv1 = nn.Conv2d(
                    in_channels=multiplier*self.input_channels, 
                    out_channels=self.output_channels,
                    kernel_size=self.kernel_size,
                    padding=1
                )
        if self.upscaling:
            transform = nn.ConvTranspose2d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        else:
            transform = nn.Conv2d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        return conv1, transform  

    def _get_convolution_pass(self, conv_layer: nn.Module) -> nn.Module:
        """
        Returns a standard Unet concolution block
        """
        convolution_pass = nn.Sequential(
            conv_layer,
            self.activation,
            nn.BatchNorm2d(num_features=self.output_channels)
        )
        return convolution_pass
    
    def _get_residual_connection(self) -> nn.Module:
        """
        Returns the residual connections block
        """
        if self.input_channels == self.output_channels:
            return nn.Identity()
        multiplier = 2 if self.upscaling else 1
        residual_connection = nn.Conv2d(
            in_channels=self.input_channels * multiplier,
            out_channels=self.output_channels,
            kernel_size=1
        )
        return residual_connection
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # First convolution:
        x_h = self.first_pass(x)

        # Time embedding
        time_embedding = self.activation(self.time_mlp(t))
        B, C = x_h.shape[:2]
        time_embedding = time_embedding.view(B, C, 1, 1)
        x_h += time_embedding

        # second convolution:
        x_h = self.second_pass(x_h)

        # residual connection:
        x_h += self.residual_connection(x)
        x_h = self.batch_norm(x_h)

        # Down / upscale
        output = self.transform(x_h)
        return output
