from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


class DiffusionSampler(ABC):
    """
    Diffusion sampler class abstraction
    """

    @abstractmethod
    def get_batch_timesteps(
            self,
            batch_size: int,
            device: str
        ) -> Tensor:
        """
        Returns the tensor of batch timesteps
        """
    
    @abstractmethod
    def forward_sample(
                self, 
                image: Tensor, 
                timestep: Tensor,
                device: str = "cpu"
            ) -> Tuple[Tensor, Tensor]:
        """
        Returns the image at noising timestep along with the noise itself
        """

    @abstractmethod
    def sample_timestep(self, x: Tensor, pred: Tensor, t: Tensor) -> Tensor:
        """
        Performs sampling for a given timestep, image and prediction
        """
