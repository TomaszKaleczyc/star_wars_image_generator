from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .beta_schedules import BETA_SCHEDULES
from .diffusion_sampler import DiffusionSampler

import config


class DiscreteDiffusionSampler(DiffusionSampler):
    """
    Handles the diffusion process
    using discrete timesteps for training
    """

    def __init__(
            self,
            timesteps: int = config.TIMESTEPS,
            beta_start: float = config.BETA_START,
            beta_end: float = config.BETA_END,
            verbose: bool = config.VERBOSE,
            beta_scheduler: str = config.BETA_SCHEDULER
        ) -> None:
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_scheduler = BETA_SCHEDULES[beta_scheduler]
        if verbose:
            print('Diffusion parameters:')
            print('\t* Beta scheduler:', beta_scheduler)
            print('\t* Timesteps:', self.timesteps)
            print(f'\t* Starting beta: {self.beta_start:.4f}')
            print(f'\t* Final beta: {self.beta_end:.4f}')
        self._setup()

    def _setup(self) -> None:
        """
        Initiates the appropriate calculations for the diffusion
        """
        self.betas = self.beta_scheduler(
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            timesteps=self.timesteps
            )
        self.alphas = 1. - self.betas
        self.alphas_cumproduct = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumproduct_prev = F.pad(self.alphas_cumproduct[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_alphas_cumproduct = torch.sqrt(self.alphas_cumproduct)
        self.sqrt_one_minus_alphas_cumproduct = torch.sqrt(1. - self.alphas_cumproduct)
        self.posterior_variance = self.betas * (1. - self.alphas_cumproduct_prev) / (1. - self.alphas_cumproduct)

    def get_batch_timesteps(
            self,
            batch_size: int,
            device: str = "cpu"
        ) -> Tensor:
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        return t

    def forward_sample(
            self, 
            image: Tensor, 
            timestep: Tensor,
            device: str = "cpu"
        ) -> Tuple[Tensor, Tensor]:
        noise = torch.randn_like(image).to(device)
        sqrt_alphas_cumproduct_t = self._get_index_from_list(self.sqrt_alphas_cumproduct, timestep, image.shape)
        sqrt_one_minus_alphas_cumproduct_t = self._get_index_from_list(self.sqrt_one_minus_alphas_cumproduct, timestep, image.shape)
        output = sqrt_alphas_cumproduct_t.to(device) * image.to(device)
        output += sqrt_one_minus_alphas_cumproduct_t.to(device) * noise
        return output, noise

    @staticmethod
    def _get_index_from_list(values: Tensor, timestep: Tensor, input_shape: Tuple[int, int, int]) -> Tensor:
        """
        Returns the value of given tensor list at givn timestep
        """
        batch_size = timestep.shape[0]
        output = values.gather(-1, timestep.cpu())
        reshape_args = ((1,) * (len(input_shape) - 1))
        return output.reshape(batch_size, *reshape_args).to(timestep.device)
    
    @torch.no_grad()
    def sample_timestep(self, x: Tensor, pred: Tensor, t: Tensor) -> Tensor:
        betas_t = self._get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumproduct_t = self._get_index_from_list(
            self.sqrt_one_minus_alphas_cumproduct, t, x.shape 
        )
        sqrt_recip_alphas_t = self._get_index_from_list(
            self.sqrt_recip_alphas, t, x.shape
        )
        posterior_variance_t = self._get_index_from_list(
            self.posterior_variance, t, x.shape
        )

        # current image - noise prediction
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred / sqrt_one_minus_alphas_cumproduct_t
        )
        if t[0] == 0:
            return model_mean
        
        noise = torch.randn_like(x)
        noise_factor = torch.sqrt(posterior_variance_t) * noise
        output = model_mean + noise_factor
        return output
