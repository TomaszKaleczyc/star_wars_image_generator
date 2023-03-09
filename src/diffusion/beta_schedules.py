from typing import Callable, Dict

import torch
from torch import Tensor


def cosine_beta_schedule(beta_start: float, beta_end: float, timesteps: int, s: float = 0.008) -> Tensor:
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, beta_start, beta_end)


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> Tensor:
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> Tensor:
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


BETA_SCHEDULES: Dict[str, Callable] = {
    'cosine': cosine_beta_schedule,
    'linear': linear_beta_schedule,
    'quadratic': quadratic_beta_schedule,
    'sigmoid': sigmoid_beta_schedule,
}
