from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .beta_schedules import BETA_SCHEDULES
from .diffusion_sampler import DiffusionSampler

import config


class ContinuousDiffusionSampler(DiffusionSampler):
    """
    Handles the diffusion process
    using continuous timesteps for training
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

    def get_batch_timesteps(
            self,
            batch_size: int,
            device: str = "cpu"
        ) -> Tensor:
        t = torch.rand((batch_size,), device=device)
        return t