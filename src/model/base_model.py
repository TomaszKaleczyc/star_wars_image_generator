from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import torch
from torch import Tensor
from torchmetrics import MeanMetric
from torchvision import utils

from pytorch_lightning import Trainer

import config


class BaseModel(ABC):
    """
    Base diffusion model abstraction
    """
    train_loss: MeanMetric
    img_size: int
    timesteps: int
    learning_rate: float
    verbose: bool
    activation_name: str
    show_validation_images: bool
    loss_function_name: str
    trainer: Trainer

    @staticmethod
    def _print_state(var, var_name, print_state=config.PRINT_STATE):
        if not print_state:
            return
        print(var_name)
        print('shape', var.shape)
        print(f'mean: {var.mean():.4f}', f'max: {var.max():.4f}', f'min: {var.min():.4f}')
        print()    

    @torch.no_grad()
    def plot_samples(self, num_samples: int = config.NUM_VALIDATION_IMAGES) -> None:
        """
        Plots random generated output samples
        """
        self.eval()
        imgs_tensor = self._get_sample_batch(num_samples)
        file_dir = f'{self.trainer.log_dir}/{config.SAMPLES_FOLDER_NAME}'
        filename = f"random_samples_step_{self.trainer.global_step}"
        filepath = f"{file_dir}/{filename}.jpg"
        n_rows = int(num_samples**0.5)
        utils.save_image(imgs_tensor, fp=filepath, nrow=n_rows)
        print(f'Samples saved to {filepath}')

    @abstractmethod
    def _get_sample_batch(self, num_samples: int) -> Tensor:
        """
        Retrieves the complete denoised batch samples
        """

    @abstractmethod
    def _loss_step(
            self, 
            batch: Tuple[Tensor, Tensor], 
            batch_idx: int,
            dataset_name: str,
            criterion: Callable
            ) -> Tensor:
        """
        Basic loss generating step for model training
        """

    @abstractmethod
    def eval(self) -> None:
        """
        Stub for lightning module method
        """

    @abstractmethod
    def log(self) -> None:
        """
        Stub for lightning module method
        """

    def _training_epoch_end(self) -> None:
        """
        Perform standard epoch end activities
        """
        loss = self.train_loss.compute().item()
        self.log(f'mean_epoch_{self.loss_function_name}_loss', loss)
        self.train_loss.reset()
        if self.show_validation_images:
            self.plot_samples() 
