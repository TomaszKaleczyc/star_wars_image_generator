import shutil
from pathlib import Path

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

import config


class PrepareExperimentFolder(Callback):
    """
    Manages the experiment folder preparation
    """


    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._copy_config(trainer)
        self._create_samples_folder(trainer)

    def _copy_config(self, trainer: Trainer) -> None:
        """
        Copies the config used for the model 
        in the experiment location
        """
        config_path = "./config.py"
        target_path = f"{trainer.log_dir}/config.py"
        shutil.copy(config_path, target_path)
        print(f'Current config copied to {target_path}')

    def _create_samples_folder(self, trainer) -> None:
        """
        Creates a folder to store sample images
        """
        samples_dir = f"{trainer.log_dir}/{config.SAMPLES_FOLDER_NAME}"
        Path(samples_dir).mkdir(exist_ok=True)
