import shutil

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class SaveConfig(Callback):
    """
    Copies the config used for the model in the experiment location
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        config_path = "./config.py"
        target_path = f"{trainer.log_dir}/config.py"
        shutil.copy(config_path, target_path)
        print(f'Current config copied to {target_path}')
