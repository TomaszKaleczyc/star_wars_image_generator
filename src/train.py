from typing import Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import DATASETS
from diffusion import DIFFUSION_SAMPLERS
from model import MODELS, callbacks

import config


CHECKPOINT_PATH = None


def main() -> None:
    data_module = DATASETS[config.DATASET]()
    diffusion_sampler = DIFFUSION_SAMPLERS[config.DIFFUSION_SAMPLER]()
    model_type = MODELS[config.MODEL_TYPE]

    model = model_type(data_module.img_size, diffusion_sampler=diffusion_sampler)
    if CHECKPOINT_PATH:
        model = model.load_from_checkpoint(CHECKPOINT_PATH)

    monitor_metric = f'mean_epoch_{config.LOSS_FUNCTION}_loss'
    filename = config.DATASET + '-{epoch}-{' + monitor_metric + ':.3f}'
    model_callbacks = [
        ModelCheckpoint(
            filename=filename,
            monitor=monitor_metric, 
            save_top_k=1,
            save_last=True,
            verbose=True, 
            mode='min'
        ),
        callbacks.PrepareExperimentFolder(),
    ]

    trainer = Trainer(
        max_epochs=config.NUM_EPOCHS,
        fast_dev_run=False,
        default_root_dir=config.SAVE_PATH,
        callbacks=model_callbacks,
        gradient_clip_val=0.5,
        accumulate_grad_batches=config.ACCUMULATE_GRADIENT_EVERY_N_BATCHES,
        accelerator='gpu',
        devices=1,
    )

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )


if __name__ == "__main__":
    main()
