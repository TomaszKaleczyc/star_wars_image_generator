from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import SWImageDataModule
from diffusion import DEFAULT_DIFFUSION_SAMPLER
from model import MODELS

import config


data_module = SWImageDataModule()

diffusion_sampler = DEFAULT_DIFFUSION_SAMPLER()

model_type = MODELS[config.MODEL_TYPE]

model = model_type(data_module.img_size, diffusion_sampler=diffusion_sampler)

callbacks = [
    ModelCheckpoint(
        filename='star_wars-{epoch}-{validation/loss:.3f}',
        monitor='validation/loss', 
        save_top_k=1,
        verbose=True, 
        mode='min'
    ),
]

trainer = Trainer(
    max_epochs=config.NUM_EPOCHS,
    fast_dev_run=False,
    default_root_dir=config.SAVE_PATH,
    callbacks=callbacks,
    limit_val_batches=config.LIMIT_VAL_BATCHES_RATIO,
    accelerator='gpu',
    devices=1,
)

trainer.fit(
    model,
    train_dataloaders=data_module.train_dataloader(),
    val_dataloaders=data_module.val_dataloader()
)
