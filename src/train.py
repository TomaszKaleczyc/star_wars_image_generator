from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from dataset import SWImageDataModule
from diffusion import DiffusionSampler
from model import Unet

import config


data_module = SWImageDataModule()

diffusion_sampler = DiffusionSampler()

model = Unet(data_module.img_size, sampler=diffusion_sampler)

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
