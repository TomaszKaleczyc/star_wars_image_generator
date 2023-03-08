import torch

from dataset import SWImageDataModule
from model import Unet

import config


data_module = SWImageDataModule()
train_loader = data_module.train_dataloader()
batch = next(iter(train_loader))

model = Unet(data_module.img_size)

batch_size = batch.shape[0]
timestep = torch.randint(0, config.TIMESTEPS, (batch_size,)).long()
print('timestep', timestep)
output = model(batch, timestep)
print(output.shape)