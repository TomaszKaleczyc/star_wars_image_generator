from dataset import DiffusionSampler, SWImageDataset

dsampler = DiffusionSampler()
dset = SWImageDataset()

tensor = dset.__getitem__(666)

from matplotlib import pyplot as plt
import numpy as np

from torch import Tensor

from torchvision import transforms

def show_tensor_image(tensor: Tensor) -> None:
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # get into [0, 1] scale
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # transform shape from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    image = reverse_transforms(tensor)
    plt.imshow(image)


plt.figure(figsize=(15, 5))
num_images = 10
stepsize = int(dsampler.timesteps / num_images)

for idx in range(0, dsampler.timesteps, stepsize):
    plt.subplot(1, num_images + 1, (idx // stepsize) + 1)
    image, noise = dsampler.forward_sample(tensor, idx)
    show_tensor_image(image)
    plt.axis('off')
plt.show()