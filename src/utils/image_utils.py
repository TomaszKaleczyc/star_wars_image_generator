from matplotlib import pyplot as plt
import numpy as np

from torch import Tensor

from torchvision import transforms


def show_tensor_image(tensor: Tensor) -> None:
    """
    Displays model output as image
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # get into [0, 1] scale
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # transform shape from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    image = reverse_transforms(tensor)
    plt.imshow(image)
