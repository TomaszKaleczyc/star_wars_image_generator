import os
import glob
from typing import List

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import torch
from torch import Tensor

from torchvision import transforms


def show_tensor_image(tensor: Tensor) -> None:
    """
    Displays model output as image
    """
    tensor = torch.clamp(tensor, -1., 1.)
    if len(tensor.shape) == 4:
        tensor = tensor[0, :, :, :]
        
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  # get into [0, 1] scale
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # transform shape from CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    image = reverse_transforms(tensor)
    plt.imshow(image)


def make_gif(
        frame_folder_dir: str, 
        output_filename: str, 
        save_dir: str = '.',
        repeat_last_frame: int = 10
        ) -> None:
    """
    Saves gif given a directory containing frame images
    """
    frame_paths = list(glob.glob(f"{frame_folder_dir}/*.jpg"))
    sorted_frame_paths = sort_frame_paths_by_creation_date(frame_paths)
    frames = [Image.open(image) for image in sorted_frame_paths]
    if repeat_last_frame:
        frames.extend([frames[-1]] * repeat_last_frame)
    frame_one = frames[0]
    frame_one.save(
        f"{save_dir}/{output_filename}.gif", 
        format="GIF", 
        append_images=frames,
        save_all=True, 
        duration=100, 
        loop=0
        )
    

def sort_frame_paths_by_creation_date(
        frame_paths: List[str], 
        reverse=False
    ) -> List[str]:
    """
    Returns list of image paths sorted by image creation date
    """
    get_creation_date = lambda frame_path: os.path.getctime(frame_path)
    frame_paths_with_ctime = [
        (frame_path, get_creation_date(frame_path))
        for frame_path in frame_paths
    ]
    sorting_key = lambda item: item[-1]
    sorted_frame_paths = [
        frame_path[0] for frame_path 
        in sorted(frame_paths_with_ctime, key=sorting_key, reverse=reverse)
        ]
    return sorted_frame_paths
