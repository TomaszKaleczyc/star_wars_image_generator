from typing import List, Tuple
from pathlib import Path, PosixPath

import numpy as np
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset

from torchvision import transforms

import config


class SWImageDataset(Dataset):
    """
    Manages the Star Wars image dataset
    """

    def __init__(
            self, 
            data_dir: str = config.DATA_DIR, 
            img_size: int = config.IMG_SIZE,
            verbose: bool = config.VERBOSE,
            augmentations_ratio: float = config.AUGMENTATIONS_RATIO,
        ) -> None:
        self.data_dir = data_dir
        self.img_size = img_size
        self.verbose = verbose
        assert augmentations_ratio >= 0
        self.augmentations_ratio = augmentations_ratio
        if self.verbose:
            print('='*60)
            print('STAR WARS Image dataset')

        self.all_image_paths = self._get_all_image_paths()

        if self.verbose:
            print(f'Loaded {len(self)} images')

        self.augmentations = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(3),
            transforms.RandomResizedCrop(size=self.img_size)
        ])

        self.data_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # scaling to [0, 1]
                transforms.Lambda(lambda t: (t*2) -1)  # scaling to [-1, 1]
            ])
    
    def _get_all_image_paths(self) -> List[Tuple[PosixPath, int]]:
        """
        Returns the list of all images to be used in the dataset
        """
        image_paths = self._get_image_paths()
        all_image_paths = [(image_path, 0) for image_path in image_paths]
        if self.augmentations_ratio == 0:
            return all_image_paths
        original_samples = len(image_paths)
        num_samples = int(original_samples * self.augmentations_ratio)
        samples = np.random.choice(image_paths, size=num_samples, replace=True)
        all_samples = [(image_path, 1) for image_path in samples]
        all_image_paths.extend(all_samples)
        if self.verbose:
            print(f'Dataset oversampled from {len(image_paths)} to {len(all_image_paths)} images')
        return all_image_paths

    def _get_image_paths(self) -> List[PosixPath]:
        """
        Returns a list of image paths
        """
        image_path_generator = Path(self.data_dir).glob('*/*.jpg')
        return list(image_path_generator)

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, index: int) -> Tensor:
        img_path, augment = self.all_image_paths[index]
        img = Image.open(img_path)
        if augment:
            img = self.augmentations(img)
        return self.data_transforms(img)
