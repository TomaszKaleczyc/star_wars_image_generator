from typing import List
from pathlib import Path, PosixPath

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
            self, data_dir: str = config.DATA_DIR, 
            img_size: int = config.IMG_SIZE,
            verbose: bool = config.VERBOSE
        ) -> None:
        self.data_dir = data_dir
        self.img_size = img_size
        self.all_image_paths = self._get_image_paths()
        if verbose:
            print('='*60)
            print('STAR WARS Image dataset')
            print(f'Loaded {len(self)} images')
        self.data_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # scaling to [0, 1]
                transforms.Lambda(lambda t: (t*2) -1)  # scaling to [-1, 1]
            ])

    def _get_image_paths(self) -> List[PosixPath]:
        """
        Returns a list of image paths
        """
        image_path_generator = Path(self.data_dir).glob('*/*.jpg')
        return list(image_path_generator)

    def __len__(self) -> int:
        return len(self.all_image_paths)

    def __getitem__(self, index: int) -> Tensor:
        img_path = self.all_image_paths[index]
        img = Image.open(img_path)
        return self.data_transforms(img)