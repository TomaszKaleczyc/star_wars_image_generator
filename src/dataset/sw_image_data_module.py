from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .sw_image_dataset import SWImageDataset

import config



class SWImageDataModule(LightningDataModule):
    """
    Manages the model datasets
    """

    def __init__(
                self, 
                data_dir: str = config.DATA_DIR,
                img_size: int = config.IMG_SIZE,
                batch_size: int = config.BATCH_SIZE,
                verbose: bool = config.VERBOSE
            ) -> None:
        super().__init__()
        self.dataset_name = 'StarWars_images'
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.verbose = verbose  

        if self.verbose:
            print('='*60)
            print('TRAINING DATA')
        self.train_dataset = SWImageDataset(
            data_dir=self.data_dir,
            img_size=self.img_size
        )
        if self.verbose:
            print('='*60)
            print('VALIDATION DATA')
        self.val_dataset = SWImageDataset(
            data_dir=self.data_dir,
            img_size=self.img_size,
            augmentations_ratio=0,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=config.NUM_WORKERS
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=config.NUM_WORKERS
            )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
