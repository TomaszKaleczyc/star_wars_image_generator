from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from torchvision import datasets, transforms

import config



class PoCDataModule(LightningDataModule):
    """
    Manages the model datasets
    for training the PoC model on
    the Stanford Cars dataset
    """

    def __init__(
                self, 
                data_dir: str = config.POC_DATA_DIR,
                img_size: int = config.IMG_SIZE,
                batch_size: int = config.BATCH_SIZE
            ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size            

        self.data_transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), # scaling to [0, 1]
                transforms.Lambda(lambda t: (t*2) -1)  # scaling to [-1, 1]
            ])
        
        self.train_dataset = self._get_dataset('train')
        self.val_dataset = self._get_dataset('test')

    def _get_dataset(self, split: str) -> Dataset:
        dataset = datasets.StanfordCars(
            root=self.data_dir,
            download=True,
            transform=self.data_transforms,
            split=split
        )
        return dataset

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
