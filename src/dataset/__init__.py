from .poc_data_module import PoCDataModule
from .sw_image_dataset import SWImageDataset
from .sw_image_data_module import SWImageDataModule


DATASETS = {
    'PoC': PoCDataModule,
    'StarWars': SWImageDataModule,
}
