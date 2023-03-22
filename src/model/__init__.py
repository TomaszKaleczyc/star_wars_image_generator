from typing import Dict

from pytorch_lightning import LightningModule

from .unet import Unet
from .rin import RIN

MODELS: Dict[str, LightningModule] = {
    'unet': Unet,
    'rin': RIN,
}
