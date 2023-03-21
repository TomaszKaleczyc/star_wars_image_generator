from typing import Callable, Dict

import torch.nn as nn
import torch.nn.functional as F


LOSS_FUNCTIONS: Dict[str, Callable] = {
    'l1': F.l1_loss,
    'l2': F.mse_loss,
    'huber': F.smooth_l1_loss,
}


ACTIVATIONS: Dict[str, Callable] = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'selu': nn.SELU,
    'gelu': nn.GELU,
}
