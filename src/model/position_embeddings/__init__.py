from typing import Dict

from torch.nn import Module

from .sinusoidal_position_embeddings import SinusoidalPositionEmbeddings
from .learned_sinusoidal_position_embeddings import LearnedSinusoidalPositionEmbeddings


POSITION_EMBEDDINGS: Dict[str, Module] = {
    'sinusoidal': SinusoidalPositionEmbeddings,
    'sinusoidal_learned': LearnedSinusoidalPositionEmbeddings,
}