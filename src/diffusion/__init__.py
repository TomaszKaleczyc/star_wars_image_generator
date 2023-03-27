from .diffusion_sampler import DiffusionSampler
from .discrete_diffusion_sampler import DiscreteDiffusionSampler


DIFFUSION_SAMPLERS = {
    'discrete': DiscreteDiffusionSampler,
    # 'continuous': NotImplementedError
}

DEFAULT_DIFFUSION_SAMPLER = DiscreteDiffusionSampler
