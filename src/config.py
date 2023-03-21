# experiment setup:
RANDOM_STATE = 5
VERBOSE = True
NUM_WORKERS = 8
SAVE_PATH = '../output'

# data:
POC_DATA_DIR = "../data/stanford-cars/"
DATA_DIR = "../data/star-wars-images/"
IMG_SIZE = 64
BATCH_SIZE = 72
IMAGE_CHANNELS = 3
AUGMENTATIONS_RATIO = 4.

# diffusion:
BETA_SCHEDULER = 'cosine'  # 'linear', 'quadratic', 'sigmoid'
# TIMESTEPS = 300
TIMESTEPS = 1000
BETA_START = 1e-4
# BETA_END = 2e-2
BETA_END = 65e-4

# model:
NUM_MODULE_LAYERS = 5
NUM_TIME_EMBEDDINGS = 32
KERNEL_SIZE = 3
LOSS_FUNCTION = 'huber'  # 'l1', 'l2'
ACTIVATION = 'silu'  # 'relu', 'selu'
POSITION_EMBEDDINGS = 'sinusoidal_learned' # 'sinusoidal'

# training:
NUM_EPOCHS = 200
# LEARNING_RATE = 1e-3
LEARNING_RATE = 5e-4
LIMIT_VAL_BATCHES_RATIO = 0.05
SHOW_VALIDATION_IMAGES = True
NUM_VALIDATION_IMAGES = 5
