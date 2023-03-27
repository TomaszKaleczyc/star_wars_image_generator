# experiment setup:
RANDOM_STATE = 5
VERBOSE = True
NUM_WORKERS = 4
SAVE_PATH = '../output'

# data:
POC_DATA_DIR = "../data/stanford-cars/"
DATA_DIR = "../data/star-wars-images/"
IMG_SIZE = 64
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
MODEL_TYPE = 'unet'  # 'rin'
NUM_TIME_EMBEDDINGS = 32
LOSS_FUNCTION = 'l2' #'huber'  # 'l1', 'l2'
ACTIVATION = 'gelu' #'silu' 'relu' 'selu'
POSITION_EMBEDDINGS = 'sinusoidal_learned' # 'sinusoidal'
# Unet specific:
NUM_MODULE_LAYERS = 5
KERNEL_SIZE = 3
# RIN specific:
RIN_PATCH_SIZE = 16
RIN_LATENT_WIDTH = 256
RIN_LATENT_SELF_ATTENTION_DEPTH = 4
RIN_NUM_BLOCKS = 6
RIN_NUM_LATENTS = 128
TRAIN_PROBABILITY_SELF_CONDITIONING = 0.9

# training:
NUM_EPOCHS = 1
BATCH_SIZE = 32
# LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4
# LEARNING_RATE = 1e-3
LIMIT_VAL_BATCHES_RATIO = 0.05
SHOW_VALIDATION_IMAGES = True
NUM_VALIDATION_IMAGES = 5
