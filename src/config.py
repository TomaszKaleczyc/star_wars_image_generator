# experiment setup:
RANDOM_STATE = 5
VERBOSE = True
NUM_WORKERS = 4
SAVE_PATH = '../output'
SAMPLES_FOLDER_NAME = 'samples'
PRINT_STATE = False

# data:
DATASET = 'PoC' # 'StarWars'
POC_DATA_DIR = "../data/stanford-cars/"
DATA_DIR = "../data/star-wars-images/"
IMG_SIZE = 64
IMAGE_CHANNELS = 3
AUGMENTATIONS_RATIO = 4.

# diffusion:
DIFFUSION_SAMPLER = 'discrete'  # 'continuous'
BETA_SCHEDULER = 'cosine'  # 'linear', 'quadratic', 'sigmoid'
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 65e-4

# model:
MODEL_TYPE = 'rin' # 'unet'
NUM_TIME_EMBEDDINGS = 32
LOSS_FUNCTION = 'l2'  # 'huber' 'l1' 'l2'
ACTIVATION = 'gelu'  # 'silu' 'relu' 'selu'
POSITION_EMBEDDINGS = 'sinusoidal_learned'  # 'sinusoidal'
# Unet specific:
NUM_MODULE_LAYERS = 5
KERNEL_SIZE = 3
# RIN specific:
PATCH_SIZE = 8
PATCHES_WIDTH = 256
LATENT_WIDTH = 512  # PATCHES_WIDTH
LATENT_SELF_ATTENTION_DEPTH = 4
NUM_BLOCKS = 6
NUM_LATENTS = 128
TRAIN_PROBABILITY_SELF_CONDITIONING = 0.9

# training:
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
LIMIT_VAL_BATCHES_RATIO = 0.05
SHOW_VALIDATION_IMAGES = True
NUM_VALIDATION_IMAGES = 9
