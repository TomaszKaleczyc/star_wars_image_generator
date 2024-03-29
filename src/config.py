# experiment setup:
RANDOM_STATE = 5
VERBOSE = True
NUM_WORKERS = 4
SAVE_PATH = '../output'
SAMPLES_FOLDER_NAME = 'samples'
PRINT_STATE = False

# data:
DATASET =  'StarWars'  # 'PoC'  # 'StarWars'
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
POSITION_EMBEDDINGS = 'sinusoidal'  # 'sinusoidal_learned'
# Unet specific:
NUM_MODULE_LAYERS = 5
KERNEL_SIZE = 3
# RIN specific:
PATCH_SIZE = 8
NUM_BLOCKS = 6
FEED_FORWARD_MULTIPLIER = 4
NUM_LATENTS = 192  # 256  # 128
PATCHES_WIDTH = 256
LATENT_WIDTH = 512  # PATCHES_WIDTH
LATENT_SELF_ATTENTION_DEPTH = 4
PATCHES_SELF_ATTENTION_DEPTH = 2
ATTENTION_BLOCK_HEADS = 4
ATTENTION_HEAD_SIZE = 32
TRAIN_PROBABILITY_SELF_CONDITIONING = 0.9

# training:
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
ACCUMULATE_GRADIENT_EVERY_N_BATCHES = 64 // BATCH_SIZE
DROPOUT_PROBABILITY = 0.2
SHOW_VALIDATION_IMAGES = True
NUM_VALIDATION_IMAGES = 9
