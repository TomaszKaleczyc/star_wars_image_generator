# experiment setup:
RANDOM_STATE = 5
VERBOSE = True
NUM_WORKERS = 8
SAVE_PATH = '../output'

# data:
DATA_DIR = "../data/star-wars-images/"
IMG_SIZE = 64
BATCH_SIZE = 32
IMAGE_CHANNELS = 3

# diffusion:
BETA_SCHEDULER = 'cosine'  # 'linear', 'quadratic', 'sigmoid'
TIMESTEPS = 300
BETA_START = 1e-4
BETA_END = 2e-2

# model:
NUM_MODULE_LAYERS = 5
NUM_TIME_EMBEDDINGS = 32
KERNEL_SIZE = 3
LOSS_FUNCTION = 'huber'  #'l1', 'l2'

# training:
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LIMIT_VAL_BATCHES_RATIO = 0.05
SHOW_VALIDATION_IMAGES = True
NUM_VALIDATION_IMAGES = 5
