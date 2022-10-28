import torch

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2

EXPERIMENT_NAME = "UNET_WITH_RESIDUAL"

# image size must be divisible by 32, because U-Net structure has 5 bi-down_sample
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"data\train"
TRAIN_MASK_DIR = r"data\train_masks"
VAL_IMG_DIR = ""
VAL_MASK_DIR = ""
TEST_PRED_FOLDER = r"artifacts\saved_images"

is_colab = True
if is_colab:
    TRAIN_IMG_DIR = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/TRAIN_DEV"
    TRAIN_MASK_DIR = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/TRAIN_DEV"
    TEST_PRED_FOLDER = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/contruction/artifacts/saved_images"
    BATCH_SIZE = 32