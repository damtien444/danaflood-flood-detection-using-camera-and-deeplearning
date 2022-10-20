import torch

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"data\train"
TRAIN_MASK_DIR = r"data\train_masks"
VAL_IMG_DIR = ""
VAL_MASK_DIR = ""
TEST_PRED_FOLDER = r"artifacts\saved_images"