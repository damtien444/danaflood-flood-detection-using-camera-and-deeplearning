import torch
import os

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_WORKERS = 2

EXPERIMENT_NAME = "UNET_WITH_RESIDUAL_CLASSIFICATION"

CHECK_POINT_MASK_NAME = "UNET_WITH_RESIDUAL_DICE_LOSS.pth.tar"
CHECK_POINT_MASK_NAME = "UNET_WITH_RESIDUAL_CLASSIFICATION.pth.tar"

ROOT_FOLDER = r"E:/DATN_local"

IS_COLAB = True
if IS_COLAB:
    ROOT_FOLDER = "/content"
    DRIVE_OUTPUT_FOLDER = "/content/drive/MyDrive/DAMQUANGTIEN_SPACE/EXPERIMENT_OUTPUT/"+EXPERIMENT_NAME
    DRIVE_CHECKPOINTS_OUTPUT = "/content/drive/MyDrive/DAMQUANGTIEN_SPACE/CHECKPOINTS_OUTPUT"
    if not os.path.exists(DRIVE_OUTPUT_FOLDER):
        os.makedirs(DRIVE_OUTPUT_FOLDER)
    BATCH_SIZE = 16
else:
    DRIVE_OUTPUT_FOLDER = None
    DRIVE_CHECKPOINTS_OUTPUT = None

DATASET = ROOT_FOLDER + r"/1_IN_USED_DATASET"


# image size must be divisible by 32, because U-Net structure has 5 bi-down_sample
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

PIN_MEMORY = True
LOAD_MODEL = True
LOAD_OPTIMIZER = True

TRAIN_IMG_DIR = DATASET + r"/TRAIN_DEV"
TRAIN_MASK_DIR = DATASET + r"/TRAIN_DEV_MASK"

# USE random split so dont need to input this
VAL_IMG_DIR = ""
VAL_MASK_DIR = ""

TEST_IMAGE_DIR = DATASET + r"/TEST"
TEST_MASK_DIR = DATASET + r"/TEST_MASK"

OUTPUT_FOLDER = ROOT_FOLDER + "/1_EXPERIMENT_OUTPUT/" + EXPERIMENT_NAME

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


CLASSIFICATION_LABEL = DATASET + r"/level_label.json"

CHECKPOINT_INPUT_PATH = ROOT_FOLDER + r"/MODEL_CHECKPOINTS/" + CHECK_POINT_MASK_NAME
CHECKPOINT_OUTPUT_PATH = ROOT_FOLDER + r"/MODEL_CHECKPOINTS/" + EXPERIMENT_NAME + ".pth.tar"

IS_TRAINING_CLASSIFIER = True


