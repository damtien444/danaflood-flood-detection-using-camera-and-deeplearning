import torch

LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_WORKERS = 2

EXPERIMENT_NAME = "UNET_WITH_RESIDUAL_classification"

# image size must be divisible by 32, because U-Net structure has 5 bi-down_sample
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"data\train"
TRAIN_MASK_DIR = r"data\train_masks"
VAL_IMG_DIR = ""
VAL_MASK_DIR = ""
TEST_PRED_FOLDER = r"C:\Users\damti\OneDrive - The University of Technology\Desktop\Study\Do an tot nghiep\WorkingSpaceDATN\contruction\artifacts\saved_images"
TEST_IMAGE_DIR = r"E:\DATN_local\0_DATASET\TEST"
TEST_MASK_DIR = r"E:\DATN_local\0_DATASET\TEST"
CLASSIFICATION_LABEL = r"E:\DATN_local\0_DATASET\level_label.json"

CHECKPOINT_PATH = r"E:\DATN_local\MODEL_CHECKPOINTS\UNET_WITH_RESIDUAL_TEST_COLLECT.pth.tar"

IS_TRAINING_CLASSIFIER = True

IS_COLAB = True
if IS_COLAB:
    TRAIN_IMG_DIR = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/TRAIN_DEV"
    TRAIN_MASK_DIR = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/TRAIN_DEV"
    TEST_PRED_FOLDER = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/contruction/artifacts/saved_images"
    TEST_IMAGE_DIR = r"/content/danaflood-flood-detection-using-camera-and-deeplearning/TEST"
    TEST_MASK_DIR = "/content/danaflood-flood-detection-using-camera-and-deeplearning/TEST"
    CHECKPOINT_PATH = r"/content/UNET_WITH_RESIDUAL_TEST_COLLECT.pth.tar"
    CLASSIFICATION_LABEL = r"/content/level_label.json"
    BATCH_SIZE = 16
