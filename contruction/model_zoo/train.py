import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from utils import save_checkpoint
from dataloader import Dataset
from augmentation import get_training_augmentation, get_validation_augmentation
from train_step import train_fn, check_performance

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['flood']
# ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'


IS_COLAB = False
ROOT_FOLDER = r"E:/DATN_local"
BATCH_SIZE = 2
loss_fusion_coefficient = 0.7
NUM_EPOCHS = 3

version = 1
EXPERIMENT_NAME = ENCODER+"_"+ENCODER_WEIGHTS+"_"+str(version)

if IS_COLAB:
    ROOT_FOLDER = "/content"
    DRIVE_OUTPUT_FOLDER = "/content/drive/MyDrive/DAMQUANGTIEN_DATN_SPACE/EXPERIMENT_OUTPUT/"+EXPERIMENT_NAME
    DRIVE_CHECKPOINTS_OUTPUT = "/content/drive/MyDrive/DAMQUANGTIEN_DATN_SPACE/CHECKPOINTS_OUTPUT"
    if not os.path.exists(DRIVE_OUTPUT_FOLDER):
        os.makedirs(DRIVE_OUTPUT_FOLDER)
    BATCH_SIZE = 4
else:
    DRIVE_OUTPUT_FOLDER = None
    DRIVE_CHECKPOINTS_OUTPUT = None

DATA_DIR = ROOT_FOLDER + r"/1_IN_USED_DATASET"


x_train_dir = os.path.join(DATA_DIR, 'TRAIN_DEV')
y_train_dir = os.path.join(DATA_DIR, 'TRAIN_DEV_MASK')

x_valid_dir = os.path.join(DATA_DIR, 'VAL')
y_valid_dir = os.path.join(DATA_DIR, 'VAL_MASK')

x_test_dir = os.path.join(DATA_DIR, 'TEST')
y_test_dir = os.path.join(DATA_DIR, 'TEST_MASK')

file_label = DATA_DIR + r"\level_label.json"


CHECKPOINT_OUTPUT_PATH = ROOT_FOLDER + r"/MODEL_CHECKPOINTS/" + EXPERIMENT_NAME + ".pth.tar"

LEARNING_RATE = 0.0001
ENCODER_LEARNING_RATE = 1e-5
DECODER_LEARNING_RATE = 1e-4


if __name__ == "__main__":
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        file_label=file_label,
        augmentation=get_training_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        file_label=file_label,
        augmentation=get_validation_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        file_label=file_label,
        augmentation=get_validation_augmentation(),
        # preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        activation=None,      # activation function, default is None
        classes=4,                 # define number of output labels
    )
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=1, aux_params=aux_params)
    model.to(DEVICE)

    mask_loss_fn = smp.losses.DiceLoss(mode="binary")
    cls_loss_fn = nn.CrossEntropyLoss()


    optimizer = optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LEARNING_RATE},
            {"params": model.decoder.parameters(), "lr": DECODER_LEARNING_RATE},
            {"params": model.classification_head.parameters()}
        ],
        lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()

    best_perform = 1000
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, mask_loss_fn, cls_loss_fn, scaler, alpha=loss_fusion_coefficient)
        val_mutual_loss = check_performance(valid_loader, model, "val", mask_loss_fn, cls_loss_fn, device=DEVICE, alpha=loss_fusion_coefficient)
        test_mutual_loss = check_performance(test_loader, model, "test", mask_loss_fn, cls_loss_fn, device=DEVICE, alpha=loss_fusion_coefficient)

        if best_perform > test_mutual_loss:
            best_perform = test_mutual_loss
            checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            save_checkpoint(checkpoint, filename=CHECKPOINT_OUTPUT_PATH)

            if IS_COLAB:
                os.system(f"cp {CHECKPOINT_OUTPUT_PATH} {DRIVE_CHECKPOINTS_OUTPUT}")