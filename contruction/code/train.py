import os

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from loss import DiceBCELoss, DiceLoss, TverskyLoss
from config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, \
    NUM_WORKERS, PIN_MEMORY, LOAD_MODEL, NUM_EPOCHS, IS_COLAB, EXPERIMENT_NAME, TEST_IMAGE_DIR, TEST_MASK_DIR, \
    TEST_PRED_FOLDER, IS_TRAINING_CLASSIFIER, CHECKPOINT_PATH
from model import UNET
from datetime import datetime
import wandb

from utils import load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, \
    get_test_loader, check_train_accuracy, check_test_accuracy


# Hyperparameter etc.




def train_fn(loader, model, optimizer, loss_fn, scaler):
    # use tqdm for progress bar
    loop = tqdm(loader)

    for batch_idx, (data, targets_m, targets_c) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets_m = targets_m.float().unsqueeze(1).to(device=DEVICE)
        targets_c = targets_c.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions_m, predictions_c = model(data)

            if IS_TRAINING_CLASSIFIER:
                loss = loss_fn(predictions_c, targets_c)
            else:
                loss = loss_fn(predictions_m, targets_m)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        wandb.log({"loss": loss})


def main():

    wandb.init(project="UNET_FLOOD", entity="damtien440")

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean=[0.,0.,0.],
                std=[1.,1.,1.],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0., 0., 0.],
                std=[1., 1., 1.],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = DiceLoss()
    loss_fn = TverskyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    wandb.watch(model, optimizer, log="all")

    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = get_test_loader(
        test_dir=TEST_IMAGE_DIR,
        test_maskdir=TEST_MASK_DIR,
        batch_size=BATCH_SIZE,
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH), model, strict=False)

        if IS_TRAINING_CLASSIFIER:
            loss_fn = nn.CrossEntropyLoss()
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

    scaler = torch.cuda.amp.GradScaler()
    best_perform = 0
    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # save_checkpoint(checkpoint, filename=f'my_checkpoint_{now.strftime("%Y_%m_%d_%H_%M_%S")}')

        # check acc
        acc = check_train_accuracy(val_loader, model, device=DEVICE)
        test_acc = check_test_accuracy(test_loader, model, device=DEVICE)

        if best_perform < test_acc:
            best_perform = test_acc
            save_checkpoint(checkpoint, filename=f'{EXPERIMENT_NAME}.pth.tar')
            if IS_COLAB:
                os.system(f"cp /content/danaflood-flood-detection-using-camera-and-deeplearning/{EXPERIMENT_NAME}.pth.tar /content/drive/MyDrive")
                os.system(f"cp -a /content/danaflood-flood-detection-using-camera-and-deeplearning/contruction/artifacts/saved_images /content/drive/MyDrive/EXPERIMENT_EXAMPLES")
        # print example
        save_predictions_as_imgs(val_loader, model, EXPERIMENT_NAME, folder=TEST_PRED_FOLDER, device=DEVICE, type='train')

    # save_predictions_as_imgs(test_loader, model,  EXPERIMENT_NAME, folder=TEST_PRED_FOLDER, device=DEVICE, type='test')



if __name__ == "__main__":
    main()


