import os

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from code.loss import DiceBCELoss
from config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE, TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, \
    NUM_WORKERS, PIN_MEMORY, LOAD_MODEL, NUM_EPOCHS, is_colab
from model import UNET
from datetime import datetime
import wandb

from utils import load_checkpoint, save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs

# Hyperparameter etc.




def train_fn(loader, model, optimizer, loss_fn, scaler):
    # use tqdm for progress bar
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            # preds = torch.sigmoid(predictions)
            # preds = (preds > 0.5).float()

            loss = loss_fn(predictions, targets)

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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceBCELoss()
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

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    best_perform = 0
    for epoch in range(NUM_EPOCHS):


        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        now = datetime.now()
        # save_checkpoint(checkpoint, filename=f'my_checkpoint_{now.strftime("%Y_%m_%d_%H_%M_%S")}')


        # check acc
        acc = check_accuracy(val_loader, model, device=DEVICE)
        if best_perform < acc:
            best_perform = acc
            save_checkpoint(checkpoint, filename=f'my_checkpoint.pth.tar')
            if is_colab:
                os.system("cp /content/danaflood-flood-detection-using-camera-and-deeplearning/my_checkpoint.pth.tar /content/drive/MyDrive")

        # print example
        save_predictions_as_imgs(val_loader, model, folder=r"", device=DEVICE)


if __name__ == "__main__":
    main()


