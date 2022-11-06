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
    OUTPUT_FOLDER, IS_TRAINING_CLASSIFIER, CHECKPOINT_INPUT_PATH, CHECKPOINT_OUTPUT_PATH, DRIVE_OUTPUT_FOLDER, \
    DRIVE_CHECKPOINTS_OUTPUT, LOAD_OPTIMIZER
from model import UNET
from datetime import datetime
import wandb

from utils import load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, \
    get_test_loader, check_dev_accuracy, check_test_accuracy, draw_ROC_ConfusionMatrix_PE, \
    calculate_classification_accuracy, MetricMonitor


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # use tqdm for progress bar
    loop = tqdm(loader)
    model.train()
    metric_monitor = MetricMonitor()
    for batch_idx, (data, targets_m, targets_c) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets_m = targets_m.float().unsqueeze(1).to(device=DEVICE, non_blocking=True)
        targets_c = targets_c.type(torch.LongTensor).to(device=DEVICE, non_blocking=True)

        num_correct = 0
        num_pixels = 0
        dice_score = 0
        accs_c = 0

        # forward
        with torch.cuda.amp.autocast():

            predictions_m, predictions_c = model(data)

            preds_m = (predictions_m > 0.5).float()
            num_correct += (preds_m == targets_m).sum()
            num_pixels += torch.numel(preds_m)
            dice_score += (2 * (preds_m * targets_m).sum()) / (
                    (preds_m + targets_m).sum() + 1e-10
            )

            if IS_TRAINING_CLASSIFIER:
                loss = loss_fn(predictions_c, targets_c)
                if batch_idx == 0:
                    list_embed_vector = predictions_c
                    list_labels = targets_c
                else:
                    list_embed_vector = torch.cat((list_embed_vector, predictions_c), dim=0)
                    list_labels = torch.cat((list_labels, targets_c), dim=0)
            else:
                loss = loss_fn(predictions_m, targets_m)
            metric_monitor.update("Loss", loss.item())


        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        wandb.log({f"{'class' if IS_TRAINING_CLASSIFIER else 'mask'}_loss_train": metric_monitor.metrics['Loss']['avg']})
        wandb.log({f'mask_acc_train': num_correct / num_pixels * 100})
        wandb.log({f'dice_score_train': dice_score / len(data)})

    # preds_batch_c = torch.argmax(list_embed_vector, dim=1)
    if IS_TRAINING_CLASSIFIER:
        class_accuracy = calculate_classification_accuracy(list_embed_vector, list_labels)
        metric_monitor.update("Accuracy", class_accuracy)
        wandb.log({f'class_acc_train': class_accuracy})


def main():

    wandb.init(project="UNET_FLOOD", entity="damtien440")

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0, 0, 0],
                std = [1, 1, 1],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLoss()
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
        load_checkpoint(torch.load(CHECKPOINT_INPUT_PATH), model, strict=False)
        if LOAD_OPTIMIZER:
            optimizer.load_state_dict(torch.load(CHECKPOINT_INPUT_PATH)["optimizer"])

        if IS_TRAINING_CLASSIFIER:
            loss_fn = nn.CrossEntropyLoss()
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True

    scaler = torch.cuda.amp.GradScaler()
    best_perform = 10000
    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        acc = check_dev_accuracy(val_loader, model, loss_fn=loss_fn, device=DEVICE)
        test_loss = check_test_accuracy(test_loader, model, loss_fn=loss_fn, device=DEVICE)
        draw_ROC_ConfusionMatrix_PE(model, test_loader, [0,1,2,3])
        save_predictions_as_imgs(val_loader, model, EXPERIMENT_NAME, folder=OUTPUT_FOLDER, device=DEVICE, type='train')


        if best_perform > test_loss:
            best_perform = test_loss

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(checkpoint, filename=CHECKPOINT_OUTPUT_PATH)
            if IS_COLAB:
                os.system(f"cp {CHECKPOINT_OUTPUT_PATH} {DRIVE_CHECKPOINTS_OUTPUT}")
                os.system(f"zip -r {EXPERIMENT_NAME}.zip {OUTPUT_FOLDER}")
                os.system(f"cp -a {EXPERIMENT_NAME}.zip {DRIVE_OUTPUT_FOLDER}")
        # print example

    # save_predictions_as_imgs(test_loader, model,  EXPERIMENT_NAME, folder=TEST_PRED_FOLDER, device=DEVICE, type='test')

if __name__ == "__main__":
    main()


