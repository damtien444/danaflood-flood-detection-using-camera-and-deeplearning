import torch

from utils import get_test_loader, load_checkpoint, save_predictions_as_imgs, check_dev_accuracy
from model import UNET
from config import DEVICE, TEST_IMAGE_DIR, TEST_MASK_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, IMAGE_HEIGHT, \
    IMAGE_WIDTH, EXPERIMENT_NAME, OUTPUT_FOLDER, CHECKPOINT_INPUT_PATH
import albumentations as A
from albumentations.pytorch import ToTensorV2


if __name__ == "__main__":
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_loader = get_test_loader(
        test_dir=TEST_IMAGE_DIR,
        test_maskdir=TEST_MASK_DIR,
        batch_size=BATCH_SIZE,
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # check_point_path = r"E:\DATN_local\MODEL_CHECKPOINTS\UNET_WITH_RESIDUAL_TEST_COLLECT.pth.tar"
    check_point_path = CHECKPOINT_INPUT_PATH

    load_checkpoint(torch.load(check_point_path), model)

    acc = check_dev_accuracy(test_loader, model, device=DEVICE)
    print(acc)
    # save_predictions_as_imgs(test_loader, model,  EXPERIMENT_NAME, folder=TEST_PRED_FOLDER, device=DEVICE, type='test')