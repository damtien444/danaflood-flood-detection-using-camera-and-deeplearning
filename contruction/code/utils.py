import torch
import torchvision
from torch.autograd import Variable

from config import CLASSIFICATION_LABEL, IS_TRAINING_CLASSIFIER
from dataset import StrFloodDataset
from torch.utils.data import DataLoader, random_split
import wandb


def save_checkpoint(state, filename='mycheckpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, strict=False):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=strict)


def get_loaders(
        train_dir,
        train_maskdir,
        batch_size,
        train_transform,
        val_dir=None,
        val_maskdir=None,
        val_transform=None,
        num_workers=4,
        pin_memory=True,
        split_ratio=0.9
):
    if val_dir is None:
        dataset = StrFloodDataset(image_dir=train_dir, mask_dir=train_maskdir, file_label=CLASSIFICATION_LABEL,
                                  transform=train_transform)
        train_ds, val_ds = random_split(dataset, [int(len(dataset) * split_ratio),
                                                  int(len(dataset) * (1 - split_ratio) + 1)])
    else:
        train_ds = StrFloodDataset(image_dir=train_dir, mask_dir=train_maskdir, file_label=CLASSIFICATION_LABEL,
                                   transform=train_transform)
        val_ds = StrFloodDataset(image_dir=val_dir, mask_dir=val_maskdir, file_label=CLASSIFICATION_LABEL,
                                 transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)

    return train_loader, val_loader


def get_test_loader(
        test_dir,
        test_maskdir,
        batch_size,
        test_transform,
        num_workers=4,
        pin_memory=True,
):
    test_ds = StrFloodDataset(image_dir=test_dir, mask_dir=test_maskdir, file_label=CLASSIFICATION_LABEL,
                              transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                             shuffle=False)
    return test_loader


def check_accuracy(loader, model, type, loss_fn, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    accs_c = 0
    losses = 0
    model.eval()

    with torch.no_grad():
        for data, target_m, target_c in loader:
            data = data.to(device)
            target_m = target_m.to(device).unsqueeze(1)
            target_c = Variable(target_c.to(device))

            preds = model(data)
            preds_m = torch.sigmoid(preds[0])
            preds_c = preds[1]

            if IS_TRAINING_CLASSIFIER:
                loss = loss_fn(preds_c, target_c)
            else:
                loss = loss_fn(preds_m, target_m)

            arg_maxs = torch.argmax(preds_c)
            num_correct_c = torch.sum(arg_maxs == target_c)
            accs_c = num_correct_c / float(len(data))

            preds_m = (preds_m > 0.5).float()

            num_correct += (preds_m == target_m).sum()
            num_pixels += torch.numel(preds_m)
            dice_score += (2 * (preds_m * target_m).sum()) / (
                    (preds_m + target_m).sum() + 1e-8
            )

            losses += loss

    print(f'{type}: Got mask acc {num_correct / num_pixels * 100:.2f}')
    print(f'{type}: Got class acc {100 * accs_c / len(loader):.2f}')
    print(f'{type}: Got {"class" if IS_TRAINING_CLASSIFIER else "mask"} loss {losses / len(loader):.2f}')
    print(f'{type}: Dice score: {dice_score / len(loader)}')

    wandb.log({f'mask_acc_{type}': num_correct / num_pixels * 100})
    wandb.log({f'{"class" if IS_TRAINING_CLASSIFIER else "mask"}_loss_{type}': losses / len(loader)})
    wandb.log({f'class_acc_{type}': 100 * accs_c / len(loader)})
    wandb.log({f'dice_score_{type}': dice_score / len(loader)})
    model.train()

    return num_correct / num_pixels if not IS_TRAINING_CLASSIFIER else accs_c / len(loader)


def check_dev_accuracy(loader, model, loss_fn, device='cuda'):
    return check_accuracy(loader, model, "dev", loss_fn, device)


def check_test_accuracy(loader, model, loss_fn, device='cuda'):
    return check_accuracy(loader, model, 'test', loss_fn, device)


def save_predictions_as_imgs(
        loader, model, experiment_name, folder="saved_images/", device="cuda", type='train'
):
    model.eval()
    for idx, (x, y, z) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds_m, preds_c = model(x)
            preds_m = torch.sigmoid(preds_m)
            preds_m = (preds_m > 0.5).float()
        # if type != "train":
        torchvision.utils.save_image(x, f"{folder}/groundtruth_{experiment_name}_{type}_{idx}.png")
        torchvision.utils.save_image(preds_m, f"{folder}/pred_{experiment_name}_{type}_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{experiment_name}_{type}_{idx}.png")

    model.train()
