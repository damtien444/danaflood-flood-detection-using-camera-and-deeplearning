import torch
import torchvision
from dataset import StrFloodDataset
from torch.utils.data import DataLoader, random_split
import wandb

def save_checkpoint(state, filename='mycheckpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


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
        dataset = StrFloodDataset(train_dir, train_maskdir, transform=train_transform)
        train_ds, val_ds = random_split(dataset, [int(len(dataset) * split_ratio),
                                                  int(len(dataset) * (1 - split_ratio) + 1)])
    else:
        train_ds = StrFloodDataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
        val_ds = StrFloodDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)

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
    test_ds = StrFloodDataset(image_dir=test_dir, mask_dir=test_maskdir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return test_loader

def check_accuracy(loader, model, type, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(f'{type}: Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}')
    print(f'{type}: Dice score: {dice_score / len(loader)}')
    wandb.log({f'acc_{type}': num_correct / num_pixels * 100})
    wandb.log({f'dice_score_{type}': dice_score / len(loader)})
    model.train()

    return num_correct/num_pixels

def check_train_accuracy(loader, model, device='cuda'):
    return check_accuracy(loader, model, "train", device)

def check_test_accuracy(loader, model, device='cuda'):
    return check_accuracy(loader, model, 'test', device)

def save_predictions_as_imgs(
        loader, model, experiment_name, folder="saved_images/", device="cuda", type='train'
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        # if type != "train":
        torchvision.utils.save_image(x, f"{folder}/groundtruth_{experiment_name}_{type}_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{experiment_name}_{type}_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/target_{experiment_name}_{type}_{idx}.png")

    model.train()
