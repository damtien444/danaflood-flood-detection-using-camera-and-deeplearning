from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as T

from config import CLASSIFICATION_LABEL, IS_TRAINING_CLASSIFIER, DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH
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
    losses = 0
    model.eval()
    metric_monitor = MetricMonitor()
    with torch.no_grad():

        for batch_idx, (data, target_m, target_c) in enumerate(loader):
            data = data.to(device)
            target_m = target_m.to(device).unsqueeze(1)
            target_c = Variable(target_c.to(device))

            preds = model(data)
            preds_m = torch.sigmoid(preds[0])
            preds_c = preds[1]

            if IS_TRAINING_CLASSIFIER:
                loss = loss_fn(preds_c, target_c)
                if batch_idx == 0:
                    list_embed_vector = preds_c
                    list_labels = target_c
                else:
                    list_embed_vector = torch.cat((list_embed_vector, preds_c), dim=0)
                    list_labels = torch.cat((list_labels, target_c), dim=0)
            else:
                loss = loss_fn(preds_m, target_m)


            preds_m = (preds_m > 0.5).float()

            num_correct += (preds_m == target_m).sum()
            num_pixels += torch.numel(preds_m)
            dice_score += (2 * (preds_m * target_m).sum()) / (
                    (preds_m + target_m).sum() + 1e-8
            )
            losses += loss.item()

        # preds_batch_c = torch.argmax(list_embed_vector, dim=1)
        if IS_TRAINING_CLASSIFIER:
            class_accuracy = calculate_classification_accuracy(list_embed_vector, list_labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", class_accuracy)

    print(f'{type}: Got mask acc {num_correct / num_pixels * 100:.2f}')
    print(f'{type}: Got {"class" if IS_TRAINING_CLASSIFIER else "mask"} loss {losses / len(loader):.2f}')
    print(f'{type}: Dice score: {dice_score / len(loader)}')

    wandb.log({f'mask_acc_{type}': num_correct / num_pixels * 100})
    wandb.log({f'{"class" if IS_TRAINING_CLASSIFIER else "mask"}_loss_{type}': losses / len(loader)})
    wandb.log({f'dice_score_{type}': dice_score / len(loader)})

    if IS_TRAINING_CLASSIFIER:
        print(f'{type}: Got class acc {class_accuracy:.2f}')
        wandb.log({f'class_acc_{type}': class_accuracy})

    model.train()

    return num_correct / num_pixels if not IS_TRAINING_CLASSIFIER else losses / len(loader)


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


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": .0, "count": .0, "avg": .0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def calculate_classification_accuracy(output, target):
    # output = torch.sigmoid(output) >= 0.5
    # print(output, target)
    output = torch.argmax(output, dim=1)
    target = target
    # print(output, target)
    return torch.true_divide((target == output).sum(), output.size(0)).item()


def draw_ROC_ConfusionMatrix_PE(model, test_data_loader, labels_class):
    list_embed_vector = torch.tensor([])
    list_labels = torch.tensor([])

    with torch.no_grad():
        model.eval()
        for batch_idx, datasets in enumerate(test_data_loader):
            images, masks, labels = datasets

            images = Variable(images.to(DEVICE))
            labels = Variable(labels.to(DEVICE))
            _mask, embed_feat = model(images)
            if batch_idx == 0:
                list_embed_vector = embed_feat
                list_labels = labels
            else:
                list_embed_vector = torch.cat((list_embed_vector, embed_feat), dim=0)
                list_labels = torch.cat((list_labels, labels), dim=0)
        preds = torch.argmax(list_embed_vector, dim=1)
        if DEVICE == "cuda":
            wandb.log({"ROC_test": wandb.plot.roc_curve(list_labels.data.cpu(), list_embed_vector.data.cpu(),
                                                        labels=labels_class),
                       "PR_test": wandb.plot.pr_curve(list_labels.data.cpu(), list_embed_vector.data.cpu(),
                                                      labels=labels_class,
                                                      classes_to_plot=None),
                       "Conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                               y_true=list_labels.data.cpu().detach().numpy(),
                                                               preds=preds.cpu().detach().numpy(),
                                                               class_names=labels_class)})
        else:
            wandb.log(
                {"ROC_test": wandb.plot.roc_curve(list_labels.data, list_embed_vector.data, labels=labels_class),
                 "PR_test": wandb.plot.pr_curve(list_labels.data, list_embed_vector.data, labels=labels_class,
                                                classes_to_plot=None),
                 "Conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=list_labels.data.detach().numpy(),
                                                         preds=preds.detach().numpy(),
                                                         class_names=labels_class)})

def otsu_thresholding(image):
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)
    return image_result

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def otsu_thresholding(image):
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return otsu_threshold


def auto_canny(image, o_threshold=0.2):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    upper = o_threshold
    lower = upper / 2
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def canny_preprocess(img, debug=True):
    _preprocessing = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        lambda x: np.array(x).astype(np.uint8),
        lambda x: auto_canny(x, otsu_thresholding(x)),
        lambda x: cv2.dilate(x, np.ones((3, 3), np.uint8), iterations=1),
        #         lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)),
        lambda x: cv2.bitwise_not(x),
        lambda x: cv2.cvtColor(x,cv2.COLOR_GRAY2RGB),
    ])

#     _postprocessing = T.Compose([
#         T.ToTensor(),
#         T.Resize((512, 512)),
#         T.Normalize(
#            mean=[0.485, 0.456, 0.406],
#            std=[0.229, 0.224, 0.225]
#        ),
#         T.ToPILImage(),
#     ])

    canny_mask = _preprocessing(img)
    applied_mask = cv2.bitwise_and(img, canny_mask)
#     result = _postprocessing(applied_mask)
    return applied_mask


