import torch
import wandb
from torchmetrics.functional import stat_scores, accuracy, precision, recall, precision_recall_curve, confusion_matrix
import segmentation_models_pytorch as smp
from tqdm import tqdm


def share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn, model, device='cuda'):
    preds_m, pred_c = model(image.float().to(device))
    mask_loss = mask_loss_fn(preds_m, targets_m)
    cls_loss = cls_loss_fn(pred_c, targets_c)

    return mask_loss, cls_loss, preds_m, pred_c

def epoch_end(dataset_mask_stat_score, dataset_cls_stat_score, dataset_cls_acc):
    mask_tp = dataset_mask_stat_score[0]
    mask_fp = dataset_mask_stat_score[1]
    mask_fn = dataset_mask_stat_score[2]
    mask_tn = dataset_mask_stat_score[3]
    dataset_mask_iou  = smp.metrics.iou_score(mask_tp, mask_fp, mask_fn, mask_tn, reduction="micro")
    dataset_mask_f1 = smp.metrics.f1_score(mask_tp, mask_fp, mask_fn, mask_tn, reduction="micro")

    dataset_cls_acc = torch.mean(torch.FloatTensor(dataset_cls_acc))

    cls_tp = dataset_cls_stat_score[0]
    cls_fp = dataset_cls_stat_score[1]
    cls_tn = dataset_cls_stat_score[2]
    cls_fn = dataset_cls_stat_score[3]
    dataset_cls_precision = cls_tp.sum() / (cls_fp.sum() + cls_tp.sum())
    dataset_cls_recall = cls_tp.sum() / (cls_tp.sum() + cls_fn.sum())

    # [tp, fp, tn, fn]
    dataset_confusion_matrix = [cls_tp.sum().item(), cls_fp.sum().item(), cls_tn.sum().item(), cls_fn.sum().item()]

    return dataset_mask_iou.item(), dataset_mask_f1.item(), dataset_cls_acc.item(), dataset_cls_precision.item(), dataset_cls_recall.item(), dataset_confusion_matrix


def check_performance(loader, model,type, mask_loss_fn, cls_loss_fn, device='cuda', alpha=0.7):
    loop = tqdm(loader)

    dataset_mutual_losses, mask_losses, cls_losses = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, targets_m, targets_c) in enumerate(loop):
            targets_m = targets_m.float().unsqueeze(1).to(device=device, non_blocking=True)
            targets_c = targets_c.type(torch.LongTensor).to(device=device, non_blocking=True)
            mask_loss, cls_loss, pred_m, pred_c = share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn,
                                                             model)
            loss = (mask_loss * alpha + (1 - alpha) * cls_loss)

            dataset_mutual_losses.append(loss)
            mask_losses.append(mask_loss)
            cls_losses.append(cls_loss)

            if batch_idx == 0:
                list_pred_m = pred_m
                list_targets_m = targets_m
                list_pred_c = pred_c
                list_targets_c = targets_c
            else:
                list_pred_m = torch.cat((list_pred_m, pred_m), dim=0)
                list_pred_c = torch.cat((list_pred_c, pred_c), dim=0)
                list_targets_m = torch.cat((list_targets_m, targets_m), dim=0)
                list_targets_c = torch.cat((list_targets_c, targets_c), dim=0)

        acc_c = accuracy(list_pred_c, list_targets_c, num_classes=4, average="macro")
        prec_c = precision(list_pred_c, list_targets_c, num_classes=4, average="macro")
        recall_c = recall(list_pred_c, list_targets_c, num_classes=4, average="macro")
        conf_mat_c = confusion_matrix(list_pred_c, list_targets_c, num_classes=4)

        mask_stat_score = smp.metrics.get_stats(list_pred_m.long(), list_targets_m.long(), mode='binary', threshold=0.5)
        acc_m = smp.metrics.accuracy(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                     reduction='macro')
        iou_m = smp.metrics.iou_score(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                      reduction='macro')
        f1_m = smp.metrics.f1_score(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                    reduction='macro')

        dataset_mutual_losses = torch.mean(torch.FloatTensor(dataset_mutual_losses)).item()
        mask_losses           = torch.mean(torch.FloatTensor(mask_losses)).item()
        cls_losses            = torch.mean(torch.FloatTensor(cls_losses)).item()
        print("EVAL:", type)
        print(type+'_dataset_mutual_losses:', dataset_mutual_losses)
        print(type+'_mask_losses:', mask_losses)
        print(type+f"_cls_losses:", cls_losses)
        print(type+f"_dataset_mask_acc:", acc_m)
        print(type+f"_dataset_mask_iou:", iou_m)
        print(type+f"_dataset_mask_f1:", f1_m)
        print(type+f"_dataset_cls_acc:", acc_c)
        print(type+f"_dataset_cls_precision:", prec_c)
        print(type+f"_dataset_cls_recall:", recall_c)
        print(type+f"_dataset_confusion_matrix:\n", conf_mat_c)
        print("----------------------------------------")

        wandb.log({type + '_dataset_mutual_losses': dataset_mutual_losses,
                   type + '_mask_losses': mask_losses,
                   type + "_cls_losses": cls_losses,
                   })
        wandb.log({
            type + "_dataset_mask_acc": acc_m
        })

        wandb.log({
            type + "_dataset_mask_iou": iou_m
        })

        wandb.log({
            type + "_dataset_mask_f1": f1_m
        })

        wandb.log({
            type + "_dataset_cls_acc": acc_c
        })

        wandb.log({
            type + "_dataset_cls_precision": prec_c
        })

        wandb.log({
            type + "_dataset_cls_recall": recall_c
        })


    model.train()
    return dataset_mutual_losses

def train_fn(loader, model, optimizer, mask_loss_fn, cls_loss_fn, scaler, alpha=0.7, device='cuda'):
    loop = tqdm(loader)
    model.train()

    dataset_mutual_losses, mask_losses, cls_losses = [], [], []

    for batch_idx, (image, targets_m, targets_c) in enumerate(loop):

        targets_m = targets_m.float().unsqueeze(1).to(device=device, non_blocking=True)
        targets_c = targets_c.type(torch.LongTensor).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast():

            mask_loss, cls_loss, pred_m, pred_c = share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn,
                                                             model)
            loss = (mask_loss * alpha + cls_loss)

            dataset_mutual_losses.append(loss)
            mask_losses.append(mask_loss)
            cls_losses.append(cls_loss)

            if batch_idx == 0:
                list_pred_m = pred_m
                list_targets_m = targets_m
                list_pred_c = pred_c
                list_targets_c = targets_c
            else:
                list_pred_m = torch.cat((list_pred_m, pred_m), dim=0)
                list_pred_c = torch.cat((list_pred_c, pred_c), dim=0)
                list_targets_m = torch.cat((list_targets_m, targets_m), dim=0)
                list_targets_c = torch.cat((list_targets_c, targets_c), dim=0)

        # squeeze  dataset_mask_stat_score[i]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item(), loss_mask=mask_loss.item(), loss_cls=cls_loss.item())

    acc_c = accuracy(list_pred_c, list_targets_c, num_classes=4, average="macro")
    prec_c = precision(list_pred_c, list_targets_c, num_classes=4, average="macro")
    recall_c = recall(list_pred_c, list_targets_c, num_classes=4, average="macro")
    conf_mat_c = confusion_matrix(list_pred_c, list_targets_c, num_classes=4)

    mask_stat_score = smp.metrics.get_stats(list_pred_m, list_targets_m, mode='binary', threshold=0.5)
    acc_m = smp.metrics.accuracy(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                 reduction='macro')
    iou_m = smp.metrics.iou_score(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                  reduction='macro')
    f1_m = smp.metrics.f1_score(mask_stat_score[0], mask_stat_score[1], mask_stat_score[2], mask_stat_score[3],
                                reduction='macro')

    dataset_mutual_losses = torch.mean(torch.FloatTensor(dataset_mutual_losses)).item()
    mask_losses = torch.mean(torch.FloatTensor(mask_losses)).item()
    cls_losses = torch.mean(torch.FloatTensor(cls_losses)).item()
    print("EVAL:", "train")
    print("train" + '_dataset_mutual_losses:', dataset_mutual_losses)
    print("train" + '_mask_losses:', mask_losses)
    print("train" + f"_cls_losses:", cls_losses)
    print("train" + f"_dataset_mask_acc:", acc_m)
    print("train" + f"_dataset_mask_iou:", iou_m)
    print("train" + f"_dataset_mask_f1:", f1_m)
    print("train" + f"_dataset_cls_acc:", acc_c)
    print("train" + f"_dataset_cls_precision:", prec_c)
    print("train" + f"_dataset_cls_recall:", recall_c)
    print("train" + f"_dataset_confusion_matrix:\n", conf_mat_c)
    print("----------------------------------------")

    wandb.log({"train" + '_dataset_mutual_losses': dataset_mutual_losses,
               "train" + '_mask_losses': mask_losses,
               "train" + "_cls_losses": cls_losses,
               })
    wandb.log({
        "train" + "_dataset_mask_acc": acc_m
    })

    wandb.log({
        "train" + "_dataset_mask_iou": iou_m
    })

    wandb.log({
        "train" + "_dataset_mask_f1": f1_m
    })

    wandb.log({
        "train" + "_dataset_cls_acc": acc_c
    })

    wandb.log({
        "train" + "_dataset_cls_precision": prec_c
    })

    wandb.log({
        "train" + "_dataset_cls_recall": recall_c
    })


