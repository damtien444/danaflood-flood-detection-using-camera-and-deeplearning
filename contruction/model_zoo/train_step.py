import torch
from torchmetrics.functional import stat_scores, accuracy, precision, recall, precision_recall_curve
import segmentation_models_pytorch as smp
from tqdm import tqdm


def share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn, model, device='cuda'):
    preds_m, pred_c = model(image.float().to(device))
    mask_loss = mask_loss_fn(preds_m, targets_m)
    cls_loss = cls_loss_fn(pred_c, targets_c)

    # class_stat_score: [tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``)
    # shape: [batch_size, 5]
    cls_stat_score = stat_scores(pred_c, targets_c, reduce='macro', num_classes=4)

    # mask_stat_score: true_positive, false_positive, false_negative, true_negative tensors (N, C) shape each.
    # shape: list of [[batch_size, 1], [batch_size, 1], [batch_size, 1], [batch_size, 1]]
    mask_stat_score = smp.metrics.get_stats(preds_m.long(), targets_m.long(), mode='binary')

    # cls_metrics -> accuracy, confusion matrix -> stat, precision, recall
    output = torch.argmax(pred_c, dim=1)

    # float
    cls_acc = torch.true_divide((targets_c == output).sum(), output.size(0)).item()
    return mask_loss, cls_loss, mask_stat_score, cls_stat_score, cls_acc

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
    print("EVAL:", type)
    dataset_mask_stat_score, dataset_cls_stat_score, dataset_cls_acc, dataset_mutual_losses, mask_losses, cls_losses = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, targets_m, targets_c) in enumerate(loop):
            targets_m = targets_m.float().unsqueeze(1).to(device=device, non_blocking=True)
            targets_c = targets_c.type(torch.LongTensor).to(device=device, non_blocking=True)
            mask_loss, cls_loss, mask_stat_score, cls_stat_score, cls_acc = share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn, model)
            loss = (mask_loss*alpha + (1-alpha)*cls_loss)

            dataset_mutual_losses.append(loss)
            mask_losses.append(mask_loss)
            cls_losses.append(cls_loss)

            dataset_cls_acc.append(cls_acc)
            if len(dataset_mask_stat_score) <= 0:
                dataset_mask_stat_score = list(mask_stat_score)
                for i in range(5):
                    dataset_cls_stat_score.append(cls_stat_score[:,i])
            else:
                for i in range(len(mask_stat_score)):
                    dataset_mask_stat_score[i] = torch.cat((dataset_mask_stat_score[i], mask_stat_score[i]), dim=0)

                for i in range(5):
                    dataset_cls_stat_score[i] = torch.cat([dataset_cls_stat_score[i], cls_stat_score[:,0]], dim=0)
        for i in range(len(dataset_mask_stat_score)):
            dataset_mask_stat_score[i] = dataset_mask_stat_score[i].squeeze()

        for i in range(len(dataset_cls_stat_score)):
            dataset_cls_stat_score[i] = dataset_cls_stat_score[i].squeeze()

        dataset_mask_iou, dataset_mask_f1, dataset_cls_acc, dataset_cls_precision, dataset_cls_recall, dataset_confusion_matrix = epoch_end(dataset_mask_stat_score, dataset_cls_stat_score, dataset_cls_acc)

        dataset_mutual_losses = torch.mean(torch.FloatTensor(dataset_mutual_losses)).item()
        mask_losses           = torch.mean(torch.FloatTensor(mask_losses)).item()
        cls_losses            = torch.mean(torch.FloatTensor(cls_losses)).item()

        print(type+'_dataset_mutual_losses:', dataset_mutual_losses)
        print(type+'_mask_losses:', mask_losses)
        print(type+f"_cls_losses:", cls_losses)
        print(type+f"_dataset_mask_iou:", dataset_mask_iou)
        print(type+f"_dataset_mask_f1:", dataset_mask_f1)
        print(type+f"_dataset_cls_acc:", dataset_cls_acc)
        print(type+f"_dataset_cls_precision:", dataset_cls_precision)
        print(type+f"_dataset_cls_recall:", dataset_cls_recall)
        print(type+f"_dataset_confusion_matrix:", dataset_confusion_matrix)
    model.train()
    return dataset_mutual_losses

def train_fn(loader, model, optimizer, mask_loss_fn, cls_loss_fn, scaler, alpha=0.7, device='cuda'):
    loop = tqdm(loader)
    model.train()

    dataset_mask_stat_score, dataset_cls_stat_score, dataset_cls_acc, dataset_mutual_losses, mask_losses, cls_losses = [], [], [], [], [], []

    for batch_idx, (image, targets_m, targets_c) in enumerate(loop):

        targets_m = targets_m.float().unsqueeze(1).to(device=device, non_blocking=True)
        targets_c = targets_c.type(torch.LongTensor).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast():

            mask_loss, cls_loss, mask_stat_score, cls_stat_score, cls_acc = share_step(image, targets_m, targets_c, mask_loss_fn, cls_loss_fn, model)
            loss = (mask_loss*alpha + (1-alpha)*cls_loss)

            dataset_mutual_losses.append(loss)
            mask_losses.append(mask_loss)
            cls_losses.append(cls_loss)

            dataset_cls_acc.append(cls_acc)
            if len(dataset_mask_stat_score) <= 0:
                dataset_mask_stat_score = list(mask_stat_score)
                for i in range(5):
                    dataset_cls_stat_score.append(cls_stat_score[:,i])
            else:
                for i in range(len(mask_stat_score)):
                    dataset_mask_stat_score[i] = torch.cat((dataset_mask_stat_score[i], mask_stat_score[i]), dim=0)

                for i in range(5):
                    dataset_cls_stat_score[i] = torch.cat([dataset_cls_stat_score[i], cls_stat_score[:,0]], dim=0)

        # squeeze  dataset_mask_stat_score[i]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item(), loss_mask=mask_loss.item(), loss_cls=cls_loss.item())

    for i in range(len(dataset_mask_stat_score)):
        dataset_mask_stat_score[i] = dataset_mask_stat_score[i].squeeze()

    for i in range(len(dataset_cls_stat_score)):
        dataset_cls_stat_score[i] = dataset_cls_stat_score[i].squeeze()

    dataset_mask_iou, dataset_mask_f1, dataset_cls_acc, dataset_cls_precision, dataset_cls_recall, dataset_confusion_matrix = epoch_end(dataset_mask_stat_score, dataset_cls_stat_score, dataset_cls_acc)

    dataset_mutual_losses = torch.mean(torch.FloatTensor(dataset_mutual_losses)).item()
    mask_losses           = torch.mean(torch.FloatTensor(mask_losses)).item()
    cls_losses            = torch.mean(torch.FloatTensor(cls_losses)).item()

    print('dataset_mutual_losses:', dataset_mutual_losses)
    print('mask_losses:', mask_losses)
    print("cls_losses:", cls_losses)
    print("dataset_mask_iou:", dataset_mask_iou)
    print("dataset_mask_f1:", dataset_mask_f1)
    print("dataset_cls_acc:", dataset_cls_acc)
    print("dataset_cls_precision:", dataset_cls_precision)
    print("dataset_cls_recall:", dataset_cls_recall)
    print("dataset_confusion_matrix:", dataset_confusion_matrix)