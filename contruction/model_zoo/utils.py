import torch
import wandb
from torch.autograd import Variable


def save_checkpoint(state, filename='mycheckpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, strict=False):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=strict)

def draw_ROC_ConfusionMatrix_PE(model, test_data_loader, labels_class, DEVICE):
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