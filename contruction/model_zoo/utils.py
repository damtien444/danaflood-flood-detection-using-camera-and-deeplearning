import torch


def save_checkpoint(state, filename='mycheckpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, strict=False):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"], strict=strict)