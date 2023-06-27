# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------
import torch
from .resnet import resnet18

def load_fad_model(device):
    model = resnet18(num_classes=309)
    model_path="checkpoints/pretrained/H.pth.tar"
    weight_dict = torch.load(model_path)['model_state_dict']
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if 'audnet' in name:
            name = '.'.join(name.split('.')[1:])
        model_dict[name].copy_(param)
    model.eval()
    model.to(device)
    return model
