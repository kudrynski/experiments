import torch.nn
from ssd.ssd_PyT.src import model


def nvidia_ssd_pyt(pretrained=False, *args, **kwargs):
    return model.SSD300
