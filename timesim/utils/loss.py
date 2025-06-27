import torch
from torch import nn


def mse_loss():
    return nn.MSELoss()


def dilate_loss(*args, **kwargs):
    raise NotImplementedError("Dilate loss not yet implemented in skeleton") 