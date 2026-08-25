from torch import nn

from .dilate import dilate_loss


def mse_loss():
    return nn.MSELoss()


__all__ = ["mse_loss", "dilate_loss"]
