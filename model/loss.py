import torch.nn as nn


def bce_loss(output, target):
    loss = nn.BCELoss()
    return loss(output, target)


def mse_loss(output, target):
    loss = nn.MSELoss()
    return loss(output, target)
