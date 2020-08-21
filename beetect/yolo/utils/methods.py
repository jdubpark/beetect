import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = torch.tensor(lam) # lam.float64()

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if use_cuda:
        lam = lam.cuda()
        index = index.cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    print(lam, criterion(pred, y_a))
    # return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    crit_a = criterion(pred, y_a)
    crit_b = criterion(pred, y_b)
    losses = []
    for loss_a, loss_b in zip(crit_a, crit_b):
        losses.append(lam * loss_a + (1. - lam) * loss_b)
    return tuple(losses)
