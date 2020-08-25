import json
import logging
import os
import socket

import numpy as np
import torch
import pandas as pd


def collater_effnet(data):
    #print(data)
    data = [item for item in data if item[1].size(0) > 0]
    imgs = [item[0] for item in data]
    boxes = [item[1] for item in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in boxes)

    # pad annot to same length
    if max_num_annots > 0:
        annot_padded = torch.ones((len(boxes), max_num_annots, 5)) * -1
        for idx, annot in enumerate(boxes):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(boxes), 1, 5)) * -1

    # imgs = imgs.permute(0, 3, 1, 2) # doesn't apply here
    return (imgs, torch.FloatTensor(annot_padded))


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, clss, regs, ancs, y_a, y_b, lam):
    return lam * criterion(clss, regs, ancs, y_a) + (1 - lam) * criterion(clss, regs, ancs, y_b)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / \
            self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
