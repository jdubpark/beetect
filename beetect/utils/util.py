import json
import logging
import os
import socket

import numpy as np
import torch
import pandas as pd

# def collater(batch):
#     batch = [item for item in batch if item[1]['boxes'].size()[0] > 0]
#     # reorder items
#     images = [item[0] for item in batch]
#     targets = [item[1] for item in batch]
#     return [images, targets]

def collater(data):
    #print(data)
    data = [item for item in data if item[1].size(0) > 0]
    imgs = [item[0] for item in data]
    boxes = [item[1] for item in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    max_num_annots = max(annot.shape[0] for annot in boxes)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(boxes), max_num_annots, 5)) * -1
        for idx, annot in enumerate(boxes):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(boxes), 1, 5)) * -1

    # imgs = imgs.permute(0, 3, 1, 2) # doesn't apply here
    return (imgs, torch.FloatTensor(annot_padded))


def convert_batch_to_tensor(batch, device):
    """ Convert a batch (list) of images and targets to tensor CPU/GPU
    """
    batch_images, batch_targets = batch
    # concat list of image tensors into a tensor at dim 0
    # batch_images = torch.cat(batch_images, dim=0)
    images = list(image.to(device) for image in batch_images)
    targets = list(target.to(device) for target in batch_targets)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
    return images, targets


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
