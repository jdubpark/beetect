import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


def main():
    data_dir = 'data'


class BeeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        """
        Return data based on index (idx)
        Data format:
            https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        Format summary:
            image: PIL image of size (H, W)
            target: dict {
                boxes (FloatTensor[N, 4]): [x0, y0, x1, y1] (N bounding boxes)
                lables (Int64Tensor[N])
                image_id (Int64Tensor[1]): unique for all images
                area (Tensor[N]): bbox area (used with the COCO metric)
                iscrowd (UInt8Tensor[N])
                # optional
                masks (UInt8Tensor[N, H, W])
                keypoitns (FloatTensor[N, K, 3]): K=[x, y, visibility]
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name =


if __name__ == '__main__':
    main()
