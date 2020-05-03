from imgaug import augmenters as iaa
import imgaug as ia

import PIL
import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

from torchvision.utils import make_grid
from torchvision import transforms

def show_dataset(dataset, n=6):
    imgs = torch.stack([dataset[i][0] for _ in range(n) for i in range(len(dataset))])
    grid = make_grid(imgs).numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.show()

class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Scale((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            # iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

tfs = transforms.Compose([
    ImgAugTransform(),
    transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder('./dataground/', transform=tfs)

show_dataset(dataset)
