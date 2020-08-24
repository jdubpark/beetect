import glob
import os
import random
import string
import math
import xml.etree.cElementTree as ET
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from albumentations import (
    Compose,
    BboxParams,
    Flip,
    Rotate,
    Resize,
    Normalize,
    CenterCrop,
    RandomCrop,
    Crop
)
from albumentations.pytorch.transforms import ToTensor
from torch import Tensor
from torch.jit.annotations import List, Tuple
from torch.utils.data import Dataset, DataLoader


def collater(batch):
    # filter out batch item with empty target
    batch = list(filter(lambda img: img is not None, batch))
    batch = [item for item in batch if item[1][0].shape[0] > 0]
    # reorder items
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # image_sizes = [img.shape[-2:] for img in images]
    # images = batch_images(images)
    # image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
    #
    # for image_size in image_sizes:
    #     assert len(image_size) == 2
    #     image_sizes_list.append((image_size[0], image_size[1]))
    #
    # image_list = ImageList(images, image_sizes_list)
    #
    # return image_list, targets
    return images, targets


def convert_batch(batch, device):
    images, targets = batch
    images = [img.to(device, dtype=torch.float32).unsqueeze(0) for img in images]
    targets = [tgt.to(device).unsqueeze(0) for tgt in targets]

    # images are fix sized, targets are padded to cfg.boxes (60)
    images = torch.cat(images, dim=0) # .transpose(0, 3, 1, 2)
    # print(targets)
    # targets = np.concatenate(targets, axis=0)
    # targets = torch.from_numpy(targets)
    targets = torch.cat(targets, dim=0)
    # print(images.shape, targets.shape)

    # images = np.concatenate(images, axis=0)
    # images = images.transpose(0, 3, 1, 2)
    # images = torch.from_numpy(images).div(255.0)
    # bboxes = np.concatenate(bboxes, axis=0)
    # bboxes = torch.from_numpy(bboxes)
    return images, targets


# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/image_list.py#L7
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py#L187
def max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py#L195
def batch_images(images, size_divisible=32):
    # type: (List[Tensor], int) -> Tensor
    if torchvision._is_tracing():
        # batch_images() does not export well to ONNX
        # call _onnx_batch_images() instead
        return _onnx_batch_images(images, size_divisible)

    max_size = max_by_axis([list(img.shape) for img in images])
    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    batch_shape = [len(images)] + max_size
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


# https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py#L165
@torch.jit.unused
def _onnx_batch_images(images, size_divisible=32):
    # type: (List[Tensor], int) -> Tensor
    max_size = []
    for i in range(images[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    stride = size_divisible
    max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # which is not yet supported in onnx
    padded_imgs = []
    for img in images:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = F.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

    return torch.stack(padded_imgs)


class TransformDataset(Dataset):
    """ Wrapper around Dataset to apply transform if needed
    """
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.dataset[idx] # dataset has __getitem__
        image, target = self.transform(image, target)

        annots = torch.empty(0, 5)
        for i in range(len(target['boxes'])):
            # annot: [x1, y1, x2, y2, label_id]
            annot = torch.zeros((1, 5))
            annot[0, :4] = target['boxes'][i]
            annot[0, 4] = target['labels'][i]
            annots = torch.cat((annots, annot), dim=0)

        return image, annots


class AugTransform:
    """Flexible transforming"""

    def __init__(self, train, size=(224, 224)):
        transforms = [Resize(*size)]

        if train:
            # default p=0.5
            transforms.extend([
                Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225), p=1),
                Flip(), Rotate()])

        # transforms.extend([
        #     ToTensor()
        # ])

        self.aug = Compose(transforms)
        self.aug_train = Compose(transforms, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, image, target=None):
        aug_arg = {'image': np.array(image)} # original is pil

        if target is None:
            augmented = self.aug(**aug_arg)
            # image = T.ToTensor()(augmented['image'])
            image = T.ToTensor()(augmented['image'])
            return image

        aug_arg['bboxes'] = target['boxes']
        aug_arg['labels'] = target['labels']
        augmented = self.aug_train(**aug_arg)

        # convert to tensor
        image = T.ToTensor()(augmented['image'])
        target['boxes'] = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)

        # return image, target
        return image, target
