import torch
import numpy as np
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
import torchvision.transforms as T

class AugTransform:
    """Flexible transforming"""

    def __init__(self, train, size=(224, 224)):
        transforms = [Resize(*size)]

        if train:
            # default p=0.5
            transforms.extend([Flip(), Rotate()])

        # normalize default imagenet
        # mean (0.485, 0.456, 0.406)
        # std (0.229, 0.224, 0.225)
        # transforms.append(Normalize())

        self.aug = Compose(transforms, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, image, target):
        aug_arg = {
            'image': np.array(image), # pil to numpy
            'bboxes': target.boxes,
            'labels': target.labels.numpy(),
        }
        augmented = self.aug(**aug_arg)

        # target.boxes =
        image = T.ToTensor()(augmented['image'])
        target.boxes = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)

        return image, target
