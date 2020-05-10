import numpy as np
import torch
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

class AugTransform:
    """Flexible transforming"""

    def __init__(self, train, size=(224, 224)):
        # transforms = [Resize(*size)]
        transforms = []

        if train:
            # default p=0.5
            transforms.extend([Flip(), Rotate()])

        # normalize default imagenet
        # mean (0.485, 0.456, 0.406)
        # std (0.229, 0.224, 0.225)
        # transforms.append(Normalize())

        self.aug = Compose(transforms)
        self.aug_train = Compose(transforms, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, image, target=None):
        aug_arg = {'image': np.array(image)} # original is pil

        if target is None:
            augmented = self.aug(**aug_arg)
            image = T.ToTensor()(augmented['image'])
            return image

        aug_arg['bboxes'] = target.boxes
        aug_arg['labels'] = target.labels
        augmented = self.aug_train(**aug_arg)

        # convert to tensor
        image = T.ToTensor()(augmented['image'])
        target.boxes = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)

        return image, target
