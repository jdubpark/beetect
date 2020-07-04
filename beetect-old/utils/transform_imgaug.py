from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(1)

class ImgAugTransform:
    """Flexible image Augmentation using imgaug and torch"""

    def __init__(self, train=False):
        self.train = train

        # augmentation for all
        aug = [
            # iaa.Resize(256),
            iaa.CropToFixedSize(224, 244),
            iaa.Fliplr(0.5),
        ]

        # more augmentation for training
        if train:
            aug.extend([
                # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
                # iaa.Sometimes(0.25, iaa.Multiply((1.2, 1.4))), # brightness
                iaa.Affine(rotate=(-20, 20), mode='symmetric'),
                # iaa.Sometimes(0.25,
                #               iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                #                          iaa.CoarseDropout(0.1, size_percent=0.5)])),
                # iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ])

        self.seq = iaa.Sequential(aug)


    def __call__(self, image, target):
        image = np.asarray(image)

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in target.boxes
        ], shape=image.shape)

        # Augment BBs and images
        image_aug, boxes_aug = self.seq(image=image, bounding_boxes=bbs)

        # convert back to PIL and array for torch transform
        image = Image.fromarray(image_aug)
        boxes = [tuple(bbox[0]) + tuple(bbox[1]) # x1, y1, x2, y2
                  for bbox in boxes_aug.bounding_boxes]

        # apply torch transform
        image = self.transform(image)

        target.boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return image, target

    def transform(self, target):
        """Transform methods from torch"""
        # image net norm
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        transform = T.Compose([
            T.ToTensor(),
            normalize,
        ])
        return transform(target)
