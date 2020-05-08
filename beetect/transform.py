import torch
from torchvision import transforms as T
from fastai.vision.image import *

class Transform:
    """Flexible transforming with fast.ai"""

    def __init__(self, train):
        self.train = train

    def __call__(self, image, target):

        # image = pil2tensor()
        # bbox = ImageBBox.create(*image.size, target.boxes, labels=target.labels)
        #
        # print(bbox)
        # print(target.labels)

        image = T.ToTensor()(image)
        target.boxes = torch.as_tensor(target.boxes, dtype=torch.float32)

        return image, target
