import argparse
import os
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from model import Yolov4
from model.utils import post_processing, plot_boxes_cv2


parser = argparse.ArgumentParser(description='Beetect Yolo Test')

parser.add_argument('--ckpt', '-C', type=str, help='Checkpoint file')
parser.add_argument('--img', '-I', type=str, help='Image for inference')


def plot(image, boxes, color=[255, 0, 0]):
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in boxes
    ], shape=image.shape)

    image_bbs = bbs.draw_on_image(image, size=2, color=color)

    ret_key = False
    while True:
        cv2.imshow('Inference', cv2.cvtColor(image_bbs, cv2.COLOR_RGB2BGR))
        # np.clip(image_bbs, 0, 1)

        k = cv2.waitKey(0) & 0xFF
        if k == ord('w') or k == ord('q'):
            ret_key = k == ord('w')
            break

    return ret_key


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        raise ValueError('Image must be a file')

    model = Yolov4(pretrained=True, n_classes=80, inference=True)
    model.eval()

    # img = Image.open(args.img)
    # img = T.Compose([
    #     T.Resize((608, 608)),
    #     T.ToTensor(),
    # ])(img)
    # # img = img.permute(-1, 0, 1) # for cv2
    # img = img.unsqueeze(0)

    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = T.ToTensor()(img)
    input = F.interpolate(input, size=608).unsqueeze(0)

    print(input.size())

    with torch.no_grad():
        # do_detect()
        output = model(input)

    boxes = post_processing(input, conf_thresh=0.4, nms_thresh=0.6, output=output)
    print(len(boxes), boxes[0])
    # print(boxes[0])
    # mp_img = np.einsum('xyz->yzx', img)
    # plot(img, boxes[0])
