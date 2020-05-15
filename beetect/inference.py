import argparse
import cv2
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
import torchvision.ops as ops
import torchvision.transforms as T
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from beetect.model import resnet50_fpn


parser = argparse.ArgumentParser(description='Beetect Inference')
parser.add_argument('--image', type=str, metavar='S',
                    help='images file path (mutually exclusive to video)')
parser.add_argument('--video', type=str, metavar='S',
                    help='video file path (mutually exclusive to image)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='S',
                    dest='checkpoint', help='checkpoint file path')
parser.add_argument('--iou', type=float, metavar='N', default=0.2,
                    help='IoU threshold (default 0.2)')


def main():
    args = parser.parse_args()

    model = resnet50_fpn()
    model.eval()

    # validate args
    if args.image is not None and args.video is not None:
        raise ValueError('Argument conflict: image and video are mutually exclusive')
    if args.image is None and args.video is None:
        raise ValueError('Argument conflict: either image or video is required')
    if args.image is not None and os.path.exists(args.image) is False:
        raise ValueError('Invalid path: image')
    if args.video is not None and os.path.exists(args.video) is False:
        raise ValueError('Invalid path: video')
    if os.path.exists(args.checkpoint) is False:
        raise ValueError('Invalid path: checkpoint')

    # find map data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # retrieve checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # load model
    model.load_state_dict(checkpoint['state_dict'])
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # print(model)
    # print(list(model.parameters()))
    print(arch, epoch, loss)
    print('=' * 10)

    if args.image:
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = run_inference(image, model, device, iou_threshold=args.iou)
        plot(image, output)

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        i = 0
        while cap.isOpened():
            print('New frame - {:d}'.format(i))
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = run_inference(frame, model, device, iou_threshold=args.iou)
            ret_key = plot(frame, output)
            i += 1

            if ret_key is False:
                cap.release()
                break

    cv2.destroyAllWindows()


def run_inference(image, model, device, iou_threshold=0.3):
    """Run inference on image one at a time"""
    image = T.ToTensor()(image)
    input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input)

    # return first pred nms'ed (only one image)
    return nms_output(output[0], iou_threshold=iou_threshold)


def nms_output(output, iou_threshold):
    """Return nms as model output format"""
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']

    # nms returns indices to keep
    keep = ops.nms(boxes, scores, iou_threshold=iou_threshold)

    return {'boxes': boxes[keep], 'scores': scores[keep], 'labels': labels[keep]}


def plot(image, output, color=[255, 0, 0]):
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in output['boxes']
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
    main()
