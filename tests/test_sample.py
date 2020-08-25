import argparse
import os
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd.profiler as profiler
import torchvision.ops as ops
import torchvision.transforms as T
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from beetect.model_old2 import EfficientDetBackbone


parser = argparse.ArgumentParser(description='EfficientDet Sample Test')

parser.add_argument('--checkpoint', '-c', dest='ckpt', type=str,
                    help='Checkpoint directory')
parser.add_argument('--image', '-i', dest='img', type=str,
                    help='Path to image file (mutually exclusive to video)')
parser.add_argument('--video', '-v', dest='vid', type=str,
                    help='Path to video file (mutually exclusive to image)')
parser.add_argument('--iou', type=float, default=0.2,
                    help='IoU threshold (default 0.2)')


def run_inference(image, model, device, iou_threshold=0.3):
    """Run inference on image one at a time"""
    image = T.ToTensor()(image)
    input = image.unsqueeze(0).to(device)

    print(input.size())
    with torch.no_grad():
        output = model(input)

    print(output)

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
    args = parser.parse_args()

    if args.img is not None and args.vid is not None:
        raise ValueError('Argument conflict: image and video are mutually exclusive')
    if args.img is None and args.vid is None:
        raise ValueError('Argument conflict: either image or video is required')
    if args.img is not None and os.path.exists(args.img) is False:
        raise ValueError('Invalid path: image')
    if args.vid is not None and os.path.exists(args.vid) is False:
        raise ValueError('Invalid path: video')
    if os.path.exists(args.ckpt) is False:
        raise ValueError('Invalid path: checkpoint')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('... Loading checkpoint: {} ...'.format(os.path.abspath(args.ckpt)))
    checkpoint = torch.load(args.ckpt, map_location=device)
    print('... Loaded checkpoint')

    cargs = checkpoint['args']
    model = EfficientDetBackbone(num_classes=cargs.num_class,
                                 compound_coef=cargs.compound_coef)
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=torch.cuda.is_available()) as prof:
    if args.img:
        # image = cv2.imread(args.img)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(args.img)
        image = T.Resize((512, 512))(image)
        output = run_inference(image, model, device, iou_threshold=args.iou)
        plot(image, output)
    else:
        cap = cv2.VideoCapture(args.vid)
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

    # print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=10))
