import argparse
import time
import os
import logging
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from beetect.models.yolov3_alt.models import YOLOv3, YOLOv3Tiny
from beetect.models.yolov3_alt.dataset import transform_images, load_tfrecord_dataset
from beetect.models.yolov3_alt.utils import draw_outputs


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--classes', type=str, help='path to classes file')
parser.add_argument('--weights', type=str, help='path to weights file')
parser.add_argument('--size', type=int, default=416)
parser.add_argument('--image', '--img', type=str, help='path to input image')
parser.add_argument('--video', '--vid', type=str, help='path to input video')
parser.add_argument('--output', '--out', type=str, help='path to output image/video')
parser.add_argument('--deb', action='store_true', help='Debian/Ubuntu for display')


def detect_image(model, img_raw, args):
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, args.size)

    t1 = time.time()
    boxes, scores, classes, nums = model(img)
    dur = t1 - time.time()

    if not args.silent:
        print('Time: {:4f}\nFPS: {:.2f}'.format(dur, 1.0/dur))

        print('Detection:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

    return img, dur


if __name__ == '__main__':
    args = parser.parse_args()
    args.silent = False

    if args.deb:
        os.environ['DISPLAY'] = ':0'

    with open(args.classes, 'r') as f:
        class_names = [c.strip() for c in f.readlines()]
        num_classes = len(class_names)

    print(f'Classes loaded, total {num_classes} classes')

    model = YOLOv3(size=args.size, num_classes=num_classes, training=False)()

    # prepare to load weights
    optimizer = tfa.optimizers.AdamW(
        weight_decay=5e-5, learning_rate=1e-3,
        beta_1=0.9, beta_2=0.999, epsilon=1e-6)
    ckpt = tf.train.Checkpoint(
        epoch=tf.Variable(1), optimizer=optimizer,
        model=model)
    manager = tf.train.CheckpointManager(ckpt, './data', max_to_keep=3)

    # load weights
    ckpt.restore(args.weights).expect_partial()
    print(f'Weights loaded from {args.weights}')

    if args.output:
        args.silent = True

    if args.image:
        assert os.path.isfile(args.image)

        img = cv2.imread(args.image)
        img_bbox, dur = detect_image(model, img, args)

        while True:
            cv2.imshow('Detect', img_bbox)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif args.video:
        assert os.path.isfile(args.video)

        cap = cv2.VideoCapture(args.video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if args.output:
            assert os.path.isdir(os.path.dirname(args.output))
            out = cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

        if frame_len < 1:
            raise ValueError('Frame length is invalid')

        times = []
        time.sleep(1)
        if not cap.isOpened():
            raise ValueError('Cap is not opened')

        pbar = tqdm(range(frame_len), desc='==> Frames')
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            if img is None:
                time.sleep(0.1)
                continue

            img_bbox, dur = detect_image(model, img, args)
            times.append(dur)
            times = times[-20:]

            img_bbox = cv2.putText(img_bbox, 'Time: {:.2f}ms'.format(sum(times)/len(times)*1000), (0, 30),
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            if args.output:
                out.write(img_bbox)
            else:
                cv2.imshow('Detect', img_bbox)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update()

        cap.release()
        if args.output:
            out.release()

    if not args.output:
        cv2.destroyAllWindows()
