import argparse
import cv2
import os
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import beetect.models.yolov3.utils as utils
from beetect.models.yolov3.model import YOLOv3, decode


parser = argparse.ArgumentParser(description='Beetect Yolov3 inference test')

parser.add_argument('--ckpt_path', '-C', type=str, help='Checkpoint file path')
parser.add_argument('--img', '-I', type=str, help='Image file path (mutually exclusive to video)')
# parser.add_argument('--vid', '-V', type=str, help='Path to video file (mutually exclusive to video)')
parser.add_argument('--input_size', '-size', type=int, default=512)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--score_threshold', '-score', type=float, default=0.3, help='Obj score (confidence) threshold')
parser.add_argument('--iou_threshold', '-iou', type=float, default=0.45, help='Min IoU threshold for NMS')

parser.add_argument('--lr_init', type=float, default=5e-4) # 1e-3 explodes for adamw (sgdw untested)
parser.add_argument('--lr_end', type=float, default=1e-6)
parser.add_argument('--decay', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float) # for adamw
parser.add_argument('--beta1', default=0.9, type=float) # "
parser.add_argument('--beta2', default=0.999, type=float) # "
parser.add_argument('--momentum', default=0.9, type=float) # for sgdw


if __name__ == '__main__':
    args = parser.parse_args()
    in_size = args.input_size

    anchors = [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]
    strides = [8, 16, 32]
    classes = ['bee']

    # for fpath in [args.ckpt_path, args.img]:
    for fpath in [args.img]:
        if fpath is None or not os.path.isfile(fpath):
            raise ValueError(f'Provided path is invalid: {fpath}')

    # process image
    img = cv2.imread(args.img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = img.shape[:2]
    img_data = utils.image_preporcess(np.copy(img), in_size)
    img_data = img_data[np.newaxis, ...].astype(np.float32)

    # create keras model
    input_layer = tf.keras.layers.Input([in_size, in_size, 3])
    feat_maps = YOLOv3(input_layer, num_classes=args.n_class,
                       anchors=anchors, strides=strides)
    bbox_tensors = []
    for i, feat_map in enumerate(feat_maps):
        bbox_tensor = decode(feat_map, strides, anchors, args.n_class, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    # model.summary()

    # reconstruct ckpt
    optimizer = tfa.optimizers.AdamW(
        weight_decay=args.wd, learning_rate=args.lr_init,
        beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, args.ckpt_path, max_to_keep=3)

    # load ckpt
    ckpt.restore(manager.latest_checkpoint)

    # inference
    # pred_bboxes = model.predict(img_data)
    pred_bboxes = model(img_data)
    # print(pred_bboxes.shape)

    # # print(pred_bboxes)
    pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]
    pred_bboxes = tf.concat(pred_bboxes, axis=0)
    print(pred_bboxes)
    bboxes = utils.postprocess_boxes(pred_bboxes, img_size, in_size, args.score_threshold)
    print(bboxes)
    bboxes = utils.nms(bboxes, args.iou_threshold, method='nms')
    # print(bboxes)

    # img_bbox = utils.draw_bbox(img, bboxes, classes)
    # img_bbox = Image.fromarray(img_bbox)
    # img_bbox.show()
