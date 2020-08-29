import math

import numpy as np
import tensorflow as tf

from .utils import read_class_names, get_anchors
from .backbone import darknet53
from .common import Conv, Upsample
from .config import cfg


# num_classes = len(read_class_names(cfg.YOLO.CLASSES))
# anchors = get_anchors(cfg.YOLO.ANCHORS)
# strides = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH


def YOLOv3(input_layer,
           num_classes=2,
           anchors=[1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875],
           strides=[8, 16, 32]):
    n_ch = num_classes + 5
    # route 1 -> x_36
    # route 2 -> x_61
    x_36, x_65, conv_out = darknet53(input_layer)

    conv_out = Conv(conv_out, (1, 1, 1024,  512))
    conv_out = Conv(conv_out, (3, 3,  512, 1024))
    conv_out = Conv(conv_out, (1, 1, 1024,  512))
    conv_out = Conv(conv_out, (3, 3,  512, 1024))
    conv_out = Conv(conv_out, (1, 1, 1024,  512))

    conv_lobj_branch = Conv(conv_out, (3, 3, 512, 1024))
    conv_lbbox = Conv(conv_lobj_branch, (1, 1, 1024, 3*n_ch), activate=False, bn=False)

    conv_out = Conv(conv_out, (1, 1,  512,  256))
    conv_out = Upsample(conv_out)

    conv_out = tf.concat([conv_out, x_65], axis=-1)

    conv_out = Conv(conv_out, (1, 1, 768, 256))
    conv_out = Conv(conv_out, (3, 3, 256, 512))
    conv_out = Conv(conv_out, (1, 1, 512, 256))
    conv_out = Conv(conv_out, (3, 3, 256, 512))
    conv_out = Conv(conv_out, (1, 1, 512, 256))

    conv_mobj_branch = Conv(conv_out, (3, 3, 256, 512))
    conv_mbbox = Conv(conv_mobj_branch, (1, 1, 512, 3*n_ch), activate=False, bn=False)

    conv_out = Conv(conv_out, (1, 1, 256, 128))
    conv_out = Upsample(conv_out)

    conv_out = tf.concat([conv_out, x_36], axis=-1)

    conv_out = Conv(conv_out, (1, 1, 384, 128))
    conv_out = Conv(conv_out, (3, 3, 128, 256))
    conv_out = Conv(conv_out, (1, 1, 256, 128))
    conv_out = Conv(conv_out, (3, 3, 128, 256))
    conv_out = Conv(conv_out, (1, 1, 256, 128))

    conv_sobj_branch = Conv(conv_out, (3, 3, 128, 256))
    conv_sbbox = Conv(conv_sobj_branch, (1, 1, 256, 3*n_ch), activate=False, bn=False)

    # print(conv_sbbox.shape, conv_mbbox.shape, conv_lbbox.shape)
    return [conv_sbbox, conv_mbbox, conv_lbbox]

    # if training:
    #     output_tensors = []
    #     for i, conv_tensor in enumerate([conv_sbbox, conv_mbbox, conv_lbbox]):
    #         pred_tensor = decode(conv_tensor, strides, anchors, args.num_classes, i)
    #         output_tensors.append(conv_tensor)
    #         output_tensors.append(pred_tensor)
    #
    #     return tf.keras.Model(input_layer, (conv_sbbox, conv_mbbox, conv_lbbox), name='yolov3')


def decode(conv_output, strides, anchors, num_classes=2, i=0):
    """
    return Tensor[(batch_size, output_size, output_size, anchor_per_scale, 5 + num_classeses)]
            contains (x, y, w, h, score, probability)
    """

    conv_shape = tf.shape(conv_output)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_classes))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    # print(boxes1.shape, boxes1.dtype, boxes2.shape, boxes2.dtype)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_giou(boxes1, boxes2):
    # print(boxes1.shape, boxes1.dtype, boxes2.shape, boxes2.dtype)
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def compute_loss(pred, conv, label, bboxes, strides, iou_loss_thresh, num_classes=2, i=0):
    n_ch = 5 + num_classes

    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = strides[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, n_ch))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    # print('labels', label_prob.shape, 'logits', conv_raw_prob.shape)
    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss_ = giou_loss
    conf_loss_ = conf_loss
    prob_loss_ = prob_loss
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    isNan = False
    if math.isnan(giou_loss):
        print('... giou', giou_loss, giou_loss_)
        isNan = True
    elif math.isnan(conf_loss):
        print('... conf', conf_loss, conf_loss_)
        isNan = True
    elif math.isnan(prob_loss):
        print('... prob', prob_loss, prob_loss_)
        isNan = True

    if isNan:
        print('-'*20)
        print(label, label.shape)
        print('-'*20)
        print(bboxes, bboxes.shape)


    return giou_loss, conf_loss, prob_loss
