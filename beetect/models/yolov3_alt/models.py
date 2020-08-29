import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

from .utils import broadcast_iou, yolo_boxes, yolo_nms
from .components import Darknet, DarknetTiny, YoloConv, YoloConvTiny, YoloOutput


YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


class YOLOWrapper():
    def load_darknet_weights(self, model, weights_file, tiny=False):
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        if tiny:
            layers = YOLOV3_TINY_LAYER_LIST
        else:
            layers = YOLOV3_LAYER_LIST

        for layer_name in layers:
            sub_model = model.get_layer(layer_name)
            for i, layer in enumerate(sub_model.layers):
                if not layer.name.startswith('conv2d'):
                    continue
                batch_norm = None
                if i + 1 < len(sub_model.layers) and \
                        sub_model.layers[i + 1].name.startswith('batch_norm'):
                    batch_norm = sub_model.layers[i + 1]

                print("{}/{} {}".format(
                    sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

                filters = layer.filters
                size = layer.kernel_size[0]
                in_dim = layer.get_input_shape_at(0)[-1]

                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    # darknet [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(
                        wf, dtype=np.float32, count=4 * filters)
                    # tf [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(
                    wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])

                if batch_norm is None:
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)

        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()


class YOLOv3(YOLOWrapper):
    def __init__(self, size=416, channels=3, num_classes=2, max_boxes=100,
                 iou_threshold=0.5, score_threshold=0.5, training=True,
                 anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                          (59, 119), (116, 90), (156, 198), (373, 326)],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        # size = 416
        super(YOLOv3, self)

        assert isinstance(size, int) and size % 32 == 0

        self.size = size
        self.channels = channels
        self.training = training
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.anchors = np.array(anchors, np.float32) / size
        self.anchor_masks = np.array(anchor_masks)

    def __call__(self):
        size = self.size
        channels = self.channels
        anchors = self.anchors
        masks = self.anchor_masks
        num_classes = self.num_classes

        x = inputs = Input([size, size, channels], name='input')

        x_36, x_61, x = Darknet(name='yolo_darknet')(x)

        x = YoloConv(512, name='yolo_conv_0')(x)
        output_0 = YoloOutput(512, len(masks[0]), num_classes, name='yolo_output_0')(x)

        x = YoloConv(256, name='yolo_conv_1')((x, x_61))
        output_1 = YoloOutput(256, len(masks[1]), num_classes, name='yolo_output_1')(x)

        x = YoloConv(128, name='yolo_conv_2')((x, x_36))
        output_2 = YoloOutput(128, len(masks[2]), num_classes, name='yolo_output_2')(x)

        if self.training:
            return Model(inputs, (output_0, output_1, output_2), name='yolov3')

        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes),
                         name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes),
                         name='yolo_boxes_1')(output_1)
        boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], num_classes),
                         name='yolo_boxes_2')(output_2)

        outputs = Lambda(lambda x: yolo_nms(
            x, anchors, masks, num_classes, max_boxes=self.max_boxes,
            iou_threshold=self.iou_threshold, score_threshold=self.score_threshold
            ), name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return Model(inputs, outputs, name='yolov3')


class YOLOv3Tiny(YOLOWrapper):
    def __init__(self, size=416, channels=3, num_classes=2, max_boxes=100,
                 iou_threshold=0.5, score_threshold=0.5, training=True,
                 anchors=[(10, 14), (23, 27), (37, 58),
                          (81, 82), (135, 169), (344, 319)],
                 anchor_masks=[[3, 4, 5], [0, 1, 2]]):
        # size = 416
        super(YOLOv3Tiny, self)

        assert isinstance(size, int) and size % 32 == 0

        self.size = size
        self.channels = channels
        self.training = training
        self.num_classes = num_classes
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.anchors = np.array(anchors, np.float32) / size
        self.anchor_masks = np.array(anchor_masks)

    def __call__(self):
        size = self.size
        channels = self.channels
        anchors = self.anchors
        masks = self.anchor_masks
        num_classes = self.num_classes

        x = inputs = Input([size, size, channels], name='input')

        x_8, x = DarknetTiny(name='yolo_darknet')(x)

        x = YoloConvTiny(256, name='yolo_conv_0')(x)
        output_0 = YoloOutput(256, len(masks[0]), num_classes, name='yolo_output_0')(x)

        x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
        output_1 = YoloOutput(128, len(masks[1]), num_classes, name='yolo_output_1')(x)

        if self.training:
            return Model(inputs, (output_0, output_1), name='yolov3')

        boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes),
                         name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes),
                         name='yolo_boxes_1')(output_1)

        outputs = Lambda(lambda x: yolo_nms(
            x, anchors, masks, num_classes, max_boxes=self.max_boxes,
            iou_threshold=self.iou_threshold, score_threshold=self.score_threshold
            ), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))

        return Model(inputs, outputs, name='yolov3_tiny')


def YOLOLoss(anchors, num_classes=2, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, num_classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
