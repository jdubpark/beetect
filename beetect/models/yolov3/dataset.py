import cv2
import os
import random
import string
import xml.etree.cElementTree as ET

import numpy as np
import tensorflow as tf

from .utils import image_preporcess
from .config import cfg
from .transform import random_horizontal_flip, random_crop, random_translate


def bbox_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return inter_area / union_area


class Dataset(object):
    def __init__(self,
                 annot_dir,
                 img_dir,
                 batch_size=2,
                 input_size=512,
                 num_classes=2,
                 anchor_per_scale=3,
                 max_bbox_per_scale=150,
                 anchors=[1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875],
                 strides=[8, 16, 32], # small, medium, large
                 is_train=True,
                 shuffle=True,
                 ext='jpg'):
        self.batch_size = batch_size
        # self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchors = np.array(anchors, dtype=np.float32).reshape(3, 3, 2)
        self.anchor_per_scale = anchor_per_scale
        self.max_bbox_per_scale = max_bbox_per_scale
        self.is_train = is_train
        # self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = num_classes # len(self.classes)

        self.input_size = input_size # random.choice(self.input_sizes)
        self.output_sizes = []
        self.strides = np.array(strides)

        for stride in strides:
            assert input_size % stride == 0
            self.output_sizes.append(input_size // stride)
            # [64, 32, 16]

        self.annot_lists = {}
        self.img_dirs = {}

        folder_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        for folder_name in folder_list:
            annot_file = os.path.join(annot_dir, folder_name + '.xml')
            annots, rand_prefix = self.load_annotations(annot_file)
            self.annot_lists.update(annots)
            self.img_dirs[rand_prefix] = os.path.join(img_dir, folder_name)

        self.frame_lists = [f for f in self.annot_lists.keys()]

        if shuffle:
            np.random.shuffle(self.frame_lists)

        self.batch_count = 0
        self.num_samples = len(self.frame_lists)
        self.num_batchs = int(np.ceil(self.num_samples / batch_size))

        self.ext = ext if ext[0] == '.' else '.'+ext

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        if self.is_train:
            np.random.shuffle(self.frame_lists)
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))

            n_ch = 5 + self.num_classes # 4 box coordinates + 1 object confidence + n class confidence
            sb_ch = self.output_sizes[0] # small box # of channel
            mb_ch = self.output_sizes[1] # medium
            lb_ch = self.output_sizes[2] # large

            batch_label_sbbox = np.zeros((self.batch_size, sb_ch, sb_ch,
                                          self.anchor_per_scale, n_ch),
                                         dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, mb_ch, mb_ch,
                                          self.anchor_per_scale, n_ch),
                                         dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, lb_ch, lb_ch,
                                          self.anchor_per_scale, n_ch),
                                         dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                # create a batch
                while num < self.batch_size:
                    idx = self.batch_count * self.batch_size + num
                    if idx >= self.num_samples:
                        idx -= self.num_samples

                    pframe = self.frame_lists[idx]
                    pre, frame = pframe.split('_')
                    img_dir = self.img_dirs[pre]
                    frame_path = os.path.join(img_dir, frame + self.ext)
                    bboxes = self.annot_lists[pframe]

                    if not os.path.exists(frame_path):
                        raise KeyError("%s does not exist ... " %frame_path)

                    image = cv2.imread(frame_path)
                    image = np.array(image)
                    bboxes = np.array(bboxes)

                    if self.is_train:
                        # np.copy(image), np.copy(bboxes)
                        image, bboxes = random_horizontal_flip(image, bboxes)
                        image, bboxes = random_crop(image, bboxes)
                        image, bboxes = random_translate(image, bboxes)

                    # np.copy(image), np.copy(bboxes)
                    image, bboxes = image_preporcess(image, self.input_size, bboxes)

                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                self.batch_count += 1

                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                # no more batch to iterate
                self.batch_count = 0
                np.random.shuffle(self.frame_lists)
                raise StopIteration

    def load_annotations(self, annot_file):
        tree = ET.parse(annot_file)
        root = tree.getroot()
        annot_frames = {}

        prefix_len = 4
        rand_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=prefix_len))

        tracks = [c for c in root if c.tag == 'track']
        for track in tracks:
            obj_id = track.attrib['id'] # assigned object id across all frames
            # box is essentially an annotated frame (of an object)
            for box in track:
                attr = box.attrib

                # skip object outside the frame (include occluded)
                if attr['outside'] != '0':
                    continue

                frame = attr['frame'] # annotated frame id
                pframe = '{}_{}'.format(rand_prefix, frame) # _ separater
                # bbox position top left, bottom right
                obj_label = 1
                bbox = [attr['xtl'], attr['ytl'], attr['xbr'], attr['ybr'], obj_label]
                # bbox = [float(n) for n in bbox] # string to float
                bbox = np.array(bbox, dtype=np.float32)

                # set up frame obj in frames
                if pframe not in annot_frames:
                    annot_frames[pframe] = []

                annot_frames[pframe].append(bbox)

        return annot_frames, rand_prefix

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4]) # might be in float

            # one hot vector of n classes
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # x, y center points and h and w
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # scaled to each stride (dividing by each stride)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            n_anchor_family = self.anchors.shape[0] # 3
            for i in range(n_anchor_family):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4), dtype=np.float32)
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
