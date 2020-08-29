import argparse
import os
import random
import string
import xml.etree.cElementTree as ET
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf

from beetect.tf_utils import dataset_util


parser = argparse.ArgumentParser(description='Convert dataset into TFRecord file format')
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--out_dir', '-O', help='Directory to dump converted TFRecord')


def create_tf_example(data, dataset_classes):
    filename, bboxes = data

    # load as encoded bytes
    with tf.compat.v1.gfile.GFile(filename, 'rb') as fid:
        encoded_img_data = fid.read()

    nparr = np.frombuffer(encoded_img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_format = b'jpg'
    filename = filename.encode('utf8')

    height, width, _ = image.shape
    height_, width_ = float(height), float(width)

    xmins, xmaxs, ymins, ymaxs, classes, classes_text = [], [], [], [], [], []
    for bbox in bboxes:
        xtl, ytl, xbr, ybr, class_id = bbox
        # Normalize coordinates (x / width, y / height)
        xmins.append(xtl / width_)
        xmaxs.append(xbr / width_)
        ymins.append(ybr / height_)
        ymaxs.append(ytl / height_)

        classes.append(class_id) # class id
        classes_text.append(dataset_classes[class_id].encode('utf8')) # class name

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        # None, doesn't really matter
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_img_data),
        }))

    return tf_example


class Dataset(object):
    def __init__(self, annot_dir, img_dir, size=416, ext='jpg', shuffle=True):
        folder_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]

        self.ext = ext if ext[0] == '.' else '.'+ext
        self.annot_lists = {}
        self.img_dirs = {}

        for folder_name in folder_list:
            folder_dir = os.path.join(img_dir, folder_name)

            # folder name is annot file name
            annot_file = os.path.join(annot_dir, folder_name + '.xml')
            annots, rand_prefix = self.load_annotations(annot_file)

            # weed out empty frame annots or path-not-found ones
            annots = {k: v for k, v in annots.items() if len(v) != 0
                      and os.path.isfile(os.path.join(folder_dir, k.split('_')[1] + self.ext))}

            self.annot_lists.update(annots)
            self.img_dirs[rand_prefix] = folder_dir

        self.frame_lists = [f for f in self.annot_lists.keys()]

        if shuffle:
            np.random.shuffle(self.frame_lists)

    def __len__(self):
        return len(self.frame_lists)

    def __getitem__(self, idx):
        pframe = self.frame_lists[idx]
        pre, frame = pframe.split('_')
        img_dir = self.img_dirs[pre]
        frame_path = os.path.join(img_dir, frame + self.ext)

        # image = cv2.imread(frame_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = self.annot_lists[pframe]

        return frame_path, boxes

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
                obj_label = 1 # one label only
                bbox = list(map(float, [attr['xtl'], attr['ytl'], attr['xbr'], attr['ybr']]))
                bbox.append(obj_label) # obj_label is int

                # set up frame obj in frames
                if pframe not in annot_frames:
                    annot_frames[pframe] = []

                annot_frames[pframe].append(bbox)

        return annot_frames, rand_prefix


if __name__ == '__main__':
    args = parser.parse_args()

    # os.path.dirname(args.out_dir)
    for dir in [args.img_dir, args.annot_dir]:
        assert os.path.isdir(dir)

    writer = tf.io.TFRecordWriter(args.out_dir)

    # TODO(user): Write code to read in your dataset to examples variable
    dataset = Dataset(annot_dir=args.annot_dir, img_dir=args.img_dir)
    dataset_classes = ['background', 'bee']

    pbar = tqdm(dataset, desc='==> Converting', position=0)
    for data in pbar:
        tf_data = create_tf_example(data, dataset_classes)
        writer.write(tf_data.SerializeToString())

    writer.close()
