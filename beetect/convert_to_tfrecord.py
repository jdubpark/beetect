#
# Adapted from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
#

import argparse
import contextlib2
import os
import random
import math
import string
import xml.etree.cElementTree as ET
from tqdm import tqdm

import cv2
import numpy as np
import tensorflow as tf

from beetect.tf_utils import dataset_util
from beetect.tf_utils import tf_record_creation_util


parser = argparse.ArgumentParser(description='Convert dataset into TFRecord file format')
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--out_file', '-O', help='Directory to dump converted TFRecord')
parser.add_argument('--no_shard', action='store_true')
parser.add_argument('--shard_size', type=int, default=1000, help='Number of images per shard')


dataset_classes = ['background', 'bee']


def create_tf_example(data):
    filename, bboxes = data

    # load as encoded bytes
    with tf.compat.v1.gfile.GFile(filename, 'rb') as fid:
        encoded_img_data = fid.read()

    nparr = np.frombuffer(encoded_img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_format = b'jpg'
    filename = filename.encode('utf8')

    height, width, _ = image.shape

    xmins, xmaxs, ymins, ymaxs, classes, classes_text = [], [], [], [], [], []
    for bbox in bboxes:
        xtl, ytl, xbr, ybr, class_id = bbox
        # Normalize coordinates (x / width, y / height)
        xmin, xmax = xtl / width, xbr / width
        ymin, ymax = ybr / height, ytl / height
        # swap values if min is bigger than max (for some reason)
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

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
        # it's easier to move around annot files than whole frame folders
        # so start with available annot files
        # folder_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        annot_list = [f for f in os.listdir(annot_dir) if not f.startswith('.')]

        self.ext = ext if ext[0] == '.' else '.'+ext
        self.annot_lists = {}
        self.img_dirs = {}

        skip_non_keyframes = ['hive-1']

        for annot_file in annot_list:
            filename = os.path.splitext(annot_file)[0] # rid of ext
            print(f'Loading {filename}')
            annot_path = os.path.join(annot_dir, annot_file) # need ext
            img_path = os.path.join(img_dir, filename)
            annots, rand_prefix = self.load_annotations(annot_path, filename in skip_non_keyframes)

            # weed out empty frame annots or path-not-found ones
            annots = {k: v for k, v in annots.items() if len(v) != 0
                      and os.path.isfile(os.path.join(img_path, k.split('_')[1] + self.ext))}

            self.annot_lists.update(annots)
            # unique rand_prefix acts as a cursor to its loaded img_dir
            self.img_dirs[rand_prefix] = img_path

        print(f'Loaded all data!')
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

    def load_annotations(self, annot_file,
                         skip_outside=True,
                         skip_occluded=True,
                         skip_non_keyframes=False):
        tree = ET.parse(annot_file)
        root = tree.getroot()
        annot_frames = {}

        prefix_len = 4
        rand_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=prefix_len))

        tracks = [c for c in root if c.tag == 'track']
        images = [c for c in root if c.tag == 'image']

        if len(tracks) > 0:
            for track in tracks:
                obj_id = track.attrib['id'] # assigned object id across all frames
                # box is essentially an annotated frame (of an object)
                for box in track:
                    attr = box.attrib

                    if (skip_outside and attr['outside'] == '1') or \
                        (skip_occluded and attr['occluded'] == '1') or \
                        (skip_non_keyframes and attr['keyframe']) == '0':
                        continue

                    pframe, bbox = self.annot_box(attr, rand_prefix)
                    # set up frame obj in frames
                    if pframe not in annot_frames:
                        annot_frames[pframe] = []
                    annot_frames[pframe].append(bbox)

        elif len(images) > 0:
            for img in images:
                img_id = img.attrib['id']
                # basename with extension (image file name)
                img_bname = os.path.basename(img.attrib['name'])

                for box in img:
                    attr = box.attrib

                    # no keyframe or outside for images
                    if skip_occluded and attr['occluded'] == '1':
                        continue

                    pframe, bbox = self.annot_box(attr, rand_prefix, img_bname)
                    # set up frame obj in frames
                    if pframe not in annot_frames:
                        annot_frames[pframe] = []
                    annot_frames[pframe].append(bbox)

        else:
            raise ValueError(f'Annot file does not contain any annotation, provided "{annot_file}"')

        return annot_frames, rand_prefix

    def annot_box(self, attr, rand_prefix, frame=None):
        """
        attr -> attrib of each child loaded from annot file
        rand_prefix -> unique classifier for each annot file (since frame name duplicates, e.g. 0.jpg, 1.jpg)
        frame -> frame name later needed to load the image '{frame}.jpg'
                 provide if using cvat image format, where all boxes are of one image,
                 else, leave it blank to use attr['frame'] (of cvat video)
        """
        if frame is None:
            frame = attr['frame'] # annotated frame id (for cvat video)

        pframe = '{}_{}'.format(rand_prefix, frame) # _ separater
        # bbox position top left, bottom right
        obj_label = 1 # one label only
        bbox = list(map(float, [attr['xtl'], attr['ytl'], attr['xbr'], attr['ybr']]))
        bbox.append(obj_label) # obj_label is int
        return pframe, bbox


if __name__ == '__main__':
    args = parser.parse_args()

    # out_file
    for dir in [args.img_dir, args.annot_dir, os.path.dirname(args.out_file)]:
        assert os.path.isdir(dir)

    dataset = Dataset(annot_dir=args.annot_dir, img_dir=args.img_dir)
    pbar = tqdm(dataset, desc='==> Converting', position=0)

    if args.no_shard:
        writer = tf.io.TFRecordWriter(args.out_file)
        for data in pbar:
            tf_example = create_tf_example(data)
            writer.write(tf_example.SerializeToString())
        writer.close()

    else:
        output_filebase = args.out_file
        num_shards = math.ceil(len(dataset) / args.shard_size)
        # shard data
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards)

            idx = 0
            for data in pbar:
                tf_example = create_tf_example(data)
                output_shard_index = idx % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
                idx += 1
