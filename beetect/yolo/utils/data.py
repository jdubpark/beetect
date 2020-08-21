import glob
import os
import random
import string
import xml.etree.cElementTree as ET
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from albumentations import (
    Compose,
    BboxParams,
    Flip,
    Rotate,
    Resize,
    Normalize,
    CenterCrop,
    RandomCrop,
    Crop
)
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


def collater(batch):
    # filter out batch item with empty target
    batch = [item for item in batch if item[1]['boxes'].size()[0] > 0]
    # reorder items
    image = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [image, target]


def convert_batch_to_tensor(batch, device):
    batch_images, batch_targets = batch
    images = list(image.to(device, dtype=torch.float32) for image in batch_images)
    # targets = list(target.to(device) for target in batch_targets)
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
    return images, targets


class BeeDataset(Dataset):
    """ Bee dataset annotated in CVAT video format
    """

    def __init__(self, annot_dir, img_dir, ext='jpg'):
        """
        Args:
            annot_dir (string): Root dir of annotation file
            img_dir (string): Root dir of folder of images
        """

        # skip folders/files starting with .
        folder_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]

        self.annot_lists = {}
        self.img_dirs = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for folder_name in folder_list:
            # folder name is annot file name
            annot_file = os.path.join(annot_dir, folder_name + '.xml')
            annots, rand_prefix = self.read_annot_file(annot_file)
            self.annot_lists.update(annots)
            self.img_dirs[rand_prefix] = os.path.join(img_dir, folder_name)

        self.frame_lists = [f for f in self.annot_lists.keys()]
        self.ext = '.' + ext

    def __len__(self):
        return len(self.frame_lists)

    def __getitem__(self, idx):
        """
        Format Doc: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        Format:
            image: PIL image of size (H, W)
            target: dict {
                boxes (list[N, 4]): [x0, y0, x1, y1] (N bounding boxes)
                labels (Int64[N])
                image_id (Int64[1]): unique for all images
            }
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pframe = self.frame_lists[idx]
        pre, frame = pframe.split('_')
        img_dir = self.img_dirs[pre]
        frame_path = os.path.join(img_dir, frame + self.ext)

        image = Image.open(frame_path).convert('RGB')
        boxes = self.annot_lists[pframe]
        num_boxes = len(boxes)

        # there is only one label for all frames (bee body)
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        image_id = torch.tensor([idx], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes # later changed to tensor
        target['labels'] = labels
        target['image_id'] = image_id

        return image, target

    def read_annot_file(self, annot_file):
        """
        Read annotation file .xml exported from cvat (PASCAL VOC format)
        and return annotations by frames. Currently doesn't support
        tracking each object by id.

        Args:
            annot_file (string): Path to the annotation file
        """
        tree = ET.parse(annot_file)
        root = tree.getroot()
        annot_frames = {} # annotated frames

        # generate unique prefix for identification
        prefix_len = 4
        rand_prefix = ''.join(random.choices(string.ascii_letters + string.digits, k=prefix_len))

        # a track contains all annotated frames for an object
        tracks = [c for c in root if c.tag == 'track']

        for track in tracks:
            obj_id = track.attrib['id'] # assigned object id across all frames

            # box is essentially an annotated frame (of an object)
            for box in track:
                attr = box.attrib

                # skip object outside the frame (include occluded)
                if attr['outside'] != '0': continue

                frame = attr['frame'] # annotated frame id
                pframe = '{}_{}'.format(rand_prefix, frame) # _ separater
                # bbox position top left, bottom right
                bbox = [attr['xtl'], attr['ytl'], attr['xbr'], attr['ybr']]
                bbox = [float(n) for n in bbox] # string to float
                # if len(bbox) is False:
                #     print(pframe, bbox)

                # set up frame obj in frames
                if pframe not in annot_frames:
                    annot_frames[pframe] = []

                annot_frames[pframe].append(bbox)

        # print('=' * 20)
        # print(rand_prefix)
        # print(annot_frames)
        # print('=' * 20)

        return annot_frames, rand_prefix


class TransformDataset(Dataset):
    """ Wrapper around Dataset to apply transform if needed
    """
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.dataset[idx] # dataset has __getitem__
        image, target = self.transform(image, target)

        return image, target


class AugTransform:
    """Flexible transforming"""

    def __init__(self, train, size=(224, 224)):
        transforms = [Resize(*size)]

        if train:
            # default p=0.5
            transforms.extend([
                Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225), p=1),
                Flip(), Rotate()])

        # transforms.extend([
        #     ToTensor()
        # ])

        self.aug = Compose(transforms)
        self.aug_train = Compose(transforms, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, image, target=None):
        aug_arg = {'image': np.array(image)} # original is pil

        if target is None:
            augmented = self.aug(**aug_arg)
            # image = T.ToTensor()(augmented['image'])
            image = T.ToTensor()(augmented['image'])
            return image

        aug_arg['bboxes'] = target['boxes']
        aug_arg['labels'] = target['labels']
        augmented = self.aug_train(**aug_arg)

        # convert to tensor
        image = T.ToTensor()(augmented['image'])
        target['boxes'] = torch.as_tensor(augmented['bboxes'], dtype=torch.float32)

        # return image, target
        return image, target
