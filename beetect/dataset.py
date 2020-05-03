import xml.etree.ElementTree as ET
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from beetect.utils import Map


class BeeDataset(Dataset):
    """Bee dataset pulled from multiple videos"""

    def __init__(self, annot_file, img_dir, device, transform=None):
        """
        Args:
            annot_file (string): Path to the annotation file
            img_dir (string): Root folder of images
        """
        self.frame_list, self.frame_annots = self.read_annot_file(annot_file)
        self.img_dir = img_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        """
        Format Doc: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        Format:
            image: PIL image of size (H, W)
            target: dict {
                boxes (FloatTensor[N, 4]): [x0, y0, x1, y1] (N bounding boxes)
                lables (Int64Tensor[N])
                image_id (Int64Tensor[1]): unique for all images
                area (Tensor[N]): bbox area (used with the COCO metric)
                iscrowd (UInt8Tensor[N])
                # optional
                masks (UInt8Tensor[N, H, W])
                keypoitns (FloatTensor[N, K, 3]): K=[x, y, visibility]
            }
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame = str(self.frame_list[idx]) # frame name (e.g. 47 -> 47th frame)
        frame_path = os.path.join(self.img_dir, frame + '.png')

        image = Image.open(frame_path).convert('RGB')
        boxes = self.frame_annots[frame] # frame boxes
        num_boxes = len(boxes)

        # boxes = torch.as_tensor(boxes)
        # there is only one label for all frames (bee body)
        labels = torch.ones((num_boxes,), dtype=torch.int64)

        target = Map({})
        target.boxes = boxes
        target.lables = labels
        target.image_id = frame

        if self.transform:
            image, target = self.transform(image, target)

        return image.to(self.device), target.to(self.device)

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
        frames = {}

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
                # bbox position top left, bottom right
                bbox = [attr['xtl'], attr['ytl'], attr['xbr'], attr['ybr']]
                bbox = [float(n) for n in bbox] # string to float

                # set up frame obj in frames
                if frame not in frames:
                    frames[frame] = []

                frames[frame].append(bbox)

        frame_list = sorted([int(n) for n in frames.keys()]) # list of annotated frames

        return frame_list, frames
