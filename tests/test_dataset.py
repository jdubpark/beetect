import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms as T
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch.utils.data import Subset, DataLoader

from beetect import BeeDatasetVid, AugTransform
from beetect.utils import Map
from beetect.scratchv1.transform import GeneralizedRCNNTransform

ia.seed(1)


parser = argparse.ArgumentParser(description='Beetect Test Dataset')
parser.add_argument('--annot', '--annots', type=str, metavar='S',
                    dest='annots', help='annotation directory')
parser.add_argument('--image', '--images', type=str, metavar='S',
                    dest='images', help='images directory')
parser.add_argument('--cpu', action='store_true', help='manually change to cpu')


def main():
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')

    dataset = Map({
        x: BeeDatasetVid(annot_dir=args.annots, img_dir=args.images,
                      transform=AugTransform(train=(x is 'train')))
        for x in ['train', 'val']
    })

    # split the dataset to train and val
    indices = torch.randperm(len(dataset.train)).tolist()
    dataset.train = Subset(dataset.train, indices[:-50])
    dataset.val = Subset(dataset.val, indices[-50:])

    # define training and validation data loaders
    # uses random sampler
    data_loader = Map({
        x: DataLoader(
            dataset[x], batch_size=5, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_fn)
        for x in ['train', 'val']
    })

    # plot(dataset)

    # test for KeyError
    # source: https://discuss.pytorch.org/t/keyerror-when-enumerating-over-dataloader/54210/5
    # for idx, (data, image) in enumerate(dataset):
    #     print(idx)

    test_transform = GeneralizedRCNNTransform(
        min_size=224, max_size=448, image_mean=[0.485, 0.456, 0.406],
        image_std = [0.229, 0.224, 0.225]
    )

    # print(next(enumerate(data_loader.train)))

    # print(dataset)
    for i, batch in enumerate(data_loader.train):
        # images, targets = batch
        images, targets = convert_batch_to_tensor(batch, device=device)
        # image_list, targets = test_transform(images, targets)
        for target in targets:
            try:
                xmin, ymin, xmax, ymax = target['boxes'].unbind(1)
            except Exception as e:
                print(target)
                raise ValueError('Unbind error')

        # fig = plt.figure()
        #
        # # reverse dims e.g. (3, 224, 244) => (224, 244, 3)
        # # since plt accepts channel as the last dim
        # image = images[0].permute(1, 2, 0)
        # target = targets[0]
        #
        # bbs = BoundingBoxesOnImage([
        #     BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in target['boxes']
        # ], shape=image.shape)
        #
        # image_bbs = bbs.draw_on_image(image, size=2, color=[0, 0, 255])
        #
        # plt.imshow((image_bbs * 255).astype(np.uint8))
        # plt.show()
        # plt.pause(1)


def plot(dataset, num_images=4):
    fig = plt.figure()

    for i in range(num_images):
        image, target = dataset[i]
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title('Sample #{}'.format(target.image_id))
        show_annots(ax, image, target)

    plt.show()


def show_annots(ax, image, target):
    """Show image with annotations (bounding boxes)"""
    ax.imshow(image)

    bboxes = target.boxes
    for bbox in bboxes:
        xtl, ytl, xbr, ybr = bbox
        height = abs(ytl - ybr)
        width = abs(xtl - xbr)

        """
        Args: (lower left x, lower left y), width, height

        lower left x = xtl
        lower left y = ytl
            Because imshow origin is 'upper' (meaning that 0 is the upper y-lim),
            lower left y axis is actually flipped and thus we need to flip
            the y axis as well (ybr -> ytl)
        """
        rect = patches.Rectangle((xtl, ytl), width, height,
                                 edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.pause(0.001) # pause for plots to update


def collate_fn(batch):
    """
    Reorders a batch for forward

    https://discuss.pytorch.org/t/making-custom-image-to-image-dataset-using-collate-fn-and-dataloader/55951/2
    default collate: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L42

    Arguments:
        batch: List[
            Tuple(
                image (List[N, img_size])
                target (Dict)
            ), ..., batch_size
        ]

    type: (...) -> List[Tuple[image, target], ..., batch_size]
    """
    # filter out batch item with empty target
    batch = [item for item in batch if item[1]['boxes'].size()[0] > 0]
    # reorder items
    image = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [image, target]

    """reference: vision/references/detection/utils.py"""
    # return tuple(zip(*batch))


def convert_batch_to_tensor(batch, device):
    batch_images, batch_targets = batch
    images = list(image.to(device) for image in batch_images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]
    return images, targets


if __name__ == '__main__':
    main()
