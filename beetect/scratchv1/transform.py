import math
import matplotlib.pyplot as plt
import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torch import nn
from torchvision.models.detection.roi_heads import paste_masks_in_image

ia.seed(1)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


class GeneralizedRCNNTransform(nn.Module):
    """
    FROM: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py

    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.
    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size
    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets):
        """
        Arguments:
            images,       # type: List[Tensor]
            targets=None  # type: Optional[List[Dict[str, Tensor]]]

        type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        """
        orig_imgs = [img.clone() for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))

            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        # test_plot_all(orig_imgs[0], images[0], targets[0], self.denormalize)
        # test_plot(image, bbox)

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def denormalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return image * std[:, None, None] + mean[:, None, None]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])

        image = _resize_image(image, size, float(self.max_size))

        if target is None:
            # test_plot(image)
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def torch_choice(self, l):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(l))).item())
        return l[index]

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,               # type: List[Dict[str, Tensor]]
                    image_shapes,         # type: List[Tuple[int, int]]
                    original_image_sizes  # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))
    max_size = float(torch.max(im_shape))
    # print('=' * 10)
    # print('Image size: {}'.format(im_shape))
    # print('Self min max: min {}, max {}'.format(self_min_size, self_max_size))
    scale_factor = self_min_size / min_size
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    # print('Scale factor: {}'.format(scale_factor))
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode='bilinear',
        align_corners=False)[0]
    return image


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def test_plot_all(orig, normalized, target, denormalize):
    # plot image, normalized images with boxes, and denormalized image
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in target['boxes']
    ], shape=normalized.shape)
    bbs_args = {'size': 2, 'color': [255, 0, 0]}
    image_bbs = bbs.draw_on_image(normalized.permute(1, 2, 0), **bbs_args)
    denormed = denormalize(normalized)
    denormed_bbs = bbs.draw_on_image(denormed.permute(1, 2, 0), **bbs_args)
    print(orig)
    print(image_bbs)
    print(denormed)
    ax1.set_title('Input')
    ax2.set_title('Normalized')
    ax3.set_title('Denormalized')
    ax1.imshow(orig.permute(1, 2, 0))
    ax2.imshow(np.clip(image_bbs, 0, 1))
    ax3.imshow(np.clip(denormed_bbs, 0, 1))
    plt.show()
    plt.pause(0.1)


def test_plot(image, bbox=None):
    fig = plt.figure()

    image_ = image.clone().detach()
    image = denormalize(image)

    if bbox is None:
        plt.imshow(image)
    else:
        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in bbox
        ], shape=image.shape)

        image_bbs = bbs.draw_on_image(image, size=2, color=[255, 0, 0])
        print(image, image_bbs)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(image)
        ax2.imshow(image_bbs)

    plt.show()
    plt.pause(0.1)

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    dtype, device = image.dtype, image.device

    image = image.clone().detach() # copy tensor
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)

    # denormalize image (image * std) + mean
    denorm = image * std[:, None, None] + mean[:, None, None]

    # hange dims (for plotting)
    # PyTorch tensors assume the channel is the first dimension
    # but matplotlib assumes the third dimension
    # e.g. (3, 224, 244) => (224, 244, 3)
    # permute takes all dims vs transpose only 2-d
    denorm = image.permute(1, 2, 0)

    return denorm
