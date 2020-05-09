# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
FROM: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # print('=' * 10)
        # print(original_image_sizes)

        images, targets = self.transform(images, targets)

        # print('=' * 10)
        # print(images.image_sizes)

        # print('=' * 10)
        # print(images.tensors)

        features = self.backbone(images.tensors)

        # print(features)
        # print(features.shape)
        # plot_ft = torch.squeeze(features, 0)[0].numpy()
        # print('=' * 10)
        # print(features)
        # fig = plt.figure()
        # plt.imshow(plot_ft)
        # plt.show()

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        # print('=' * 10)
        # print(proposals, proposal_losses)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        #
        # print('=' * 10)
        # print(detections, detector_losses)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        #
        # print('=' * 10)
        # print(detections)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)
