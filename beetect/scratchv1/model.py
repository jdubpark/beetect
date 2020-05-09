from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from beetect.scratchv1.faster_rcnn import FasterRCNN


def resnet18(pretrained=True, num_classes=2, **kwargs):
    """Pretrained ResNet-18 with FasterRCNN"""
    model = resnet.resnet18(pretrained=pretrained)
    # get fc in features
    in_features = model.fc.in_features
    # Detach head (single fc layer for ResNet), thus leaving only the backbone.
    # Will be replaced with FasterRCNN head (multiple-heads).
    # For concept of single vs. multiple-heads, see https://stackoverflow.com/a/56004582/13086908
    backbone = list(model.children())[:-4]
    backbone = nn.Sequential(*backbone)

    # define out channels (for FasterRCNN)
    backbone.out_channels = in_features

    # attach new head - FasterRCNN
    model = FasterRCNN(backbone, num_classes=num_classes,
                       min_size=224)
    return model


def resnet18_fpn(pretrained=True, num_classes=2, **kwargs):
    """Pretrained FPN-ResNet-18 with FasterRCNN"""
    backbone = resnet_fpn_backbone('resnet18', pretrained=pretrained)

    # out channels is already defined as 256
    # attach FasterRCNN head
    model = FasterRCNN(backbone, num_classes=num_classes,
                       min_size=224)

    return model

    """
    torchvision model forward()

    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    To compute total loss
    follow: https://github.com/pytorch/vision/blob/master/references/detection/engine.py#L30
    """



if __name__ == '__main__':
    resnet18()
