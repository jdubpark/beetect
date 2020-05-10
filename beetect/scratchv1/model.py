from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from beetect.scratchv1.faster_rcnn import FasterRCNN


def resnet18(pretrained=True, num_classes=2, **kwargs):
    """Pretrained ResNet-18 with FasterRCNN"""
    model = resnet.resnet50(pretrained=pretrained)
    # get fc in features
    in_features = model.fc.in_features
    # Detach head (single fc layer for ResNet), thus leaving only the backbone.
    # Will be replaced with FasterRCNN head (multiple-heads).
    # For concept of single vs. multiple-heads, see https://stackoverflow.com/a/56004582/13086908
    backbone = list(model.children())[:-1]
    backbone = nn.Sequential(*backbone)

    # define out channels (for FasterRCNN)
    backbone.out_channels = in_features

    # attach new head - FasterRCNN
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def resnet50(pretrained=True, num_classes=2, **kwargs):
    """Pretrained ResNet-50 with FasterRCNN"""
    model = resnet.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features # 2048 for resnet-50

    # detach fc at the end
    backbone = list(model.children())[:-1]
    backbone = nn.Sequential(*backbone)

    # define out channels (for FasterRCNN)
    backbone.out_channels = in_features

    # attach new head - FasterRCNN
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def resnet50_fpn(pretrained=True, num_classes=2, **kwargs):
    """Pretrained FPN-ResNet-18 with FasterRCNN"""

    norm_layer = misc_nn_ops.FrozenBatchNorm2d

    # BackboneWithFPN only gets layers specified in return_layers (below)
    # using IntermediateLayerGetter, so avgpool, fc, etc. aren't important
    backbone = resnet.resnet50(pretrained=pretrained, norm_layer=norm_layer)

    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256

    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    # backbone = resnet_fpn_backbone('resnet18', pretrained=pretrained)

    # out channels is already defined as 256
    # attach FasterRCNN head
    model = FasterRCNN(backbone, num_classes=num_classes, min_size=224)
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
