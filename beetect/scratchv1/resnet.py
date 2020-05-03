from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN


def resnet18(pretrained=True, num_classes=2, **kwargs):
    """Pretrained ResNet-18 with FasterRCNN"""
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)

    # out channels is already defined as 256
    # attach FasterRCNN head
    model = FasterRCNN(backbone, num_classes=num_classes,
                       min_size=720)

    return model

    """
    Below is for resnet18 without fpn
    """
    # model = resnet18(pretrained=True)
    # get fc in features
    # in_features = model.fc.in_features
    # Detach head (single fc layer for ResNet), thus leaving only the backbone.
    # Will be replaced with FasterRCNN head (multiple-heads).
    # For concept of single vs. multiple-heads, see https://stackoverflow.com/a/56004582/13086908
    # backbone = list(model.children())[:-1]
    # backbone = nn.Sequential(*backbone)
    #
    # # define out channels (for FasterRCNN)
    # backbone.out_channels = in_features
    #
    # # attach new head - FasterRCNN
    # model = FasterRCNN(backbone, num_classes=num_classes,
    #                    min_size=720)
    #
    # print(model)
    #
    # return model


if __name__ == '__main__':
    resnet18()
