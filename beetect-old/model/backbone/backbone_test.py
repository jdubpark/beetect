from beetect.model.backbone import resnet18, resnet18_fpn

resnet18 = resnet18(pretrained=True)
resnet18_fpn = resnet18_fpn(pretrained=True)

print(resnet18_fpn)
