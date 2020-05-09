from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from beetect import AugTransform
from beetect.scratchv1 import resnet18, resnet18_fpn

ia.seed(1)


def main():
    # model = resnet18()
    model = resnet18_fpn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load('./model_best.pt', map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # print(list(model.parameters()))

    model.eval()

    # image = Image.open('732.png')
    image = Image.open('bee1.jpg')
    # image = Image.open('FudanPed00009.png')

    aug = AugTransform(train=False)
    image = aug(image)

    input = image.unsqueeze(0).to(device)
    image_np = np.asarray(image)

    # print(image.shape)
    # print(input.size())
    with torch.no_grad():
        output = model(input)

    # print(arch, epoch, loss)
    print(output)

    # plot(image_np, output)


def plot(image, output, color=[0, 0, 255]):
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in output[0]['boxes']
    ], shape=image.shape)

    image_bbs = bbs.draw_on_image(image, size=2, color=color)
    fig = plt.figure()
    plt.imshow(image_bbs)
    plt.show()


if __name__ == '__main__':
    main()
