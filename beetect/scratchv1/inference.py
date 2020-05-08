from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from beetect.scratchv1 import resnet18
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(1)


def main():
    model = resnet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load('./model_best.pth.tar', map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    arch = checkpoint['arch']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model.eval()

    image = Image.open('732.png')
    input = T.ToTensor()(image).unsqueeze(0).to(device)
    image_np = np.asarray(image)

    with torch.no_grad():
        output = model(input)

    print(arch, epoch, loss)
    print(output)

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=x[0], x2=x[2], y1=x[1], y2=x[3]) for x in output[0]['boxes']
    ], shape=image_np.shape)

    image_bbs = bbs.draw_on_image(image_np, size=2, color=[0, 0, 255])
    fig = plt.figure()
    plt.imshow(image_bbs)
    plt.show()


if __name__ == '__main__':
    main()
