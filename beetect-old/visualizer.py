import random

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox

import get_image_net
import custom_config


def main():
    C = custom_config.get()
    img_filepaths = get_image_net.get_images(C.dir.images)
    img_filepath = random.sample(img_filepaths, 1)[0]
    root = get_image_net.get_xml(img_filepath.stem, C.dir.annots)
    bbox = get_image_net.xml_pascal_bbox(root)
    show_image(img_filepath.absolute().as_posix(), bbox)
    return


def show_image(img_filepath, bbox):
    print(img_filepath)
    im = cv2.imread(img_filepath)
    # plt uses RGB, cv2 uses BGR
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # bbs = BoundingBox(x1=bbox[0][0], x2=bbox[1][0], y1=bbox[0][1], y2=bbox[1][1])
    # ia.imshow(bbs.draw_on_image(im, size=2))

    fig, ax = plt.subplots(1) # create figure and axes
    ax.imshow(im) # display the image

    xh = bbox[1][0] - bbox[0][0]
    yh = bbox[1][1] - bbox[0][1]

    # create a rectangle patch
    rect = patches.Rectangle(bbox[0], xh, yh, linewidth=1, edgecolor='r', facecolor='none')
    # add the patch to the axes
    ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()
