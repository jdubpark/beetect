import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def main():
    simple_imgaug_test()
    return


def simple_imgaug_test():
    """ https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb """
    ia.seed(1)

    image = imageio.imread("https://upload.wikimedia.org/wikipedia/commons/8/8e/Yellow-headed_caracara_%28Milvago_chimachima%29_on_capybara_%28Hydrochoeris_hydrochaeris%29.JPG")
    image = ia.imresize_single_image(image, (298, 447))

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=0.2*447, x2=0.85*447, y1=0.3*298, y2=0.95*298),
        BoundingBox(x1=0.4*447, x2=0.65*447, y1=0.1*298, y2=0.4*298)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.GammaContrast(1.5),
        iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
    ])

    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # ia.imshow(bbs.draw_on_image(image, size=2))
    ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))


def default_imgaug_test():
    """
    simple imgaug augmentation, source:
    https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html#a-simple-example

    more info:
    https://nbviewer.jupyter.org/github/aleju/imgaug-doc/blob/master/notebooks/B02%20-%20Augment%20Bounding%20Boxes.ipynb
    """
    ia.seed(1)

    image = ia.quokka(size=(256, 256))
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=65, y1=100, x2=200, y2=150),
        BoundingBox(x1=150, y1=80, x2=200, y2=130)
    ], shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        after = bbs_aug.bounding_boxes[i]
        print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
            i,
            before.x1, before.y1, before.x2, before.y2,
            after.x1, after.y1, after.x2, after.y2)
        )

    # image with BBs before/after augmentation (shown below)
    image_before = bbs.draw_on_image(image, size=2)
    image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])


if __name__ == '__main__':
    main()
