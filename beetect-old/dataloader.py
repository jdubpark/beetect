import random
from pathlib import Path

import cv2

import get_image_net
import get_custom_images
import custom_config
import visualizer


def main():
    C = custom_config.get()

    # get_image_net.download_images(C.wnid, C.dir.annots, C.dir.images, C.dir.synset)
    img_filepaths = get_image_net.get_images(C.dir.images)

    # cv_display(random.sample(img_filepaths, len(img_filepaths), C.dir.annots)
    plt_display(random.sample(img_filepaths, 1)[0], C.dir.annots)


# def get_dicts():


def plt_display(img_filepath, annot_dir):
    root = get_image_net.get_xml(img_filepath.stem, annot_dir)
    bbox = get_image_net.xml_pascal_bbox(root)
    visualizer.show_image(img_filepath.absolute().as_posix(), bbox)


def cv_display(img_filepaths, annot_dir):
    """ Display shuffled images one-by-one (orig size), enter -> next, q -> quit """
    for img_filepath in img_filepaths:
        fp_name = img_filepath.name
        root = get_image_net.get_xml(img_filepath.stem, annot_dir)
        bbox = get_image_net.xml_pascal_bbox(root)

        im = cv2.imread(img_filepath.absolute().as_posix())
        cv2.rectangle(im, bbox[0], bbox[1], (0, 0, 255))

        quit = False
        # print(im.shape)
        while True:
            cv2.imshow(fp_name, im)
            k = cv2.waitKey(1)

            if k == 13: # key enter (go to next)
                cv2.destroyWindow(fp_name)
                break

            elif k & 0xFF == ord('q'): # key q (exit)
                quit = True
                break

        if quit: # end cycle (triggered by key q)
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
