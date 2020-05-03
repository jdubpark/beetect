import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox

from utils.dict_map import Map


def IoU(bbox1, bbox2):
    """
    Calculate Intersection over Union value for two bounding boxes, source:
    https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html#computing-intersections-unions-and-ious

    preferably, bbox1 = ground-truth, bbox2 = predicted

    bbox1, bbox2 = ((x min, y min), (x max, y max))
    """
    bb1 = BoundingBox(x1=bbox1[0][0], x2=bbox1[1][0], y1=bbox1[1][1], y2=bbox1[2][1])
    bb2 = BoundingBox(x1=bbox2[0][0], x2=bbox2[1][0], y1=bbox2[1][1], y2=bbox2[2][1])

    # Compute intersection, union and IoU value
    bb_inters = bb1.intersection(bb2).extend(all_sides=-1)
    bb_union = bb1.union(bb2).extend(all_sides=2)
    iou = bb1.iou(bb2)

    return Map({inter: bb_inters, union: bb_union, iou: iou})
