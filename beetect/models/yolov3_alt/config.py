
from easydict import EasyDict as edict

__C = edict()
cfg = __C # from .config import cfg

# DIVIDE anchors by input size! e.g. anchors / 512
# anchors = np.array(cfg.anchors, dtype=np.float32) / size
# anchor_masks = np(cfg.anchor_masks)
__C.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
               (59, 119), (116, 90), (156, 198), (373, 326)]
__C.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
__C.anchors_tiny = [(10, 14), (23, 27), (37, 58),
                    (81, 82), (135, 169), (344, 319)]
__C.anchor_masks_tiny = [[3, 4, 5], [0, 1, 2]]
