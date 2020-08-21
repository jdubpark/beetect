import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# shallow
class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def init_args(args):
    params = Map({})

    assert os.path.isdir(args.dump_dir), 'Dump dir must be a valid dir'

    ckpt_save_dir = os.path.join(args.dump_dir, 'checkpoints')
    log_dir = os.path.join(args.dump_dir, 'logs')

    for dir in [ckpt_save_dir, log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    args.is_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.is_cuda else 'cpu')

    if args.is_cuda:
        torch.cuda.empty_cache()
        cudnn.deterministic = True
        # os.environ["CUDA_VISIBLE_DEVICES"] 

    if args.seed is not None:
        random.seed(args.seed)
        np.random(args.seed)
        torch.manual_seed(args.seed)
        if args.is_cuda:
            torch.cuda.manual_seed(args.seed)

    # don't pass it as args since it can't be serialized
    # https://discuss.pytorch.org/t/how-to-debug-saving-model-typeerror-cant-pickle-swigpyobject-objects/66304
    params.tensorboard = SummaryWriter(log_dir=log_dir)

    return args, params


def collater():
    return
