import argparse
import os
import math
import time
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as O
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from beetect.model import FocalLoss, EfficientDet, BBoxTransform, ClipBoxes
from beetect.utils import collater, convert_batch_to_tensor, BeeDataset, TransformDataset, AugTransform

"""
Train with single GPU. For distributed training,
use dist-train.py, which requires mpi and horovod
"""

parser = argparse.ArgumentParser(description='Beetect Training')

# model
parser.add_argument('--compound_coef', '-C', type=int, default=3,
                    help='Coefficient of efficientdet [0 to 7]')

# dirs
parser.add_argument('--dump_dir', '-O', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--resume', '-R', type=str, help='Checkpoint file path to resume training')
parser.add_argument('--state_dict_dir', '-S', type=str, help='Local state dict in case downloading does not work')

# training
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--grad_accum_steps', type=int, default=2,
                    help='Gradient accumulation steps, used to increase batch size before optimizing to offset GPU memory constraint')
parser.add_argument('--max_grad_norm', type=float, default=0.1)

# hyperparams
parser.add_argument('--lr', dest='lr', type=float, default=1e-2)
parser.add_argument('--decay', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--patience', default=3, type=int, help='Patience for ReduceLROnPlateau before changing LR value')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

# other
parser.add_argument('--seed', type=int)
parser.add_argument('--workers', '-j', type=int, default=4, help='Number of workers, used only if using GPU')
parser.add_argument('--start_epoch', type=int, help='Start epoch, used for resume')

# interval
parser.add_argument('--log_interval', type=int, default=300, help='Log interval per X iterations')
parser.add_argument('--val_interval', type=int, default=1, help='Val interval per X epoch')


iter = 0

# shallow
class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def iter_step(epoch, mean_loss, cls_loss, reg_loss, optimizer, params, args):
    global iter
    iter += 1
    tensorboard = params.tensorboard

    if iter % args.log_interval:
        tensorboard.add_scalar(tag='loss/cls_loss', scalar_value=cls_loss.item(), global_step=iter)
        tensorboard.add_scalar(tag='loss/reg_loss', scalar_value=reg_loss.item(), global_step=iter)
        tensorboard.add_scalar(tag='loss/total_loss', scalar_value=mean_loss, global_step=iter)
        tensorboard.add_scalar(tag='lr/lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=iter)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, train_loader, criterion, scheduler, optimizer, epoch, device, params, args):
    start = time.time()
    total_loss = []

    model.train()
    model.is_training = True
    model.freeze_bn()

    pbar = tqdm(train_loader, desc='==> Train', position=1)
    idx = 0
    for (images, targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device).float()

        regression, classification, anchors = model(images)
        cls_loss, reg_loss = criterion(classification, regression, anchors, targets)

        # print(cls_loss, reg_loss)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            print('loss equal zero(0)')
            continue

        loss.backward()
        total_loss.append(loss.item())
        mean_loss = np.mean(total_loss)
        if (idx + 1) % args.grad_accum_steps == 0:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # zero grad first since first step requires zero grad beforehand
            optimizer.zero_grad()
            optimizer.step()

        iter_step(epoch, mean_loss, cls_loss, reg_loss, optimizer, params, args)
        idx += 1
        pbar.update()
        pbar.set_postfix({
            'Cls_loss': cls_loss.item(),
            'Reg_loss': reg_loss.item(),
            'Mean_loss': mean_loss,
            })
        # pbar.set_description()

    # end of training epoch
    scheduler.step(mean_loss)
    result = {'time': time.time()-start, 'loss': mean_loss}
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))

    return mean_loss


def validate(model, val_loader, optimizer, epoch, device, params, args):
    model.eval()
    model.is_training = False
    # with torch.no_grad():
    #     evaluate(dataset, model)


def train(model, test_loader, criterion, device, params, args):
    model.eval()

    pbar = tqdm(train_loader, desc='==> Train', position=1)
    idx = 0

    with torch.no_grad():
        for (images, targets) in pbar:
            images = images.to(device).float()
            targets = targets.to(device)

            regression, classification, anchors = model(images)

            transformed_anchors = BBoxTransform(anchors, regression) # regress boxes
            transformed_anchors = ClipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


if __name__ == '__main__':
    args = parser.parse_args()
    params = Map({})

    dump_dir = os.path.abspath(args.dump_dir)
    annot_dir = os.path.abspath(args.annot_dir)
    img_dir = os.path.abspath(args.img_dir)
    ckpt_save_dir = os.path.join(dump_dir, 'checkpoints')
    log_dir = os.path.join(dump_dir, 'logs')

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    if args.state_dict_dir is not None:
        args.state_dict_dir = os.path.abspath(args.state_dict_dir)

    for dir in [ckpt_save_dir, log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    if is_cuda:
        torch.cuda.empty_cache()

    # don't pass it as args since it can't be serialized
    # https://discuss.pytorch.org/t/how-to-debug-saving-model-typeerror-cant-pickle-swigpyobject-objects/66304
    params.tensorboard = SummaryWriter(log_dir=log_dir)

    dataset = BeeDataset(annot_dir=annot_dir, img_dir=img_dir)

    train_prop = 0.8
    train_size = math.ceil(train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # wrap dataset with transform wrapper
    input_size = (896, 896) # smaller = memory efficient
    train_dataset = TransformDataset(dataset=train_dataset,
                                     transform=AugTransform(train=True, size=input_size))
    val_dataset = TransformDataset(dataset=val_dataset,
                                   transform=AugTransform(train=False, size=input_size))

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if is_cuda else {}
    kwargs['shuffle'] = True
    kwargs['collate_fn'] = collater

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, **kwargs)

    network = 'efficientdet-d3'
    config = {'input_size': 896, 'backbone': 'B3', 'W_bifpn': 160,
              'D_bifpn': 5, 'D_class': 4}

    model = EfficientDet(num_classes=args.num_class,
                                 compound_coef=args.compound_coef)
    model.to(device)

    criterion = FocalLoss()
    optimizer = O.AdamW(model.parameters(), lr=args.lr,
                        eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    pbar = tqdm(range(args.n_epoch), desc='==> Epoch', position=0)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f'... Loading checkpoint from {args.resume}')
            ckpt = torch.load(args.resume)
            args.start_epoch = ckpt['epoch']
            pbar = tqdm(range(args.n_epoch+args.start_epoch), desc='==> Epoch')
            iter = ckpt['iter']
            loss = ckpt['last_loss']
            best_loss = ckpt['best_loss']
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in pbar:
        loss = train(model, train_loader, criterion, scheduler, optimizer, epoch, device, params, args)

        # validate(model, val_loader, criterion, optimizer, epoch, device, params, args)

        is_best = False
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            is_best = True

        state = {
            'epoch': epoch,
            'iter': iter,
            'args': args,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_path = os.path.join(ckpt_save_dir, 'checkpoint_{}.pt'.format(epoch))
        torch.save(state, save_path)

        if is_best:
            best_path = os.path.join(ckpt_save_dir, 'best_ckpt.pt')
            shutil.copy(save_path, best_path)
