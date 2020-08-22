import argparse
import datetime
import os
import math
import time
import logging
import shutil
import sys
from collections import deque
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as O
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# don't prepend dot
from cfg import Cfg
from model import Yolov4, Yolo_loss
from utils.setup import init_args
from utils.data import collater, convert_batch, TransformDataset, AugTransform
from utils.dataset import YoloWrapper, BeeDataset
from utils.scheduler import GradualWarmupScheduler
from utils.methods import mixup_data, mixup_criterion

"""
Train with single GPU. For distributed training,
use dist-train.py, which requires mpi and horovod
"""

parser = argparse.ArgumentParser(description='Beetect Training with YOLO')

# dirs
parser.add_argument('--dump_dir', '-O', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--resume', '-R', type=str, help='Checkpoint file path to resume training')
parser.add_argument('--state_dict_dir', '-S', type=str, help='Local state dict in case downloading does not work')

# training
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--grad_accum_steps', type=int, default=1,
                    help='Gradient accumulation steps, used to increase batch size before optimizing to offset GPU memory constraint')
parser.add_argument('--max_grad_norm', type=float, default=0.1)
parser.add_argument('--img_h', type=int, default=608, help='Image size')
parser.add_argument('--img_w', type=int, default=608, help='Image size')
parser.add_argument('--iou_type', type=str, default='iou', help='iou type (iou, giou, diou, ciou)')

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
parser.add_argument('--max_checkpoint', type=int, default=10, help='Maximum number of checkpoints to keep (newest), set 0 to save all.')

# interval
parser.add_argument('--log_interval', type=int, default=300, help='Log interval per X iterations')
parser.add_argument('--val_interval', type=int, default=1, help='Val interval per X epoch')


iter = 0


def iter_step(epoch, loss, mean_loss, optimizer, params, args):
    global iter
    iter += 1
    tensorboard = params.tensorboard

    if iter % args.log_interval:
        tensorboard.add_scalar(tag='loss/loss', scalar_value=loss.item(), global_step=iter)
        tensorboard.add_scalar(tag='loss/total_loss', scalar_value=mean_loss, global_step=iter)
        tensorboard.add_scalar(tag='lr/lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=iter)


def train(model, train_loader, criterion, scheduler, optimizer, epoch, params, args):
    start = time.time()
    total_loss = []

    model.train()
    model.is_training = True

    pbar = tqdm(train_loader, desc='==> Train', position=1)
    idx = 0
    for batch in pbar:
        images, targets = convert_batch(batch, args.device)
        #print(images.tensors.shape)
        #print(targets)

        #images, targets_a, targets_b, lam = mixup_data(images, targets,
        #                                              args.alpha, args.is_cuda)
        #images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))

        outputs = model(images)
        #print(f'{epoch}-{idx}', len(outputs))

        #loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(outputs, targets)
        #print(loss)

        loss = loss.mean()
        # total_loss += loss.data[0]
        if loss == 0 or not torch.isfinite(loss):
            print('loss equal zero(0)')
            continue

        loss.backward()
        total_loss.append(loss.item())
        mean_loss = np.mean(total_loss)
        if (idx + 1) % args.grad_accum_steps == 0:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # zero grad first since first step requires zero grad before step
            optimizer.zero_grad()
            optimizer.step()

        iter_step(epoch, loss, mean_loss, optimizer, params, args)
        idx += 1
        pbar.update()
        pbar.set_postfix({
            'Loss': loss.item(),
            'Mean_loss': mean_loss,
            })
        # pbar.set_description()

    # end of training epoch
    # scheduler.step(mean_loss)
    scheduler.step(epoch)
    result = {'time': time.time()-start, 'loss': mean_loss}
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))

    return mean_loss


@torch.no_grad()
def validate(model, val_loader, optimizer, epoch, params, args):
    start = time.time()
    total_loss = []

    model.eval()
    model.is_training = False

    pbar = tqdm(val_loader, desc='==> Validate', position=2)
    for (images, targets) in pbar:
        images = images.to(args.device).float()
        targets = targets.to(args.device).float()

        images, targets_a, targets_b, lam = mixup_data(images, targets,
                                                      args.alpha, args.is_cuda)
        images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))

        outputs = model(images)

        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss = loss.mean()
        if loss == 0 or not torch.isfinite(loss):
            print('loss equal zero(0)')
            continue

        total_loss.append(loss.item())
        mean_loss = np.mean(total_loss)

        pbar.update()
        pbar.set_postfix({
            'Loss': loss.item(),
            'Mean_loss': mean_loss,
            })
        # pbar.set_description()

    # end of training epoch
    result = {'time': time.time()-start, 'loss': mean_loss}
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))

    return mean_loss


@torch.no_grad()
def test(model, test_loader, criterion, params, args):
    model.eval()

    pbar = tqdm(train_loader, desc='==> Train', position=1)
    idx = 0


if __name__ == '__main__':
    args = parser.parse_args()
    args, params = init_args(args, **Cfg)

    dataset = BeeDataset(annot_dir=args.annot_dir, img_dir=args.img_dir)

    train_prop = 0.8
    train_size = math.ceil(train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 320, 416, 512, 608, etc... default: 800 * 800
    # height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
    # width = 320 + 96 * m, m in {0, 1, 2, 3, ...}
    img_size = (args.img_h, args.img_w)

    # wrap dataset with transform wrapper
    train_dataset = YoloWrapper(dataset=train_dataset,
                                # transform=AugTransform(train=True, size=img_size),
                                cfg=args)
    val_dataset = YoloWrapper(dataset=val_dataset,
                              # transform=AugTransform(train=False, size=img_size),
                              cfg=args)

    kwargs = {'shuffle': True, 'collate_fn': collater}
    if args.is_cuda:
        kwargs['num_workers'] = args.workers
        kwargs['pin_memory'] = True
        # kwargs['drop_last'] = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, **kwargs)

    model = Yolov4(pretrained=True, num_classes=args.num_classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=args.device)

    criterion = Yolo_loss(device=args.device, batch=args.batch_size, num_classes=args.num_classes)
    #optimizer = O.AdamW(model.parameters(), lr=args.lr,
    #                    eps=args.eps, betas=(args.beta1, args.beta2))
    optimizer = O.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

    # chain warmup scheduler with plateau scheduler
    scheduler_plateau = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, verbose=True)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_plateau)

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

    # optimizer.zero_grad()
    # optimizer.step()

    for epoch in pbar:
        loss_train = train(model, train_loader, criterion, scheduler, optimizer, epoch, params, args)

        loss_val = validate(model, val_loader, criterion, optimizer, epoch, params, args)

        is_best = False
        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            is_best = True

        state = {
            'epoch': epoch,
            'iter': iter,
            'args': args,
            'loss': loss_val,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_path = os.path.join(ckpt_save_dir, 'checkpoint_{}.pt'.format(epoch))
        torch.save(state, save_path)

        if is_best:
            best_path = os.path.join(ckpt_save_dir, 'best_ckpt.pt')
            shutil.copy(save_path, best_path)
