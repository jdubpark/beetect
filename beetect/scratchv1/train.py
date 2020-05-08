import argparse
import copy
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as O
from beetect import BeeDataset, AugTransform
from beetect.scratchv1 import resnet18, utils
from beetect.utils import Map
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

model_names = ['resnet18']

# reference: https://github.com/pytorch/examples/blob/master/imagenet/main.py
parser = argparse.ArgumentParser(description='PyTorch ScratchV1 Training')
parser.add_argument('-a', '--arch', default='resnet18', metavar='ARCH',
                    choices=model_names,
                    help='model architecture: '+
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--data', default='/Users/pjw/pyProjects/dataset/honeybee/video',
                    type=str, metavar='S', help='data directory')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', dest='batch_size',
                    default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)',
                    dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--step-size', default=3, type=int, metavar='N',
                    help='lr step size (default: 3)')
parser.add_argument('--gamma', default=0.1, type=float, metavar='N',
                    help='gamma (default: 0.1)')
parser.add_argument('--val-size', default=50, type=int, metavar='N',
                    help='number of images used for val dataset',
                    dest='val_size')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('--anomaly', action='store_true',
                    help='Run train with torch.autograd.detect_anomaly')


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

def main():
    args = parser.parse_args()

    model = resnet18()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # prepare dataset
    annot_file = args.data+'/annot/hive-entrance-1-1min.xml'
    img_dir = args.data+'/frame/hive-entrance-1-1min/'

    dataset = Map({
        x: BeeDataset(annot_file=annot_file, img_dir=img_dir,
                      transform=get_transform(train=(x is 'train')))
        for x in ['train', 'val']
    })

    # split the dataset to train and val
    # indices = torch.randperm(len(dataset.train)).tolist()
    indices = list(range(len(dataset.train)))
    dataset.train = Subset(dataset.train, indices[:-args.val_size])
    dataset.val = Subset(dataset.val, indices[-args.val_size:])

    # define training and validation data loaders
    data_loader = Map({
        x: DataLoader(
            dataset[x], batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        for x in ['train', 'val']
    })

    # optimizer, take parameters directly since we are fine-tuning
    params = model.parameters() # [p for p in models.parameters() if p.requires_grad]
    optimizer = O.SGD(params, lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

    # learning rate scheduler
    lr_scheduler = O.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # printing info for optimizer
    # print('Params to learn:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print('\t', name)

    # optionally resume training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # if args.evaluate:
    #     validate(data_loader.val, model, device, args)
    #     return

    best_loss = 0
    running_batch = 0 # running batch count for tensorboard

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        running_batch = train(data_loader.train, model, optimizer, lr_scheduler, epoch, device, running_batch, args)

        # evaluate on val set
        loss = validate(data_loader.val, model, device, args)

        writer.add_scalar('epoch loss (val)', loss, epoch)

        # remember best loss and save checkpoint
        is_best = loss > best_loss
        best_loss = max(loss, best_loss)

        # save checkpoint
        save_checkpoint({
            'arch': args.arch,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': best_loss,
        }, is_best)

    writer.close()


def train(train_loader, model, optimizer, lr_scheduler, epoch, device, running_batch, args):
    """ Similar torchvision function is available
    function: train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
    source: https://github.com/pytorch/vision/blob/master/references/detection/engine.py#L13
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: {}/{}".format(epoch, args.epochs - 1))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images, targets = convert_batch_to_tensor(batch, device=device)

        # compute output
        with torch.autograd.set_detect_anomaly(mode=args.anomaly):

            # https://github.com/pytorch/vision/blob/master/references/detection/engine.py#L30

            loss_dict = model(images, targets)
            # print(batch_idx, loss_dict)
            loss = compute_total_loss(loss_dict)

            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # record loss
            # losses.update(losses_reduced.item())
            losses.update(loss.item())

            a = list(model.parameters())[0].clone()

            # compute gradient and do SGD and lr step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            b = list(model.parameters())[0].clone()

            print('Parameters being updated? {}'.format(torch.equal(a.data, b.data) is not True))

            for param_group in optimizer.param_groups:
                print('Parameters being updated? {}'.format(param_group['lr']))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)
            writer.add_scalar('batch loss (train)', loss, running_batch)
            running_batch += 1

    return running_batch


def validate(val_loader, model, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Validate: ')

    # switch to evaluate mode
    # model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, batch in enumerate(val_loader):
            images, targets = convert_batch_to_tensor(batch, device=device)

            # compute output
            loss_dict = model(images, targets)
            loss = compute_total_loss(loss_dict)

            # record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.print_freq == 0:
                progress.display(batch_idx)

    return losses.avg


def compute_total_loss(loss_dict):
    """Sum of all losses in dict returned by torchvision Faster RCNN
    Includes gradient - ready for backward()
    """
    total_loss = sum(loss for loss in loss_dict.values())
    return total_loss


def get_transform(train=False):
    """Returns transform"""
    return AugTransform(train)


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def convert_batch_to_tensor(batch, device):
    """Convert a batch (list) of images and targets to tensor CPU/GPU
    reference: https://github.com/pytorch/vision/blob/master/references/detection/engine.py#L27
    L27: images = list(image.to(device) for image in images)
    L28: targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    """
    batch_images, batch_targets = batch

    # concat list of image tensors into a tensor at dim 0
    # batch_images = torch.cat(batch_images, dim=0)

    images = list(image.to(device) for image in batch_images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets]

    return images, targets


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


if __name__ == '__main__':
    main()
