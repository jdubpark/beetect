import argparse
import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as O
from torch.utils.data import Subset, DataLoader
from beetect import BeeDataset, ImgAugTransform
from beetect.utils import Map
from beetect.scratchv1 import resnet18

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

# pretty bad accuracy :/
best_acc1 = 0


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
                      transform=get_transform(train=(x is 'train')),
                      device=device)
        for x in ['train', 'val']
    })

    # split the dataset to train and val
    # indices = torch.randperm(len(dataset.train)).tolist()
    indices = range(len(dataset.train))
    dataset.train = Subset(dataset.train, indices[:-args.val_size])
    dataset.val = Subset(dataset.val, indices[-args.val_size:])

    # define training and validation data loaders
    data_loader = Map({
        x: DataLoader(
            dataset[x], batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        for x in ['train', 'val']
    })

    # loss function
    criterion = nn.CrossEntropyLoss().cuda()

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
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(data_loader.train, model, criterion, optimizer, epoch, device, args)

        # evaluate on val set
        acc1 = validate(data_loader.val, model, criterion, device, args)

        # loss step after train and val (changed after PyTirch 1.1.0)
        lr_scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # save checkpoint
        cpt_meta = Map({})
        cpt_meta.arch = args.arch
        cpt_meta.epoch = epoch + 1
        cpt_meta.state_dict = model.state_dict()
        cpt_meta.best_acc1 = best_acc1
        cpt_meta.optimizer = optimizer
        save_checkpoint(cpt_meta, is_best)


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: {}/{}".format(epoch, args.epochs - 1))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # images = [img.to(device) for img in images]
        # for target in targets:
        #     target.boxes = [tbox.to(device) for tbox in target]

        # compute output
        output = model(images, targets)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Validate: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            # images = [img.to(device) for img in images]
            # targets = [tg.to(device) for tg in targets]

            # compute output
            output = model(images)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def get_transform(train=False):
    """Returns transform"""
    return ImgAugTransform(train)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


if __name__ == '__main__':
    main()
