import argparse
import numpy as np
import os
import random
import shutil
import sys
import time
import warnings
from tqdm import tqdm

import horovod.torch as hvd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim as O
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from model.efficientdet import EfficientDet
from model.losses import FocalLoss
from datasets import VOCDetection, CocoDataset, get_augumentation, detection_collate, Resizer, Normalizer, Augmenter, collater
from utils import EFFICIENTDET, get_state_dict
from eval import evaluate

parser = argparse.ArgumentParser(description='Beetect Distributed Training')
# parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
#                     type=str, help='VOC or COCO')
# parser.add_argument(
#     '--dataset_root',
#     default='/root/data/VOCdevkit/',
#     help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/]')

# dir
parser.add_argument('--dump_dir', type=str, help='Directory for dumping outputs')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# training
parser.add_argument('--num_epoch', default=500, type=int)
parser.add_argument('--batch_size', '-b', dest='batch_size' default=32, type=int)
parser.add_argument('--num_class', default=2, type=int, help='Number of class used in model')
# parser.add_argument('--device', default=[0, 1], type=list, help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', '--grad_accum_steps',
                    dest='grad_accum_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--log_steps', default=300, type=int, help='Log interval')

# hyperparams
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

parser.add_argument('--cpu', action='store_true', type=bool, help='Use CPU only')
# gpu
parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument(
    '--multiprocessing-distributed',
    action='store_true',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training')

# other
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=24, type=int, help='seed for initializing training. ')

iteration = 1

def update_iter(args):
    global iteration

    iteration += 1

    if iteration % args.log_steps:
        pass


def train(train_loader, model, scheduler, optimizer, epoch, args):


    start = time.time()
    total_loss = []
    model.train()
    model.module.is_training = True
    model.module.freeze_bn()
    optimizer.zero_grad()

    pbar = tqdm(train_loader, total=len(train_loader), desc=f'==> Training Epoch {epoch}')
    for idx, (images, annotations) in pbar:
        images = images.cuda().float()
        annotations = annotations.cuda()
        cls_loss, reg_loss = model([images, annotations])
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if bool(loss == 0):
            print('loss equal zero(0)')
            continue
        loss.backward()
        if (idx + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        total_loss.append(loss.item())

        update_iter()
        pbar.update()
        pbar.set_postfix({
            'Cls_loss': cls_loss.item(),
            'Reg_loss': reg_loss.item(),
            'Mean_loss': np.mean(total_loss),
            })
    scheduler.step(np.mean(total_loss))
    result = {
        'time': time.time() - start,
        'loss': np.mean(total_loss)
    }
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))


def test(dataset, model, epoch, args):
    model = model.module
    model.eval()
    model.is_training = False
    with torch.no_grad():
        evaluate(dataset, model)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            # args.rank = int(os.environ["RANK"])
            args.rank = 1
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    # Training dataset
    train_dataset = []
    if(args.dataset == 'VOC'):
        train_dataset = VOCDetection(root=args.dataset_root, transform=transforms.Compose(
            [Normalizer(), Augmenter(), Resizer()]))
        valid_dataset = VOCDetection(root=args.dataset_root, image_sets=[(
            '2007', 'test')], transform=transforms.Compose([Normalizer(), Resizer()]))
        args.num_class = train_dataset.num_classes()
    elif(args.dataset == 'COCO'):
        train_dataset = CocoDataset(
            root_dir=args.dataset_root,
            set_name='train2017',
            transform=transforms.Compose(
                [
                    Normalizer(),
                    Augmenter(),
                    Resizer()]))
        valid_dataset = CocoDataset(
            root_dir=args.dataset_root,
            set_name='val2017',
            transform=transforms.Compose(
                [
                    Normalizer(),
                    Resizer()]))
        args.num_class = train_dataset.num_classes()

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)

    checkpoint = []
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        network = 'efficientdet-d5'
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    model = EfficientDet(num_classes=args.num_class,
                         network=network,
                         W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                         D_class=EFFICIENTDET[network]['D_class']
                         )
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            print('Run with DistributedDataParallel with divice_ids....')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            print('Run with DistributedDataParallel without device_ids....')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()
        print('Run with DataParallel ....')
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) , optimizer, scheduler
    optimizer = O.AdamW(model.parameters(), lr=args.lr)
    scheduler = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)
    cudnn.benchmark = True

    pbar = tqdm(range(args.start_epoch, args.num_epoch), desc='==> Epoch')
    for epoch in pbar:
        train(train_loader, model, scheduler, optimizer, epoch, args)

        if (epoch + 1) % 5 == 0:
            test(valid_dataset, model, epoch, args)

        state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': get_state_dict(model)
        }
        ckpt_path = os.path.join(args.ckpt_save_dir, f'checkpoint_{epoch}.pt')
        torch.save(state, ckpt_path)


if __name__ == "__main__":
    args = parser.parse_args()

    args.dump_dir = os.path.abspath(args.dump_dir)
    ckpt_save_dir = os.path.join(args.dump_dir, 'checkpoints')
    log_dir = os.path.join(args.dump_dir, 'logs')

    for dir in [ckpt_save_dir, log_dir]
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    # for easier access throughout the code
    args.ckpt_save_dir = ckpt_save_dir
    args.log_dir = log_dir

    hvd.init()

    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device('cuda' if cuda else 'cpu')

    tensorboard = SummaryWriter(log_dir=log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if cuda:
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)

    if cuda:
        torch.cuda.set_device(hvd.local_rank())

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context')
        and mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'


    train_dataset = ''
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    network = 'efficientdet-d5'
    model = EfficientDet(num_classes=args.num_class, network=network,
                         W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                         D_class=EFFICIENTDET[network]['D_class']
                         )

    lr_scaler = hvd.size()

    if cuda:
        model.to(device)


    optimizer = O.AdamW(model.parameters(), lr=args.lr)
    scheduler = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)
    cudnn.benchmark = True
