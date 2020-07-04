import argparse
import os
import math
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as O
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from model.efficientdet import EfficientDet
from utils import collater, convert_batch_to_tensor, BeeDataset, TransformDataset, AugTransform

"""
Train with single GPU. For distributed training,
use dist-train.py, which requires mpi and horovod
"""

parser = argparse.ArgumentParser(description='Beetect Training')

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
parser.add_argument('--grad_accum_steps', type=int, default=1,
                    help='Gradient accumulation steps, used to increase batch size before optimizing to offset GPU memory constraint')
parser.add_argument('--max_grad_norm', type=float, default=0.1)

# hyperparams
parser.add_argument('--learning_rate', '-lr', dest='lr', type=float, default=0.01)
parser.add_argument('--weight_decay', '-wd', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--patience', default=3, type=int, help='Patience for ReduceLROnPlateau before changing LR value')

# other
parser.add_argument('--seed', type=int)
parser.add_argument('--workers', '-j', type=int, default=4, help='Number of workers, used only if using GPU')
parser.add_argument('--start_epoch', type=int, help='Start epoch, used for resume')

# interval
parser.add_argument('--log_interval', type=int, default=300, help='Log interval per X iterations')


iter = 0


def iter_step(epoch, mean_loss, cls_loss, reg_loss, scheduler, args):
    global iter
    iter += 1
    tensorboard = args.tensorboard

    if iter % args.log_interval:
        tensorboard.add_scalar(tag='loss/cls_loss', scalar_value=cls_loss.item(), global_step=iter)
        tensorboard.add_scalar(tag='loss/reg_loss', scalar_value=reg_loss.item(), global_step=iter)
        tensorboard.add_scalar(tag='loss/total_loss', scalar_value=mean_loss, global_step=iter)
        #tensorboard.add_scalar(tag='lr/lr', scalar_value=scheduler.get_last_lr()[-1], global_step=iter)


def train(model, train_loader, scheduler, optimizer, epoch, device, args):
    start = time.time()
    total_loss = []

    model.train()
    model.is_training = True
    model.freeze_bn()

    pbar = tqdm(train_loader, desc='==> Train', position=1)
    idx = 0
    for (images, targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device)
        images = images.float()

        cls_loss, reg_loss = model([images, targets])
        # print(cls_loss, reg_loss)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if bool(loss == 0):
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

        iter_step(epoch, mean_loss, cls_loss, reg_loss, scheduler, args)
        idx += 1
        pbar.update()
        pbar.set_postfix({
            'Cls_loss': cls_loss.item(),
            'Reg_loss': reg_loss.item(),
            'Mean_loss': mean_loss,
            })

    # end of training epoch
    scheduler.step(mean_loss)
    result = {'time': time.time()-start, 'loss': mean_loss}
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))

    return mean_loss


def validate(model, val_loader, optimizer, epoch, device, args):
    model.eval()
    model.is_training = False
    # with torch.no_grad():
    #     evaluate(dataset, model)


if __name__ == '__main__':
    args = parser.parse_args()

    dump_dir = os.path.abspath(args.dump_dir)
    annot_dir = os.path.abspath(args.annot_dir)
    img_dir = os.path.abspath(args.img_dir)
    ckpt_save_dir = os.path.join(dump_dir, 'checkpoints')
    log_dir = os.path.join(dump_dir, 'logs')

    if args.state_dict_dir is not None:
        args.state_dict_dir = os.path.abspath(args.state_dict_dir)

    for dir in [ckpt_save_dir, log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    args.tensorboard = SummaryWriter(log_dir=log_dir)

    torch.cuda.empty_cache()

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if is_cuda:
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)

    dataset = BeeDataset(annot_dir=annot_dir, img_dir=img_dir)

    train_prop = 0.8
    train_size = math.ceil(train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # wrap dataset with transform wrapper
    input_size = (896, 896)
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
    # network = 'efficientdet-d5'
    # config = {'input_size': 1280, 'backbone': 'B5', 'W_bifpn': 288,
    #           'D_bifpn': 7, 'D_class': 4}

    model = EfficientDet(num_classes=args.num_class, network=network,
                         local_state_dict=args.state_dict_dir,
                         W_bifpn=config['W_bifpn'], D_bifpn=config['D_bifpn'],
                         D_class=config['D_class'])
    model.to(device)

    optimizer = O.AdamW(model.parameters(), lr=args.lr,
                        eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, verbose=True)

    iter = 0
    best_loss = 1000
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
        loss = train(model, train_loader, scheduler, optimizer, epoch, device, args)

        # validate(model, val_loader, optimizer, epoch, device, args)

        if loss < best_loss:
            best_loss = loss

        state = {
            'epoch': epoch,
            'iter': iter,
            'args': args,
            'loss': loss,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(ckpt_save_dir, f'checkpoint_{epoch}.pt'))
