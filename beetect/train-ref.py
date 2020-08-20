import argparse
import os
import shutil
from tqdm.autonotebook import tqdm

import numpy as np
import torch
import torch.optim as O
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as T

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from beetect.model import EfficientDet


parser = argparse.ArgumentParser('Beetect training with EfficientDet implementation')

# model
parser.add_argument('--compound_coef', '--coef', type=int, default=3,
                    help='Coefficient of efficientdet [0 to 7]')

# paths
parser.add_argument('--dump_dir', '-O', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--resume', '-R', type=str, help='Checkpoint file path to resume training')
parser.add_argument('--state_dict_dir', '-S', type=str, help='Local state dict in case downloading does not work')

# hyperparams
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--gamma', type=float, default=1.5)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--patience', default=3, type=int, help='Patience for ReduceLROnPlateau before changing LR value')

parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--batch_size', '-b' type=int, default=8, help='Image batch size')
parser.add_argument('--image_size', type=int, default=512, help='Size of all images')
parser.add_argument('--num_class', type=int, default=2)

parser.add_argument('--workers', '-j', type=int, default=4, help='Number of workers, used only if using GPU')
parser.add_argument('--start_epoch', type=int, help='Start epoch, used for resume')
parser.add_argument('--seed', type=int, default=123)

# intervals
parser.add_argument('--test_interval', type=int, default=1, help='Test interval per X epoch')
parser.add_argument('--log_interval', type=int, default=300, help='Log interval per X iterations')


parser.add_argument("--es_min_delta", type=float, default=0.0,
                    help="Early stopping's parameter: minimum change loss to qualify as an improvement")
parser.add_argument("--es_patience", type=int, default=0,
                    help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_efficientdet_coco")
parser.add_argument("--saved_path", type=str, default="trained_models")



if __name__ == "__main__":
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

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if is_cuda:
            cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed)

    # don't pass it as args since it can't be serialized
    # https://discuss.pytorch.org/t/how-to-debug-saving-model-typeerror-cant-pickle-swigpyobject-objects/66304
    params.tensorboard = SummaryWriter(log_dir=log_dir)

    dataset = BeeDataset(annot_dir=annot_dir, img_dir=img_dir)

    train_prop = 0.8
    train_size = math.ceil(train_prop * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    kwargs = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater}
    if is_cuda:
        kwargs['num_workers'] = args.workers
        kwargs['pin_memory'] = True
        kwargs['batch_size'] = args.batch_size * torch.cuda.device_count()

    torch.cuda.empty_cache()

    training_set = CocoDataset(root_dir=args.data_path, set="train2017",
                               transform=T.Compose([Normalizer(), Augmenter(), Resizer()]))
    training_generator = DataLoader(training_set, **training_params)

    test_set = CocoDataset(root_dir=args.data_path, set="val2017",
                           transform=T.Compose([Normalizer(), Resizer()]))
    test_generator = DataLoader(test_set, **test_params)

    model = EfficientDet(num_classes=args.num_class,
                         compound_coef=args.coef)


    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)

    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)

    writer = SummaryWriter(args.log_path)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = O.AdamW(model.parameters(), lr=args.lr,
                        eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = O.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(args.n_epoch):
        model.train()
        # if torch.cuda.is_available():
        #     model.module.freeze_bn()
        # else:
        #     model.freeze_bn()
        epoch_loss = []
        pbar = tqdm(training_generator)
        for iter, data in enumerate(pbar):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                total_loss = np.mean(epoch_loss)

                pbar.set_description(
                    'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                        epoch + 1, args.n_epoch, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                        total_loss))
                writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

            except Exception as e:
                print(e)
                continue
        scheduler.step(np.mean(epoch_loss))

        if epoch % args.test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, args.n_epoch, cls_loss, reg_loss,
                    np.mean(loss)))
            writer.add_scalar('Test/Total_loss', loss, epoch)
            writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + args.es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(args.saved_path, "signatrix_efficientdet_coco.pth"))

                dummy_input = torch.rand(args.batch_size, 3, 512, 512)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.cuda()
                if isinstance(model, nn.DataParallel):
                    model.module.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model.module, dummy_input,
                                      os.path.join(args.saved_path, "signatrix_efficientdet_coco.onnx"),
                                      verbose=False)
                    model.module.backbone_net.model.set_swish(memory_efficient=True)
                else:
                    model.backbone_net.model.set_swish(memory_efficient=False)

                    torch.onnx.export(model, dummy_input,
                                      os.path.join(args.saved_path, "signatrix_efficientdet_coco.onnx"),
                                      verbose=False)
                    model.backbone_net.model.set_swish(memory_efficient=True)

            # Early stopping
            if epoch - best_epoch > args.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break
    writer.close()
