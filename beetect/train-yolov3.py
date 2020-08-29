import argparse
import datetime
import os
import time
import math
import shutil
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from beetect.models.yolov3.config import cfg
from beetect.models.yolov3.dataset import Dataset
from beetect.models.yolov3.model import YOLOv3, decode, compute_loss


parser = argparse.ArgumentParser(description='Beetect Yolov3 training')

# dirs
parser.add_argument('--dump_dir', '-O', type=str)
parser.add_argument('--annot_dir', '-A', type=str)
parser.add_argument('--img_dir', '-I', type=str)
parser.add_argument('--ckpt_path', '-C', type=str, default=None, help='Checkpoint file to load for training')

# training
parser.add_argument('--num_epochs', '-e', type=int, default=100, help='Number of epoch')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--optim', type=str, default='adamw', help='Optimizer')
parser.add_argument('--warmup', type=float, default=5, help='Number of epoch for warmup')
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--grad_accum_steps', '-grad', type=int, default=1,
                    help='Gradient accumulation steps (optimize per X batch iterations) to increase batch size')

# hyperparams
parser.add_argument('--lr_init', type=float, default=5e-4) # 1e-3 explodes for adamw (sgdw untested)
parser.add_argument('--lr_end', type=float, default=1e-6)
parser.add_argument('--decay', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float) # for adamw
parser.add_argument('--beta1', default=0.9, type=float) # "
parser.add_argument('--beta2', default=0.999, type=float) # "
parser.add_argument('--momentum', default=0.9, type=float) # for sgdw

# intervals
parser.add_argument('--log_interval', type=int, default=10, help='Log interval per X batch iterations')
parser.add_argument('--val_interval', type=int, default=1, help='Val interval per X epoch')
parser.add_argument('--ckpt_interval', type=int, default=2, help='Checkpoint interval')

# misc
parser.add_argument('--ckpt_max_keep', type=int, default=20, help='Maximum number of checkpoints to keep while saving new')

# shallow
class Map(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def calc_lr(current_steps, params, args):
    if current_steps < params.warmup_steps:
        lr = current_steps / params.warmup_steps * args.lr_init
    else:
        lr = args.lr_end + 0.5 * (args.lr_init - args.lr_end) * (
            (1 + np.cos((current_steps - params.warmup_steps) / (params.total_steps - params.warmup_steps) * np.pi))
        )
    return lr


def train_step(model, trainset, optimizer, params, args):
    tvs = model.trainable_variables
    acc_loss = 0
    should_accum = args.grad_accum_steps > 1

    if should_accum:
        # create empty gradient list (not a tf.Variable list)
        accum_gradient = [tf.zeros_like(tv) for tv in tvs]

    pbar = tqdm(trainset, desc='==> Train', position=1)
    local_steps = 1
    for image_data, target in pbar:
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss, conf_loss, prob_loss = 0, 0, 0

            # optimizing process for THREE targets:
            # 1) smaller / 2) medium / 3) large
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], params.strides,
                                          params.iou_loss_thresh, args.num_classes, i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

                loss_name = False
                for idx, loss in enumerate(loss_items):
                    if math.isnan(loss):
                        loss_name = ['giou', 'conf', 'prob'][idx]

                if loss_name != False:
                    raise ValueError('{} is nan'.format(loss_name))


            total_loss = giou_loss + conf_loss + prob_loss

            # get gradient
            gradients = tape.gradient(total_loss, tvs)

        acc_loss += total_loss
        mean_loss = acc_loss / local_steps
        lr = calc_lr(params.global_steps, params, args)

        if should_accum and params.global_steps % args.grad_accum_steps:
            # accumulate gradient
            accum_gradient = [(acc_grad+grad) for acc_grad, grad in zip(accum_gradient, gradients)]
            # calculate mean grad
            accum_gradient = [grad/args.grad_accum_steps for grad in accum_gradient]
            # apply mean-calculated accum_grad
            optimizer.apply_gradients(zip(accum_gradient, tvs))
            # reset accum grad
            accum_gradient = [tf.zeros_like(tv) for tv in tvs]
            # update lr (after applying accum_grad)
            optimizer.lr.assign(lr)

        else:
            optimizer.apply_gradients(zip(gradients, tvs))
            # update lr
            optimizer.lr.assign(lr)

        # tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
        #          "prob_loss: %4.2f   total_loss: %4.2f" %(params.global_steps, optimizer.lr.numpy(),
        #                                                   giou_loss, conf_loss,
        #                                                   prob_loss, total_loss))

        pbar.update()
        pbar.set_description('lr {:.6f}'.format(lr))
        pbar.set_postfix({
            'Mean': '{:4.2f}'.format(mean_loss), # accumulated
            'GIoU': '{:4.2f}'.format(giou_loss),
            'Conf': '{:4.2f}'.format(conf_loss),
            'Prob': '{:4.2f}'.format(prob_loss),
            })

        # writing summary data
        if params.global_steps % args.log_interval == 0:
            with params.writer.as_default():
                tf.summary.scalar('lr', optimizer.lr, step=params.global_steps)
                tf.summary.scalar('loss/total_loss', total_loss, step=params.global_steps)
                tf.summary.scalar('loss/mean_loss', mean_loss, step=params.global_steps)
                tf.summary.scalar('loss/giou_loss', giou_loss, step=params.global_steps)
                tf.summary.scalar('loss/conf_loss', conf_loss, step=params.global_steps)
                tf.summary.scalar('loss/prob_loss', prob_loss, step=params.global_steps)

            params.writer.flush()

        # at last
        local_steps += 1
        params.global_steps += 1

    return mean_loss


if __name__ == '__main__':
    args = parser.parse_args()
    params = Map({})

    params.ckpt_save_dir = os.path.join(args.dump_dir, 'checkpoints', 'yolov3')
    params.log_dir = os.path.join(args.dump_dir, 'logs', 'yolov3')

    params.iou_loss_thresh = 0.5
    params.strides = [8, 16, 32]
    params.anchors = [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]

    for dir in [params.ckpt_save_dir, params.log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    trainset = Dataset(annot_dir=args.annot_dir, img_dir=args.img_dir, batch_size=args.batch_size)
    steps_per_epoch = len(trainset) # number of batches
    params.global_steps = 1 # init at 1
    params.total_steps = args.num_epochs * steps_per_epoch
    # params.warmup_steps = int(args.warmup * params.total_steps) # for percentage
    params.warmup_steps = args.warmup * steps_per_epoch

    input_size = 512
    input_tensor = tf.keras.layers.Input([input_size, input_size, 3])
    conv_tensors = YOLOv3(input_tensor, strides=params.strides, anchors=params.anchors,
                          num_classes=args.num_classes)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, params.strides, params.anchors, args.num_classes, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)

    # first_decay_steps = 1000
    # lr_decayed = cosine_decay_restarts(learning_rate, global_step, first_decay_steps)
    # optimizer = tf.keras.optimizers.Adam()
    if args.optim not in ['adam', 'adamw', 'sgdw']:
        raise ValueError(f'Optimizer must be a valid option. Provided: {args.optim}')

    if args.optim == 'adam':
        # use trackable optim for ckpt manager
        optimizer = tf.compat.v2.keras.optimizers.Adam(learning_rate=args.lr_init)

    elif args.optim == 'adamw':
        optimizer = tfa.optimizers.AdamW(
            weight_decay=args.wd, learning_rate=args.lr_init,
            beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps)

    elif args.optim == 'sgdw':
        optimizer = tfa.optimizers.SGDW(weight_decay=args.wd, learningrate=args.lr_init, momentum=args.momentum)

    now = datetime.datetime.now()
    writer_log_dir = os.path.join(params.log_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))
    params.writer = tf.summary.create_file_writer(writer_log_dir)

    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print([gpu.name for gpu in gpus])
    device = '/GPU:0' if gpus else '/CPU:0'

    best_loss = 1e5

    # all kwargs to Checkpoint should be trackable
    ckpt = tf.train.Checkpoint(
        epoch=tf.Variable(1), optimizer=optimizer,
        model=model)

    if args.ckpt_path is not None:
        # raises error since ckpt_path provides prefix, not the actual file (there .data and .index)
        # and .meta (if tf.v1 -> saved  graph structure is loaded, not eager)
        #if not os.path.isfile(args.ckpt_path):
        #    raise ValueError(f'Invalid checkpoint path, provided "{args.ckpt_path}"')

        ckpt.restore(args.ckpt_path)
        print(f'Model checkpoint "{os.path.basename(args.ckpt_path)}" restored from "{os.path.dirname(args.ckpt_path)}"')

        
    manager = tf.train.CheckpointManager(ckpt, params.ckpt_save_dir, max_to_keep=args.ckpt_max_keep)

    pbar = tqdm(range(args.num_epochs), desc='==> Epoch', position=0)
    for epoch in pbar:
        with tf.device(device):
            mean_loss = train_step(model, trainset, optimizer, params, args)

        ckpt.epoch.assign_add(1)
        # checkpoint.save(file_prefix=params.ckpt_save_dir)
        # checkpoint.restore(params.ckpt_save_dir).assert_consumed()

        #save_epoch = epoch % args.ckpt_interval == 0
        # save_epoch = int(ckpt.epoch) % args.ckpt_interval == 0
        # ckpt_epoch_file = os.path.join(params.ckpt_save_dir, f'epoch_{epoch}.h5')

        #if save_epoch:
        # model.save(ckpt_epoch_file)
        save_path = manager.save(checkpoint_number=epoch)
        print('Saved checkpoint for epoch {} at "{}"'.format(epoch, save_path))
        # ckpt.restore(manager.latest_checkpoint)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_ckpt_file = os.path.join(params.ckpt_save_dir, 'best_epoch')
            for ext in ['.data-00000-of-00001', '.index', '.meta']:
                if not os.path.isfile(save_path+ext):
                    continue

                shutil.copyfile(save_path+ext, best_ckpt_file+ext)
        #     model.save(best_ckpt_file)
