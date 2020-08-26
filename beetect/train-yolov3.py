import argparse
import os
import time
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

# training
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epoch')
parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
parser.add_argument('--warmup', type=float, default=0.1, help='% of epoch for warmup')
parser.add_argument('--batch_size', '-b', type=int, default=64)

# hyperparams
parser.add_argument('--lr_init', type=float, default=1e-3)
parser.add_argument('--lr_end', type=float, default=1e-6)
parser.add_argument('--decay', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)

# intervals
parser.add_argument('--log_interval', type=int, default=300, help='Log interval per X iterations')
parser.add_argument('--val_interval', type=int, default=1, help='Val interval per X epoch')

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
    pbar = tqdm(trainset, desc='==> Train', position=2)
    for image_data, target in pbar:
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss, conf_loss, prob_loss = 0, 0, 0

            # optimizing process for THREE targets:
            # 1) smaller / 2) medium / 3) large
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], params.strides,
                                          params.iou_loss_thresh, args.n_classes, i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
        #          "prob_loss: %4.2f   total_loss: %4.2f" %(params.global_steps, optimizer.lr.numpy(),
        #                                                   giou_loss, conf_loss,
        #                                                   prob_loss, total_loss))

        # update learning rate
        params.global_steps += 1
        lr = calc_lr(params.global_steps, params, args)
        optimizer.lr.assign(lr)

        pbar.update()
        pbar.set_postfix({
            'Total': '{:4.2f}'.format(total_loss),
            'GIoU': '{:4.2f}'.format(giou_loss),
            'Conf': '{:4.2f}'.format(conf_loss),
            'Prob': '{:4.2f}'.format(prob_loss),
            })

        # writing summary data
        if params.global_steps % args.log_interval == 0:
            with params.writer.as_default():
                tf.summary.scalar('lr', optimizer.lr, step=params.global_steps)
                tf.summary.scalar('loss/total_loss', total_loss, step=params.global_steps)
                tf.summary.scalar('loss/giou_loss', giou_loss, step=params.global_steps)
                tf.summary.scalar('loss/conf_loss', conf_loss, step=params.global_steps)
                tf.summary.scalar('loss/prob_loss', prob_loss, step=params.global_steps)

            params.writer.flush()


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
    params.global_steps = 0
    params.total_steps = args.n_epoch * steps_per_epoch
    params.warmup_steps = int(args.warmup * params.total_steps)

    input_size = 512
    input_tensor = tf.keras.layers.Input([input_size, input_size, 3])
    conv_tensors = YOLOv3(input_tensor, strides=params.strides, anchors=params.anchors,
                          num_classes=args.n_classes)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, params.strides, params.anchors, args.n_classes, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)

    # first_decay_steps = 1000
    # lr_decayed = cosine_decay_restarts(learning_rate, global_step, first_decay_steps)
    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tfa.optimizers.AdamW(
        weight_decay=args.wd, learning_rate=args.lr_init,
        beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps,
    )

    # if os.path.exists(params.log_dir):
    #     shutil.rmtree(params.log_dir)

    params.writer = tf.summary.create_file_writer(params.log_dir)

    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print([gpu.name for gpu in gpus])
    device = '/GPU:0' if gpus else '/CPU:0'

    pbar = tqdm(range(args.n_epoch), desc='==> Epoch', position=1)
    for epoch in pbar:
        with tf.device(device):
            train_step(model, trainset, optimizer, params, args)

        model.save_weights(params.ckpt_save_dir)
