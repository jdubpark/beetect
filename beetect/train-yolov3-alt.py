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
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau,
#     EarlyStopping,
#     ModelCheckpoint,
#     TensorBoard
# )

import beetect.models.yolov3_alt.dataset as dataset
from beetect.models.yolov3_alt.models import YOLOv3, YOLOLoss
from beetect.models.yolov3_alt.utils import freeze_all


parser = argparse.ArgumentParser(description='Beetect YOLOv3 TF2 training')

# dirs
parser.add_argument('--dump_dir', '-O', type=str)
parser.add_argument('--dataset_dir', '-D', type=str,
                    help='Path to dataset in TFRecord file format. Use convert_to_tfrecord.py for format conversion.')
parser.add_argument('--class_path', '-C', type=str,
                    help='Path to .txt file containing classes')
parser.add_argument('--weight_path', '-W', type=str)

# training
# parser.add_argument('--mode', '-mode', type=str, default='eager', help=''+
#                     '[DEFAULT] eager: tf.GradientTape'+
#                     'fit: model.fit\n'+
#                     'fit_eagerly: model.fit(run_eagerly=True)')
parser.add_argument('--num_epochs', '-epoch', type=int, default=100)
parser.add_argument('--batch_size', '-batch', type=int, default=8)
parser.add_argument('--image_size', type=int, default=416) # 512
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--weights_num_classes', type=int, default=80, help='Num classes of loaded weight, default to COCO')
parser.add_argument('--optim', type=str, default='adamw', help='Optimizer')
parser.add_argument('--warmup', type=float, default=5, help='Number of epoch for warmup')
parser.add_argument('--transfer', type=str, default='darknet', help=''+
                    'None: Train from scratch\n'+
                    '[DEFAULT] darknet: Transfer darknet\n'+
                    'no_output: Transfer all but output\n'+
                    'frozen: Transfer and freeze all\n'+
                    'fine_tune: Transfer all and freeze darknet only')

# hyperparams
parser.add_argument('--lr_init', type=float, default=1e-3) # 5e-4
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
parser.add_argument('--ckpt_max_keep', type=int, default=10, help='Maximum number of checkpoints to keep while saving new')

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


def train_step(model, train_dataset, optimizer, loss_fns, params, args):
    acc_loss = 0
    local_steps = 0

    pbar = tqdm(range(1, params.dataset_len+1), desc='==> Train', position=1)
    for idx, (image_data, target) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = model(image_data, training=True)
            reg_loss = tf.reduce_sum(model.losses)
            pred_loss = []

            for output, label, loss_fn in zip(outputs, target, loss_fns):
                pred_loss.append(loss_fn(label, output))

            total_loss = tf.reduce_sum(pred_loss) + reg_loss
            grads = tape.gradient(total_loss, model.trainable_variables)

        acc_loss += total_loss
        mean_loss = acc_loss / local_steps

        lr = calc_lr(params.global_steps, params, args)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        optimizer.lr.assign(lr)

        pbar.update()
        # pbar.set_description('lr {:.6f}'.format(lr))
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

        # at last of each batch
        params.global_steps += 1
        local_steps += 1

    # at last of each epoch
    pbar.close()


def val_step():
    # for batch, (images, labels) in enumerate(val_dataset):
    #     outputs = model(images)
    #     regularization_loss = tf.reduce_sum(model.losses)
    #     pred_loss = []
    #     for output, label, loss_fn in zip(outputs, labels, loss):
    #         pred_loss.append(loss_fn(label, output))
    #     total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    #
    #     logging.info("{}_val_{}, {}, {}".format(
    #         epoch, batch, total_loss.numpy(),
    #         list(map(lambda x: np.sum(x.numpy()), pred_loss))))
    #     avg_val_loss.update_state(total_loss)
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    params = Map({})

    params.ckpt_save_dir = os.path.join(args.dump_dir, 'checkpoints', 'yolov3')
    params.log_dir = os.path.join(args.dump_dir, 'logs', 'yolov3')

    # params.iou_loss_thresh = 0.5
    # params.strides = [8, 16, 32]
    params.anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                               (59, 119), (116, 90), (156, 198), (373, 326)],
                              np.float32) / args.image_size
    params.anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    gpus = tf.config.experimental.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'

    for dir in [params.ckpt_save_dir, params.log_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)


    train_dataset = dataset.load_tfrecord_dataset(
        args.dataset_dir, args.class_path, args.image_size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, args.image_size),
        dataset.transform_targets(y, params.anchors, params.anchor_masks, args.image_size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    dataset_len = 0
    # for record in tf.compat.v1.python_io.tf_record_iterator(args.dataset_dir):
    for record in tf.data.TFRecordDataset(args.dataset_dir):
        dataset_len += 1

    steps_per_epoch = dataset_len
    params.dataset_len = dataset_len
    params.global_steps = 1 # init at 1
    params.total_steps = args.num_epochs * steps_per_epoch
    # params.warmup_steps = int(args.warmup * params.total_steps) # for percentage
    params.warmup_steps = args.warmup * steps_per_epoch

    # create YOLOv3 model
    model = YOLOv3(size=args.image_size, num_classes=args.num_classes)()

    # check if transfer learning or not and load accordingly
    ckpt_save_dir_ext = 'from_scratch'
    if args.transfer is not None:
        assert args.transfer in ['darknet', 'no_output', 'frozen', 'fine_tune']
        assert args.weight_path is not None

        ckpt_save_dir_ext = 'transfer_'+args.transfer

        if args.transfer in ['darknet', 'no_output']:
            # Darknet transfer is a special case that works with incompatible number of classes
            # (below) Reset top layers, assume preloaded is coco
            # assume
            model_pretrained = YOLOv3(args.image_size,
                                      num_classes=args.weights_num_classes,
                                      training=True)()
            model_pretrained.load_weights(args.weight_path)
            # model_pretrained.load_weights(args.weight_path).expect_partial()

            if args.transfer == 'darknet':
                model.get_layer('yolo_darknet').set_weights(
                    model_pretrained.get_layer('yolo_darknet').get_weights())
                freeze_all(model.get_layer('yolo_darknet'))

            elif args.transfer == 'no_output':
                for l in model.layers:
                    if not l.name.startswith('yolo_output'):
                        l.set_weights(model_pretrained.get_layer(
                            l.name).get_weights())
                        freeze_all(l)
        else:
            # All other transfer require matching classes
            model.load_weights(args.weight_path)
            if args.transfer == 'fine_tune':
                # freeze darknet and fine tune other layers
                darknet = model.get_layer('yolo_darknet')
                freeze_all(darknet)
            elif args.transfer == 'frozen':
                # freeze everything
                freeze_all(model)

    # modify checkpoint save dir to indicate whether trasnfer learning or not
    params.ckpt_save_dir = os.path.join(params.ckpt_save_dir, ckpt_save_dir_ext)
    if not os.path.isdir(params.ckpt_save_dir):
        os.makedirs(params.ckpt_save_dir, exist_ok=True)

    # set up optimizers
    assert args.optim in ['adam', 'adamw', 'sgdw']
    if args.optim == 'adam':
        # use trackable optim for ckpt manager
        optimizer = tf.compat.v2.keras.optimizers.Adam(learning_rate=args.lr_init)
    elif args.optim == 'adamw':
        optimizer = tfa.optimizers.AdamW(
            weight_decay=args.wd, learning_rate=args.lr_init,
            beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps)
    elif args.optim == 'sgdw':
        optimizer = tfa.optimizers.SGDW(weight_decay=args.wd, learningrate=args.lr_init, momentum=args.momentum)

    # set up loss for each anchor w/ anchor mask
    loss_fns = [YOLOLoss(params.anchors[mask], num_classes=args.num_classes)
                for mask in params.anchor_masks]

    now = datetime.datetime.now()
    writer_log_dir = os.path.join(params.log_dir, now.strftime("%Y-%m-%d_%H-%M-%S"))
    params.writer = tf.summary.create_file_writer(writer_log_dir)

    # all kwargs to Checkpoint should be trackable
    ckpt = tf.train.Checkpoint(
        epoch=tf.Variable(1), optimizer=optimizer,
        model=model)
    manager = tf.train.CheckpointManager(ckpt, params.ckpt_save_dir, max_to_keep=args.ckpt_max_keep)

    # eager enabled by default in tf2. use that for now
    # assert args.mode in ['eager', 'fit', 'fit_eagerly']
    # if args.mode == 'eager':
    best_loss = 1e5
    pbar = tqdm(range(1, args.num_epochs+1), desc='==> Epoch', position=0)
    for epoch in pbar:
        with tf.device(device):
            mean_loss = train_step(model, train_dataset, optimizer, loss_fns, params, args)

        # val_step()

        ckpt.epoch.assign_add(1)
        save_epoch = epoch % args.ckpt_interval == 0

        save_path = manager.save(checkpoint_number=epoch)
        print('Saved checkpoint for epoch {} at "{}"'.format(epoch, save_path))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_ckpt_file = os.path.join(params.ckpt_save_dir, 'best_epoch')
            for ext in ['.data-00000-of-00001', '.index', '.meta']:
                if not os.path.isfile(save_path+ext):
                    continue

                shutil.copyfile(save_path+ext, best_ckpt_file+ext)

    # else:
    #     model.compile(optimizer=optimizer, loss=loss_fns,
    #                   run_eagerly=(args.mode == 'fit_eagerly'))
    #
    #     callbacks = [
    #         ReduceLROnPlateau(verbose=1),
    #         EarlyStopping(patience=3, verbose=1),
    #         ModelCheckpoint(save_path,
    #                         verbose=1, save_weights_only=True)
    #         TensorBoard(log_dir=args.writer_log_dir)
    #     ]
    #
    #     history = model.fit(train_dataset,
    #                         epochs=args.num_epochs,
    #                         callbacks=callbacks)
    #                         #validation_data=val_dataset
