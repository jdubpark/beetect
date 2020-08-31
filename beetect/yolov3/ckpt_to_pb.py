import argparse
import os

import numpy as np
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from beetect.models.yolov3.model import YOLOv3, decode
from beetect.yolov3.weights_to_ckpt import load_weights


parser = argparse.ArgumentParser(description='Convert pretrained (converted) darknet tf checkpoint to .pb')
parser.add_argument('--num_classes', type=int, default=2)
# parser.add_argument('--weight_file', '-C', type=str)
# parser.add_argument('--ckpt_file', '-C', type=str)
parser.add_argument('--ckpt_path', '-C', type=str)
parser.add_argument('--save_path', '-O', type=str)
parser.add_argument('--pb_name', '-N', type=str, default='frozen_yolov3.pb')

parser.add_argument('--lr_init', type=float, default=5e-4) # 1e-3 explodes for adamw (sgdw untested)
parser.add_argument('--lr_end', type=float, default=1e-6)
parser.add_argument('--decay', dest='wd', type=float, default=5e-5)
parser.add_argument('--eps', default=1e-6, type=float) # for adamw
parser.add_argument('--beta1', default=0.9, type=float) # "
parser.add_argument('--beta2', default=0.999, type=float) # "
parser.add_argument('--momentum', default=0.9, type=float) # for sgdw


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name='')

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    # print("-" * 50)
    # print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    # if print_graph == True:
    #     for layer in layers:
    #         print(layer)
    # print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


if __name__ == '__main__':
    args = parser.parse_args()

    save_pb_path = os.path.join(args.save_path, args.pb_name)

    img_size = 512
    strides = [8, 16, 32]
    anchors = [1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875]

    # reconstruct model
    input_data = tf.keras.layers.Input([img_size, img_size, 3])
    conv_tensors = YOLOv3(input_data, num_classes=args.num_classes)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, strides, anchors, args.num_classes, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_data, output_tensors)
    # model.summary()

    # reconstruct ckpt
    optimizer = tfa.optimizers.AdamW(
        weight_decay=args.wd, learning_rate=args.lr_init,
        beta_1=args.beta1, beta_2=args.beta2, epsilon=args.eps)
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, args.ckpt_path, max_to_keep=3)

    # load ckpt
    ckpt.restore(manager.latest_checkpoint)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    ops = frozen_func.graph.get_operations()
    print('-' * 20)
    print('Frozen model layers: ')
    for op in ops:
        print('\t', op.name)

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=args.save_path,
                      name=args.pb_name,
                      as_text=False)

    print(f'Checkpoint converted to "{save_pb_path}"')


    # output_node_names = ['input/input_data', 'pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2']
    # tf.compat.v1.disable_eager_execution() # for placeholders
    # with tf.name_scope('input'):
    #     input_data = tf.placeholder(dtype=tf.float32,shape=(None, img_size, img_size, 3), name='input_data')
    #
    # model = YOLOv3(input_data, num_classes=args.num_classes)
    # # => [conv_sbbox, conv_mbbox, conv_lbbox]
    # print(model)
    #
    # print(tf.variables)
    #
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     saver = tf.train.Saver(tf.global_variables())
    #     saver.restore(sess, args.ckpt_file)
    #
    #     converted_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess, input_graph_def=sess.graph.as_graph_def(),
    #         output_node_names=output_node_names)
    #
    #     pb_file = os.path.join(args.save_path, 'frozen_darknet_yolov3_model.pb')
    #     with tf.gfile.GFile(pb_file, 'wb') as f:
    #         f.write(converted_graph_def.SerializeToString())



    # load_ops = load_weights(tf.global_variables(), args.weight_file)
    #
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    #     sess.run(load_ops)
    #     output_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess,
    #         tf.get_default_graph().as_graph_def(),
    #         output_node_names=output_node_names
    #     )
    #
    #     output_graph = os.path.join(args.save_path, 'frozen_darknet_yolov3_model.pb')
    #     with tf.gfile.GFile(output_graph, 'wb') as f:
    #         f.write(output_graph_def.SerializeToString())
    #
    #     print('{} ops written to {}.'.format(len(output_graph_def.node), output_graph))
