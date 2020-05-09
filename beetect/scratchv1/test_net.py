import matplotlib.pyplot as plt

import numpy as np
import torch
from beetect.scratchv1 import resnet50
from torch import nn


def main():
    model = resnet50()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    # checkpoint = torch.load('./resnet18_model.pt', map_location=map_location)
    # model.load_state_dict(checkpoint['state_dict'])
    # arch = checkpoint['arch']
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    #
    # params = model.parameters()

    print(model)
    # print(list(params))

    # analyze_weights(model)


def analyze_weights(model):
    backbone = model.backbone

    # print(backbone)

    # index starts at 0, so -1
    layer_n = lambda n: backbone[n-1]

    # print(layer_n(1).weight.data)
    # print(layer_n(7)[0].conv1.weight)

    weights = get_weights(layer_n(7))
    # print(weights)

    plot_weights(weights)


def get_weights(layer):
    """
    Arguments:
        layer (nn.Sequential): nn.Sequential class

    Returns:
        weight (tensor): weight object from nn.Sequential {data, grad_fn}
    """
    if isinstance(layer, nn.Conv2d):
        # conv2 at top nn.Sequential
        return layer.weight
    elif isinstance(layer, nn.Sequential) and layer[0].conv1 is not None:
        # conv2 within sub-level nn.Sequential
        return layer[0].conv1.weight
    else:
        print("Layer doesn't contain Conv2d layer to get weights from.")
        return None


def plot_weights(weight, single_channel=True, collated=False,
                 max_show=12, max_depth=6, max_num=6):
    """
    FROM: https://github.com/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb

    modified: passing weight directly
    """

    #getting the weight tensor data
    weight_tensor = weight.data

    if single_channel:
        if collated:
            plot_filters_single_channel_big(weight_tensor, max_show=max_show)
        else:
            plot_filters_single_channel(weight_tensor, max_depth=max_depth, max_num=max_num)

    else:
        if weight_tensor.shape[1] == 3:
            plot_filters_multi_channel(weight_tensor, max_show=max_show)
        else:
            print('Can only plot weights with three channels with single channel = False')


def plot_filters_single_channel_big(t, max_show):
    #setting the rows and columns
    nrows = t.shape[0]*t.shape[2]
    ncols = t.shape[1]*t.shape[3]


    npimg = np.array(t.numpy(), np.float32)
    npimg = npimg.transpose((0, 2, 1, 3))
    npimg = npimg.ravel().reshape(nrows, ncols)

    npimg = npimg.T

    fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))
    imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)


def plot_filters_single_channel(t, max_depth, max_num):
    trunc = t[:max_depth, :max_num]
    # same as:
    # trunc = torch.narrow(0, 0, max_depth)
    # trunc = torch.narrow(1, 0, max_num)
    # print(trunc.shape)

    # kernel depth and number of kernels (t.shape[0] and t.shape[1])
    k_depth = max_depth if max_depth <= t.shape[0] else t.shape[0]
    num_k = max_num if max_num <= t.shape[1] else t.shape[1]

    # kernels depth * number of kernels
    nplots = k_depth * num_k
    ncols = 6 # might use num_k instead

    nrows = 1 + nplots // ncols
    # convert tensor to numpy image
    npimg = np.array(trunc.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(k_depth):
        for j in range(num_k):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(trunc[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.show()


def plot_filters_multi_channel(t, max_show, num_cols=6):

    # get the number of kernals
    num_kernels = max_show if max_show is not None else t.shape[0]

    # define number of columns for subplots
    num_cols = num_cols
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols,num_rows))

    # looping through all the kernels
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)

        #for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.savefig('myimage.png', dpi=100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
