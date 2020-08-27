import tensorflow as tf

from .common import Conv, ResBlock


def darknet53(input_data):
    output = Conv(input_data, (3, 3,  3,  32))
    output = Conv(output, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        output = ResBlock(output,  64,  32, 64)

    output = Conv(output, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        output = ResBlock(output, 128,  64, 128)

    output = Conv(output, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        output = ResBlock(output, 256, 128, 256)

    route_1 = output
    output = Conv(output, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        output = ResBlock(output, 512, 256, 512)

    route_2 = output
    output = Conv(output, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        output = ResBlock(output, 1024, 512, 1024)

    return route_1, route_2, output
