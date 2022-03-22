from torch import nn
from typing import Tuple


def vgg_block(n_convs: int, in_ch: int, out_ch: int) -> nn.Sequential:
    layers = list()
    for _ in range(n_convs):
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_ch = out_ch
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(
    conv_arch: Tuple[Tuple[int, int], ...],
    in_channels: int = 1,
    in_dims: Tuple[int, int] = (244, 244),
    n_classes: int = 10,
) -> nn.Sequential:
    conv_blks = []
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
        in_dims = (in_dims[0] // 2, in_dims[1] // 2)

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        # The fully-connected part
        nn.Linear(out_channels * in_dims[0] * in_dims[1], 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, n_classes)
    )
