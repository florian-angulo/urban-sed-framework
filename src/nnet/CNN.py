import torch
import torch.nn as nn

#! TO DO : Replace permute with einops

"""
? GLU à remplacer par implémentation Pytorch ?
"""


class GLU(nn.Module):
    def __init__(self, in_size):
        """Initialize a Gated Linear Unit
        Args:
            in_size (int): size of input
        """
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_size, in_size)

    def forward(self, x):
        """Apply GLU on a given tensor
        Args:
            x (tensor)
        Returns:
            tensor
        """
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        # ? In the general case, sigmoid is applied to another Linear output from x
        sig = self.sigmoid(lin)
        return lin * sig


class ContextGating(nn.Module):
    def __init__(self, in_size):
        """Initialize a Context Gating Unit
        Args:
            in_size (int): size of input
        """
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(in_size, in_size)

    def forward(self, x):
        """Apply Context Gating on a given tensor
        Args:
            x (tensor)
        Returns:
            (tensor)
        """
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        return x * sig


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **transformer_kwargs
    ):
        """Initialization of a CNN

        Args:
            n_in_channel (int): number of input channels
            activation (str, optional): activation function. Defaults to "Relu".
            conv_dropout (int, optional): dropout. Defaults to 0.
            kernel_size (list, optional): kernel sizes. Defaults to [3,3,3].
            padding (list, optional): padding. Defaults to [1,1,1].
            stride (list, optional): stride. Defaults to [1,1,1].
            nb_filters (list, optional): number of filters. Defaults to [64,64,64].
            pooling (list, optional): time and frequency pooling. Defaults to [(1, 4), (1, 4), (1, 4)].
            normalization (str, optional): "batch" for BatchNormalization or "layer" for LayerNormalization. Defaults to "batch".
        """
        super().__init__()
        cnn = nn.Sequential()

        def conv(i):
            # special case for the first block
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]

            # 2D Convolution layer
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i], bias=False),
            )

            # Normalization layer
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))
            else:
                raise ValueError(
                    "Unrecognized normalization : {0}".format(normalization)
                )

            # Activation function
            if activation.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activation.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activation.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activation.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))
            else:
                raise ValueError("Unrecognized activation : {0}".format(activation))

            # Dropout
            if conv_dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(conv_dropout))

            # Average Pooling
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))

        # Assembling each block into the Sequential module
        for i in range(len(nb_filters)):
            conv(i)
        self.cnn = cnn
        self.nb_filters = nb_filters

    def forward(self, x):
        """Apply the CNN to input tensor x

        Args:
            x (tensor): input of expected size (n_channels=128,n_frames=862,freq_bin=64)

        Returns:
            Tensor : output embedding
        """
        return self.cnn(x)
