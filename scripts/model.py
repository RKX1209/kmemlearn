#!/usr/bin/env python

"""
simple MLP
"""
import pdb
import six
import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):

    """A VGG-style network (shrinked)
    copied from chainer example cifar-10
    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(32, (9, 1), pad=(4,0))
            self.block1_2 = Block(32, (1, 9), pad=(0,4))
            self.block2_1 = Block(48, (9, 1), pad=(4,0))
            self.block2_2 = Block(48, (1, 9), pad=(0,4))
            self.block3_1 = Block(64, (9, 1), pad=(4,0))
            self.block3_2 = Block(64, (1, 9), pad=(0,4))
            self.att = L.Linear(None, 20, nobias=True)
            self.fc1 = L.Linear(None, 64, nobias=True)
            self.bn_fc1 = L.BatchNormalization(64)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h1 = self.block1_1(x)
        h1 = F.dropout(h1, ratio=0.3)
        h2 = self.block1_2(x) # split network
        h2 = F.dropout(h2, ratio=0.3)

        # 128 channel blocks:
        h1 = self.block2_1(h1)
        h1 = F.dropout(h1, ratio=0.4)
        h2 = self.block2_2(h2)
        h2 = F.dropout(h2, ratio=0.4)
        #h1 = F.max_pooling_2d(h1, ksize=2, stride=2)
        #h2 = F.max_pooling_2d(h2, ksize=2, stride=2)

        # 256 channel blocks:
        h1 = self.block3_1(h1)
        h1 = F.dropout(h1, ratio=0.4)
        h2 = self.block3_2(h2)
        h2 = F.dropout(h2, ratio=0.4)
        #h1 = F.max_pooling_2d(h1, ksize=2, stride=2)
        #h2 = F.max_pooling_2d(h2, ksize=2, stride=2)
        h = F.concat((h1, h2), axis=1)
        #h = h2
        #h = h1
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)
