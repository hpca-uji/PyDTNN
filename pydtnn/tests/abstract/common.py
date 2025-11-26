"""
Common methods and properties for various unitary tests
"""

import os
import sys
import unittest
import warnings

import numpy as np

from pydtnn.utils import random
from pydtnn.utils.tensor import TensorFormat


# @warning: must be a function, don't use a @property decorator
def verbose_test():
    """Returns True if unittest has been called with -v or --verbose options."""
    return '-v' in sys.argv or '--verbose' in sys.argv


class Params:
    def __init__(self) -> None:
        self.parallel = "sequential"
        self.dtype: np.dtype = np.dtype(np.float32)
        self.tensor_format = TensorFormat.NHWC.upper()
        self.conv_variant = "i2c"

        # model_name = os.environ.get("PYDTNN_TEST_MODEL", "simplecnn")
        # self.model_name = model_name

        self.dataset_name = "synthetic"
        self.synthetic_train_samples = "50000"
        self.synthetic_test_samples = "10000"
        self.synthetic_input_shape = "3,32,32"
        self.synthetic_output_shape = "10"


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        random.seed(0)
        warnings.simplefilter("error")

    def tearDown(self) -> None:
        warnings.resetwarnings()
        super().tearDown()


class D:
    def __init__(self, b=1, c=1, h=128, w=100, kn=1, kh=16, kw=10, vpadding=1, hpadding=1,
                 vstride=1, hstride=1, vdilation=1, hdilation=1):
        self.b = b  # Batch size
        self.c = c  # Channels per layer
        self.h = h  # Layers height
        self.w = w  # Layers width
        self.kn = kn  # Number of filters
        self.kh = kh  # Filters weights height
        self.kw = kw  # Filters weights width
        self.vpadding = vpadding  # Vertical padding
        self.hpadding = hpadding  # Horizontal padding
        self.vstride = vstride  # Vertical stride
        self.hstride = hstride  # Horizontal stride
        self.vdilation = vdilation  # Vertical dilation
        self.hdilation = hdilation  # Horizontal dilation

    @property
    def ho(self):
        return (self.h + 2 * self.vpadding - self.vdilation * (self.kh - 1) - 1) // self.vstride + 1

    @property
    def wo(self):
        return (self.w + 2 * self.hpadding - self.hdilation * (self.kw - 1) - 1) // self.hstride + 1

    @property
    def shape(self):
        return self.b, self.c, self.h, self.w

    def __repr__(self):
        return f"""\
x, weights, and y parameters:
  (b, c, h, w)    = {self.b} {self.c} {self.h} {self.w}
  (kn, c, kh, kw) = {self.kn} {self.c} {self.kh} {self.kw}
  (kn, b, ho, wo) = {self.kn} {self.b} {self.ho} {self.wo}
  padding         = {self.vpadding} {self.hpadding}
  stride          = {self.vstride} {self.hstride}
  dilation        = {self.vdilation} {self.hdilation}
"""


alexnet_layers = [
    # AlexNet Cifar
    D(64, 3, 32, 32, 64, 3, 3, 1, 1, 2, 2, 1, 1),
    D(64, 64, 8, 8, 192, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 192, 4, 4, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 256, 4, 4, 256, 3, 3, 1, 1, 1, 1, 1, 1),
    # AlexNet ImageNet
    D(64, 3, 227, 227, 96, 11, 11, 1, 1, 4, 4, 1, 1),
    D(64, 96, 27, 27, 256, 5, 5, 1, 1, 1, 1, 1, 1),
    D(64, 256, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 384, 3, 3, 1, 1, 1, 1, 1, 1),
    D(64, 384, 13, 13, 256, 3, 3, 1, 1, 1, 1, 1, 1),
]

alexnet_backward_layers = []
for layer in alexnet_layers:
    # w <- y (kn * b * ho * wo)
    alexnet_backward_layers.append(D(layer.c, layer.b, layer.h, layer.w, layer.kn, layer.ho, layer.wo,
                                     layer.vpadding, layer.hpadding, layer.vstride, layer.hstride))

alexnet_all_layers = alexnet_layers + alexnet_backward_layers
