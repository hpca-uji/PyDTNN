import inspect
import sys
import unittest
from copy import deepcopy

import numpy as np

from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.conv_2d_cpu import Conv2DCPU
from pydtnn.tests.common import verbose_test, D
from pydtnn.tests.common import Params, TestCase
from pydtnn.tests.conv2d_conv_gemm import Conv2DConvGemmTestCase as _Conv2DConvGemmTestCase
from pydtnn.utils.tensor import TensorFormat
from pydtnn.utils import print_with_header, random
from pydtnn.utils.initializers import glorot_uniform, zeros


class Conv2DConvGroupTestCase(_Conv2DConvGemmTestCase):
    """
    Tests that Conv2D with Depth+Pair leads to the same results than Conv2D Standard
    """

    @staticmethod
    def _get_layers(d: D, deconv=False, trans=False) -> tuple[Conv2DCPU, Conv2DCPU]:
        params = Params()
        params.tensor_format = TensorFormat.NHWC.upper()
        params.batch_size = d.b
        model = Model(**vars(params))
        model.mode = Model.Mode.TRAIN

        conv2d_depth = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                              grouping=Conv2D.Grouping.DEPTHWISE,
                              padding=(d.vpadding, d.hpadding),
                              stride=(d.vstride, d.hstride),
                              dilation=(d.vdilation, d.hdilation),
                              use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d_pair = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                             grouping=Conv2D.Grouping.POINTWISE,
                             padding=(d.vpadding, d.hpadding),
                             stride=(d.vstride, d.hstride),
                             dilation=(d.vdilation, d.hdilation),
                             use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        chain = ConcatenationBlock([
            conv2d_depth,
            conv2d_pair
        ])
        chain.set_backend(model._backend)
        chain.set_model(model)
        chain.initialize(prev_shape=(d.c, d.h, d.w))

        conv2d = Conv2D(nfilters=d.kn, filter_shape=(d.kh, d.kw),
                        grouping=Conv2D.Grouping.STANDARD,
                        padding=(d.vpadding, d.hpadding),
                        stride=(d.vstride, d.hstride),
                        dilation=(d.vdilation, d.hdilation),
                        use_bias=True, weights_initializer=glorot_uniform, biases_initializer=zeros)
        conv2d.set_backend(model._backend)
        conv2d.set_model(model)
        conv2d.initialize(prev_shape=(d.c, d.h, d.w))

        # Set the same initial weights and biases to both layers
        conv2d_depth.weights = conv2d.weights.copy()
        conv2d_depth.biases = conv2d.biases.copy()
        conv2d_pair.weights = conv2d.weights.copy()
        conv2d_pair.biases = conv2d.biases.copy()

        return conv2d, chain


if __name__ == '__main__':
    try:
        Conv2DCPU()
    except NameError:
        sys.exit(-1)
    unittest.main()
